#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_accessors.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

/*============================================================================
   Oseen–Darcy Problem (with Beavers–Joseph–Saffman (BJS) interface coupling)

   Domain: Ω = Ω_O ∪ Ω_D ⊂ ℝ^d, where:
       - Ω_O: fluid region (Oseen)
       - Ω_D: porous medium region (Darcy)
       - Γ = ∂Ω_O ∩ ∂Ω_D: interface between the two regions

   Unknowns: (u, p_O, p_D)
       - u  : velocity in Ω_O
       - p_O: pressure in Ω_O
       - p_D: pressure in Ω_D

   Governing equations:

   In Ω_O (Oseen region):
       -ν Δu + (β · ∇)u + ∇p_O = f_O          (momentum balance)
                          ∇ · u = 0          (mass conservation)

   In Ω_D (Darcy region):
       -∇ · (K ∇p_D) = f_D           (Darcy’s law)

   Interface conditions on Γ:
       1. u · n = −(K ∇p_D) · n                     (normal flux continuity)
       2. (−p_O I + ν ∇u) · n = −p_D n              (normal stress balance)
       3. (ν / G) (u · τ) = τ · (−p_O I + ν ∇u) · n   (BJS condition)

   Boundary conditions:
       - On ∂Ω_O \ Γ:    u = gD_O                    (Dirichlet)
       - On ∂Ω_D \ Γ:    −(K ∇p_D) · n = g_D          (Neumann)

   Integral constraint:
       ∫_Ω (p_O + p_D) dx = 0

   Parameters:
       - ν: fluid viscosity (constant)
       - β: prescribed convection (advection) velocity field (BetaFunction)
       - K: permeability tensor (symmetric, positive-definite)
       - n: unit normal vector
       - τ: unit tangential vector on Γ
       - α_BJ: Beavers–Joseph coefficient (from experiment)
       - G = √[ν (τ · K · τ)] / α_BJ   (BJS slip coefficient)
============================================================================*/

namespace OseenDarcyNamespace
{
  using namespace dealii;

  enum class MeshType
  {
    Mesh9,
    Mesh10,
    Mesh11,
    Mesh12
  };
  std::string
  Mesh_str(MeshType type)
  {
    static constexpr std::array<std::string_view, 4> names = {
      {"Mesh9", "Mesh10", "Mesh11", "Mesh12"}};

    return std::string(names[static_cast<size_t>(type)]);
  }

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution(const double one_over_G = 0.5 / std::sqrt(0.1))
      : Function<dim>(dim + 2)
      , one_over_G(one_over_G)
    {}

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>>   &value_list) const override;

    virtual void
    vector_gradient_list(
      const std::vector<Point<dim>>            &points,
      std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const override;

  private:
    const double one_over_G;
    const double mean_pressure = -28. / 45. - M_PI * 16. / 3. / 25. / 25. / 25.;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>>   &value_list) const
  {
    using std::exp;
    using std::pow;
    using std::sqrt;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        double r = sqrt(x * x + y * y);
        double g = exp(one_over_G * r);

        value_list[i].reinit(4);
        value_list[i][0] = -g * y;
        value_list[i][1] = g * x;
        value_list[i][2] = -pow(r, 4) - mean_pressure;
        value_list[i][3] = -8. / 25. * r * r + pow(r, 4) - mean_pressure;
      }
  }

  template <int dim>
  void
  ExactSolution<dim>::vector_gradient_list(
    const std::vector<Point<dim>>            &points,
    std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const
  {
    using std::exp;
    using std::pow;
    using std::sqrt;

    AssertDimension(points.size(), gradient_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        double r = sqrt(x * x + y * y);
        double g = exp(one_over_G * r);

        gradient_list[i].resize(4);
        gradient_list[i][0][0] = -y * one_over_G * g * x / r;
        gradient_list[i][0][1] = -y * one_over_G * g * y / r - g;

        gradient_list[i][1][0] = x * one_over_G * g * x / r + g;
        gradient_list[i][1][1] = x * one_over_G * g * y / r;

        gradient_list[i][2][0] = -4. * pow(r, 2) * x;
        gradient_list[i][2][1] = -4. * pow(r, 2) * y;

        gradient_list[i][3][0] = (-16. / 25. + 4. * pow(r, 2)) * x;
        gradient_list[i][3][1] = (-16. / 25. + 4. * pow(r, 2)) * y;
      }
  }
  /*============================================================================
     RightHandSide_O and RightHandSide_D

     The right-hand side functions for the Oseen and Darcy problems.
  ============================================================================*/
  template <int dim>
  class RightHandSide_O : public TensorFunction<1, dim, double>
  {
  public:
    RightHandSide_O(const double one_over_G   = 0.5 / std::sqrt(0.1),
                    const double viscosity_nu = 0.1)
      : TensorFunction<1, dim, double>()
      , one_over_G(one_over_G)
      , viscosity_nu(viscosity_nu)
    {}

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    const double one_over_G;
    const double viscosity_nu;
  };

  template <int dim>
  void
  RightHandSide_O<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::exp;
    using std::pow;
    using std::sqrt;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        double r = sqrt(x * x + y * y);
        double g = exp(one_over_G * r);

        value_list[i][0] = viscosity_nu * (2 * one_over_G * g * y / r +
                                           y * one_over_G * one_over_G * g +
                                           y * one_over_G * g / r) -
                           4. * pow(r, 2) * x - x * g * g;
        value_list[i][1] = -viscosity_nu * (2 * one_over_G * g * x / r +
                                            x * one_over_G * one_over_G * g +
                                            x * one_over_G * g / r) -
                           4. * pow(r, 2) * y - y * g * g;
      }
  }

  template <int dim>
  class RightHandSide_D : public Function<dim>
  {
  public:
    RightHandSide_D()
      : Function<dim>()
    {}

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double>           &value_list,
               const unsigned int /*component*/) const override;
  };

  template <int dim>
  void
  RightHandSide_D<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double>           &value_list,
                                   const unsigned int /*component*/) const
  {
    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        value_list[i] = 32. / 25. - 16. * (x * x + y * y);
      }
  }
  /*============================================================================
     BoundaryDirichlet_O
 ============================================================================*/
  template <int dim>
  class BoundaryDirichlet_O : public TensorFunction<1, dim, double>
  {
  public:
    BoundaryDirichlet_O(const double one_over_G = 0.5 / std::sqrt(0.1))
      : TensorFunction<1, dim, double>()
      , one_over_G(one_over_G)
    {}

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    const double one_over_G;
  };

  template <int dim>
  void
  BoundaryDirichlet_O<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::exp;
    using std::sqrt;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const double x = points[i][0];
        const double y = points[i][1];

        double g = exp(one_over_G * sqrt(x * x + y * y));

        // left + right + top + bottom boundaries
        value_list[i][0] = -g * y;
        value_list[i][1] = g * x;
      }
  }

  template <int dim>
  class BetaFunction : public TensorFunction<1, dim, double>
  {
  public:
    BetaFunction(const double one_over_G = 0.5 / std::sqrt(0.1))
      : TensorFunction<1, dim, double>()
      , one_over_G(one_over_G)
    {}

    void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>>   &value_list) const override;

  private:
    const double one_over_G;
  };

  template <int dim>
  void
  BetaFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<Tensor<1, dim>>   &value_list) const
  {
    using std::exp;
    using std::sqrt;

    AssertDimension(points.size(), value_list.size());

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        const auto  &p = points[i];
        const double x = p[0];
        const double y = p[1];

        const double r = sqrt(x * x + y * y);
        const double g = exp(one_over_G * r);

        value_list[i][0] = -g * y;
        value_list[i][1] = g * x;
      }
  }

  enum class PartitionerType
  {
    metis,
    rtree
  };

  /*============================================================================
     The OseenDarcyProblem class below solves a coupled Oseen–Darcy flow
     using the Interior Penalty Discontinuous Galerkin (IPDG) method with a
     Beavers–Joseph–Saffman (BJS) interface condition.
  ============================================================================*/
  template <int dim>
  class OseenDarcyProblem
  {
  public:
    OseenDarcyProblem(const unsigned int degree_velocities = 2,
                      const unsigned int degree_pressure_O = 1,
                      const unsigned int degree_pressure_D = 1,
                      const unsigned int extraction_level  = 2,
                      const MeshType     mesh_type         = MeshType::Mesh9);

    // Run the simulation
    void
    run();

    // Get error norms
    double
    get_error_velocity_L2() const;
    double
    get_error_velocity_H1() const;
    double
    get_error_pressure_O() const;
    double
    get_error_pressure_D_L2() const;
    double
    get_error_pressure_D_semiH1() const;
    double
    get_error_velocity_global() const;
    double
    get_error_pressure_global() const;

    // Get number of degrees of freedom
    unsigned int
    get_n_dofs() const;

    unsigned int
    get_n_polytopes() const;

  private:
    static bool
    polytope_is_in_Oseen_domain(
      const typename AgglomerationHandler<dim>::agglomeration_iterator
        &polytope);
    static bool
    polytope_is_in_Darcy_domain(
      const typename AgglomerationHandler<dim>::agglomeration_iterator
        &polytope);

    // Grid and agglomeration setup
    void
    make_base_grid();
    void
    make_agglo_grid();
    void
    set_active_fe_indices();
    void
    setup_agglomeration();

    // Assemble system matrix and solve linear system
    void
    assemble_system();
    void
    solve();

    // Constraint handling
    void
    zero_pressure_dof_constraint();
    void
    mean_pressure_to_zero();

    // Post-processing
    void
    compute_errors();
    void
    output_results(unsigned int n_subdomains) const;

    // Degrees of finite elements (velocity and pressure)
    const unsigned int degree_velocities;
    const unsigned int degree_pressure_O;
    const unsigned int degree_pressure_D;
    const unsigned int extraction_level;

    const std::string example_name = "example4";
    const MeshType mesh_type;
    std::string    output_prefix;

    PartitionerType partitioner_type;
    bool            agglo_switch;
    unsigned int    n_subdomains; // Number of subdomains after agglomeration.

    // Physical parameters and domain info
    const double         viscosity_nu        = 1.;
    const double         permeability_scalar = 1.0;
    const Tensor<2, dim> permeability_K =
      permeability_scalar * unit_symmetric_tensor<dim>();
    const double alpha_BJ = 0.5; // Beavers–Joseph coefficient
    const double one_over_G =
      alpha_BJ / std::sqrt(viscosity_nu) /
      std::sqrt(permeability_scalar); // 1/G = α_BJ / √ν / √(τ · K · τ)
    const double nu_over_G =
      alpha_BJ * std::sqrt(viscosity_nu) /
      std::sqrt(permeability_scalar); // ν / G = α_BJ * √ν / √(τ · K · τ)
    double       domain_area = 4.;
    unsigned int num_domain; // Number of distinct domains.
                             // Agglomeration is restricted within each domain,
                             // allowing curved interfaces between different
                             // domains to be preserved.

    Triangulation<dim>            triangulation;
    std::unique_ptr<Mapping<dim>> mapping;

    hp::FECollection<dim>    fe_collection;
    hp::QCollection<dim>     q_collection;
    hp::QCollection<dim - 1> face_q_collection;

    AffineConstraints<double> constraints;

    std::unique_ptr<AgglomerationHandler<dim>> agglo_handler;
    std::unique_ptr<GridTools::Cache<dim>>     cached_tria;

    SparsityPattern      sparsity;
    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       system_rhs;

    std::unique_ptr<const Function<dim>>                  exact_solution;
    std::unique_ptr<const TensorFunction<1, dim, double>> rhs_function_O;
    std::unique_ptr<const Function<dim>>                  rhs_function_D;
    std::unique_ptr<const TensorFunction<1, dim, double>> bcDirichlet_O;
    std::unique_ptr<const TensorFunction<1, dim, double>> beta_function;

    static constexpr double penalty_constant_v   = 40.0;
    static constexpr double penalty_constant_p_O = 1.0;
    static constexpr double penalty_constant_p_D = 10.0;

    Vector<double> interpolated_solution;

    double error_velocity_L2       = 0.;
    double error_velocity_H1       = 0.;
    double error_pressure_O        = 0.;
    double error_pressure_D_L2     = 0.;
    double error_pressure_D_semiH1 = 0.;
    double error_velocity_global   = 0.;
    double error_pressure_global   = 0.;
  };

  template <int dim>
  OseenDarcyProblem<dim>::OseenDarcyProblem(
    const unsigned int degree_velocities,
    const unsigned int degree_pressure_O,
    const unsigned int degree_pressure_D,
    const unsigned int extraction_level,
    const MeshType     mesh_type)
    : degree_velocities(degree_velocities)
    , degree_pressure_O(degree_pressure_O)
    , degree_pressure_D(degree_pressure_D)
    , extraction_level(extraction_level)
    , mesh_type(mesh_type)
    , triangulation(Triangulation<dim>::maximum_smoothing)
  {
    FESystem<dim> stokes_fe(FE_AggloDGP<dim>(degree_velocities) ^ dim,
                            FE_AggloDGP<dim>(degree_pressure_O),
                            FE_Nothing<dim>());
    FESystem<dim> darcy_fe(FE_Nothing<dim>() ^ dim,
                           FE_Nothing<dim>(),
                           FE_AggloDGP<dim>(degree_pressure_D));
    fe_collection.push_back(stokes_fe);
    fe_collection.push_back(darcy_fe);

    const QGauss<dim>     quadrature_O(degree_velocities);
    const QGauss<dim>     quadrature_D(degree_pressure_D);
    const QGauss<dim - 1> face_quadrature_O(degree_velocities + 1);
    const QGauss<dim - 1> face_quadrature_D(degree_pressure_D + 1);

    q_collection.push_back(quadrature_O);
    q_collection.push_back(quadrature_D);
    face_q_collection.push_back(face_quadrature_O);
    face_q_collection.push_back(face_quadrature_D);

    exact_solution = std::make_unique<const ExactSolution<dim>>(one_over_G);
    rhs_function_O =
      std::make_unique<const RightHandSide_O<dim>>(one_over_G, viscosity_nu);
    rhs_function_D = std::make_unique<const RightHandSide_D<dim>>();
    bcDirichlet_O =
      std::make_unique<const BoundaryDirichlet_O<dim>>(one_over_G);
    beta_function = std::make_unique<const BetaFunction<dim>>(one_over_G);

    switch (mesh_type)
      {
        case MeshType::Mesh9:
          partitioner_type = PartitionerType::rtree;
          agglo_switch     = true;
          mapping          = std::make_unique<MappingQ<dim>>(1);
          break;
        case MeshType::Mesh10:
          partitioner_type = PartitionerType::metis;
          agglo_switch     = true;
          mapping          = std::make_unique<MappingQ<dim>>(1);
          break;
        case MeshType::Mesh11:
          partitioner_type = PartitionerType::rtree;
          agglo_switch     = false;
          mapping          = std::make_unique<MappingQ<dim>>(2);
          break;
        case MeshType::Mesh12:
          partitioner_type = PartitionerType::rtree;
          agglo_switch     = false;
          mapping          = std::make_unique<MappingQ<dim>>(1);
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid mesh type."));
      }

    const std::string type_str = example_name + "_" + Mesh_str(mesh_type);
    const std::string output_path = "output/" + example_name;
    if (!std::filesystem::exists(output_path))
      std::filesystem::create_directories(output_path);
    output_prefix = output_path + "/" + type_str;
  }

  /*============================================================================
     Create and cache the base computational grid for the Oseen–Darcy problem.
  ============================================================================*/
  template <int dim>
  void
  OseenDarcyProblem<dim>::make_base_grid()
  {
    double             radius = 0.4;
    Triangulation<dim> triangulation_outside;
    GridGenerator::hyper_ball_balanced(triangulation, Point<2>(0., 0.), radius);
    GridGenerator::hyper_cube_with_cylindrical_hole(triangulation_outside,
                                                    radius,
                                                    1.);
    GridGenerator::merge_triangulations(
      triangulation, triangulation_outside, triangulation, 1.0e-12, true);

    domain_area = 4.0;
    num_domain  = 2;

    if (agglo_switch)
      {
        std::string cachefilename = output_prefix + "_cached_base";
        if (std::filesystem::exists(cachefilename + "_triangulation.data"))
          {
            std::cout << "     Loading cached base grid from " << cachefilename
                      << " ..." << std::endl;
            triangulation.load(cachefilename);
            triangulation.set_manifold(0, PolarManifold<2>({Point<2>(0., 0.)}));
            triangulation.set_manifold(1, PolarManifold<2>({Point<2>(0., 0.)}));
          }
        else
          {
            std::cout << "     Generating grid..." << std::endl;

            triangulation.set_manifold(0, PolarManifold<2>({Point<2>(0., 0.)}));
            triangulation.set_manifold(1, PolarManifold<2>({Point<2>(0., 0.)}));

            for (const auto &cell : triangulation.active_cell_iterators())
              {
                if (cell->center().distance(Point<2>(0., 0.)) > radius)
                  cell->set_material_id(0); // Oseen domain
                else
                  cell->set_material_id(1); // Darcy domain
              }
            triangulation.refine_global(9);
            triangulation.save(cachefilename);
            std::cout << "     Saved grid to " << cachefilename << std::endl;
          }
      }
    else
      {
        std::cout << "     Generating grid..." << std::endl;

        triangulation.set_manifold(0, PolarManifold<2>({Point<2>(0., 0.)}));
        triangulation.set_manifold(1, PolarManifold<2>({Point<2>(0., 0.)}));

        for (const auto &cell : triangulation.active_cell_iterators())
          {
            if (cell->center().distance(Point<2>(0., 0.)) > radius)
              cell->set_material_id(0); // Oseen domain
            else
              cell->set_material_id(1); // Darcy domain
          }
        triangulation.refine_global(extraction_level - 1);

        std::ofstream      out(output_prefix + "_paramapping_" +
                          std::to_string(extraction_level) + ".dat");
        const unsigned int n_samples_curved = 60;
        for (auto cell : triangulation.active_cell_iterators())
          {
            for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
              {
                auto face = cell->face(f);

                const bool is_flat =
                  (face->manifold_id() == numbers::flat_manifold_id);
                const unsigned int n_samples = is_flat ? 1 : n_samples_curved;

                for (unsigned int i = 0; i <= n_samples; ++i)
                  {
                    const double t = double(i) / n_samples;

                    const unsigned int v0_idx =
                      GeometryInfo<2>::face_to_cell_vertices(f, 0);
                    const unsigned int v1_idx =
                      GeometryInfo<2>::face_to_cell_vertices(f, 1);

                    const Point<2> rv0 =
                      GeometryInfo<2>::unit_cell_vertex(v0_idx);
                    const Point<2> rv1 =
                      GeometryInfo<2>::unit_cell_vertex(v1_idx);

                    const Point<2> p_ref = (1.0 - t) * rv0 + t * rv1;

                    const Point<2> p_real =
                      mapping->transform_unit_to_real_cell(cell, p_ref);

                    out << p_real[0] << " " << p_real[1] << "\n";
                  }
                out << "\n\n";
              }
          }
        out.close();
      }
  }

  /*============================================================================
    Generate an agglomerated mesh with different partitioning strategies.
  ============================================================================*/
  template <int dim>
  void
  OseenDarcyProblem<dim>::make_agglo_grid()
  {
    make_base_grid();

    std::cout << "     Size of base grid: " << triangulation.n_active_cells()
              << std::endl;
    cached_tria =
      std::make_unique<GridTools::Cache<dim>>(triangulation, *mapping);
    agglo_handler = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

    if (partitioner_type == PartitionerType::rtree)
      {
        std::cout << "     Partition with Rtree." << std::endl;
        namespace bgi = boost::geometry::index;
        static constexpr unsigned int max_elem_per_node =
          PolyUtils::constexpr_pow(2, dim);

        std::vector<std::vector<
          std::pair<BoundingBox<dim>,
                    typename Triangulation<dim>::active_cell_iterator>>>
          all_boxes(num_domain);
        // To preserve the curved boundaries of the domains, we use separate
        // R-trees for each domain. A "boxes" is a collection of bounding boxes
        // for a single domain. "all_boxes" is a collection of all such "boxes".
        for (const auto &cell : triangulation.active_cell_iterators())
          all_boxes[cell->material_id()].emplace_back(
            mapping->get_bounding_box(cell), cell);

        for (unsigned int i = 0; i < num_domain; ++i)
          {
            auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(all_boxes[i]);

            std::cout << "     Total number of available levels in domain_" << i
                      << ": " << n_levels(tree) << std::endl;

            const unsigned int extraction_level_sub =
              std::min(extraction_level + 1, n_levels(tree));

            CellsAgglomerator<dim, decltype(tree)> agglomerator{
              tree, extraction_level_sub};
            const auto vec_agglomerates = agglomerator.extract_agglomerates();
            for (const auto &agglo : vec_agglomerates)
              agglo_handler->define_agglomerate(agglo, fe_collection.size());
          }
      }
    else if (partitioner_type == PartitionerType::metis)
      {
        std::cout << "     Partition with Metis." << std::endl;
        n_subdomains = 20 * (int)pow(4, extraction_level - 1); // xxxxxxxx

        // Build cell connectivity graph: each active cell is represented as a
        // node.
        // An undirected edge is added between two neighboring cells only if:
        //   (1) they share a common face, and
        //   (2) they belong to the same material subdomain (same material_id).
        //
        // Thus, cells with different material_id are disconnected in the graph,
        // ensuring agglomerated polytopes do not cross curved material
        // interfaces.
        DynamicSparsityPattern cell_connectivity(
          triangulation.n_active_cells(), triangulation.n_active_cells());
        std::vector<unsigned int> cell_weights(triangulation.n_active_cells());
        for (const auto &cell : triangulation.active_cell_iterators())
          {
            for (const auto &f : cell->face_indices())
              if (!cell->at_boundary(f))
                {
                  const auto neighbor = cell->neighbor(f);
                  if (cell->material_id() == neighbor->material_id())
                    {
                      const unsigned int i = cell->active_cell_index();
                      const unsigned int j = neighbor->active_cell_index();
                      cell_connectivity.add(i, j);
                      cell_connectivity.add(j, i);
                    }
                }
            if (cell->material_id() == 0)
              cell_weights[cell->active_cell_index()] =
                1; // Oseen cells have higher weights (less cells per polytope)
            else
              cell_weights[cell->active_cell_index()] =
                1; // Darcy cells have lower weights (more cells per polytope)
          }


        // Finalize the connectivity graph and call METIS to partition it.
        // Each cell is assigned to one of the n_subdomains.
        SparsityPattern sp_cell_connectivity;
        sp_cell_connectivity.copy_from(cell_connectivity);
        std::vector<unsigned int> partition_indices(
          triangulation.n_active_cells());
        SparsityTools::partition(sp_cell_connectivity,
                                 cell_weights,
                                 n_subdomains,
                                 partition_indices,
                                 SparsityTools::Partitioner::metis);

        // Collect cells belonging to the same subdomain
        std::vector<
          std::vector<typename Triangulation<dim>::active_cell_iterator>>
          cells_per_subdomain(n_subdomains);
        for (const auto &cell : triangulation.active_cell_iterators())
          cells_per_subdomain[partition_indices[cell->active_cell_index()]]
            .push_back(cell);

        /*         // Check that cells in each subdomain have the same
           material_id for (std::size_t i = 0; i < n_subdomains; ++i)
                  {
                    const auto m_id = cells_per_subdomain[i][0]->material_id();
                    for (const auto &cell : cells_per_subdomain[i])
                      Assert(cell->material_id() == m_id,
                             ExcMessage("Cells in a subdomain must have the same
           material_id."));
                  } */

        // For every subdomain, agglomerate elements together
        for (std::size_t i = 0; i < n_subdomains; ++i)
          agglo_handler->define_agglomerate(cells_per_subdomain[i],
                                            fe_collection.size());
      }
    else
      {
        AssertThrow(false,
                    ExcMessage(
                      "Only RTree and METIS are supported at the moment."));
      }

    n_subdomains = agglo_handler->n_agglomerates();
    std::cout << "     N subdomains = " << n_subdomains << std::endl;
  }

  template <int dim>
  bool
  OseenDarcyProblem<dim>::polytope_is_in_Oseen_domain(
    const typename AgglomerationHandler<dim>::agglomeration_iterator &polytope)
  {
    if (polytope.master_cell()->material_id() == 0)
      return true;
    else
      return false;
  }

  template <int dim>
  bool
  OseenDarcyProblem<dim>::polytope_is_in_Darcy_domain(
    const typename AgglomerationHandler<dim>::agglomeration_iterator &polytope)
  {
    if (polytope.master_cell()->material_id() == 1)
      return true;
    else
      return false;
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::set_active_fe_indices()
  {
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        if (polytope_is_in_Oseen_domain(polytope))
          polytope->set_active_fe_index(0); // Oseen
        else if (polytope_is_in_Darcy_domain(polytope))
          polytope->set_active_fe_index(1); // Darcy
        else
          Assert(false,
                 ExcMessage("Polytope with index " +
                            std::to_string(polytope->index()) +
                            " should belong to either Oseen or Darcy domain."));
      }
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::setup_agglomeration()
  {
    set_active_fe_indices();
    agglo_handler->distribute_agglomerated_dofs(fe_collection);
    DynamicSparsityPattern dsp;
    constraints.clear();
    zero_pressure_dof_constraint();
    constraints.close();
    agglo_handler->create_agglomeration_sparsity_pattern(dsp, constraints);
    sparsity.copy_from(dsp);

    // Export polygon boundaries to CSV file based on partitioner type
    std::string partitioner_name;
    switch (partitioner_type)
      {
        case PartitionerType::rtree:
          partitioner_name = "rtree";
          break;
        case PartitionerType::metis:
          partitioner_name = "metis";
          break;
        default:
          partitioner_name = "unknown";
          break;
      }
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::assemble_system()
  {
    system_matrix.reinit(sparsity);
    solution.reinit(agglo_handler->n_dofs());
    system_rhs.reinit(agglo_handler->n_dofs());

    agglo_handler->initialize_fe_values(q_collection,
                                        update_gradients | update_JxW_values |
                                          update_quadrature_points |
                                          update_JxW_values | update_values,
                                        face_q_collection);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure_O(dim);
    const FEValuesExtractors::Scalar pressure_D(dim + 1);

    // Loop over all agglomerated polytopes (cells)
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        const unsigned int current_dofs_per_cell =
          polytope->get_fe().dofs_per_cell;

        FullMatrix<double> cell_matrix(current_dofs_per_cell,
                                       current_dofs_per_cell);
        Vector<double>     cell_rhs(current_dofs_per_cell);
        cell_matrix = 0;
        cell_rhs    = 0;

        std::vector<types::global_dof_index> local_dof_indices(
          current_dofs_per_cell);
        polytope->get_dof_indices(local_dof_indices);

        const auto &agglo_values = agglo_handler->reinit(polytope);
        const auto &q_points     = agglo_values.get_quadrature_points();

        if (polytope_is_in_Oseen_domain(polytope)) // Oseen domain
          {
            std::vector<Tensor<1, dim>> rhs_O(q_points.size());
            rhs_function_O->value_list(q_points, rhs_O);
            std::vector<Tensor<1, dim>> beta(q_points.size());
            beta_function->value_list(q_points, beta);

            std::vector<Tensor<1, dim>> O_phi_u(current_dofs_per_cell);
            std::vector<Tensor<2, dim>> O_grad_phi_u(current_dofs_per_cell);
            std::vector<SymmetricTensor<2, dim>> O_symgrad_phi_u(
              current_dofs_per_cell);
            std::vector<double> O_div_phi_u(current_dofs_per_cell);
            std::vector<double> O_phi_p(current_dofs_per_cell);

            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                  {
                    O_phi_u[k] = agglo_values[velocities].value(k, q_index);
                    O_grad_phi_u[k] =
                      agglo_values[velocities].gradient(k, q_index);
                    O_symgrad_phi_u[k] =
                      agglo_values[velocities].symmetric_gradient(k, q_index);
                    O_div_phi_u[k] =
                      agglo_values[velocities].divergence(k, q_index);
                    O_phi_p[k] = agglo_values[pressure_O].value(k, q_index);
                  }

                for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (2 * viscosity_nu *
                             scalar_product(O_symgrad_phi_u[i],
                                            O_symgrad_phi_u[j]) // + ν ∇v:∇u
                           - O_div_phi_u[i] * O_phi_p[j]        // - ∇·v p_O
                           + O_phi_p[i] * O_div_phi_u[j]        // + ∇·u q_O
                           + O_phi_u[i] *
                               (O_grad_phi_u[j] * beta[q_index]) // + v· (β·∇)u
                           ) *
                          agglo_values.JxW(q_index); // dx
                        // + ν ∫ ∇v : ∇u dx
                        // -   ∫ (∇·v) p_O dx
                        // +   ∫ (∇·u) q_O dx
                      }
                    cell_rhs(i) += O_phi_u[i] * rhs_O[q_index] *
                                   agglo_values.JxW(q_index); // ∫ v·f_O dx
                  }
              }
          }
        else if (polytope_is_in_Darcy_domain(polytope)) // Darcy domain
          {
            std::vector<double> rhs_D(q_points.size());
            rhs_function_D->value_list(q_points, rhs_D);

            std::vector<double>         D_phi_p(current_dofs_per_cell);
            std::vector<Tensor<1, dim>> D_grad_phi_p(current_dofs_per_cell);

            for (unsigned int q_index : agglo_values.quadrature_point_indices())
              {
                for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                  {
                    D_phi_p[k] = agglo_values[pressure_D].value(k, q_index);
                    D_grad_phi_p[k] =
                      agglo_values[pressure_D].gradient(k, q_index);
                  }

                for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (permeability_K * D_grad_phi_p[i]) *
                          D_grad_phi_p[j] *          // + K ∇q_D · ∇p_D
                          agglo_values.JxW(q_index); // dx
                                                     // + ∫ K ∇q_D · ∇p_D dx
                      }
                    cell_rhs(i) += D_phi_p[i] * rhs_D[q_index] *
                                   agglo_values.JxW(q_index); // + q_D f_D dx
                  }
              }
          }
        else
          Assert(false,
                 ExcMessage("Polytope with index " +
                            std::to_string(polytope->index()) +
                            " should belong to either Oseen or Darcy domain."));



        // Loop over faces of the current polytope
        const unsigned int n_faces = polytope->n_faces();
        for (unsigned int f = 0; f < n_faces; ++f)
          {
            if (polytope->at_boundary(f))
              {
                // Handle boundary faces
                const auto &fe_face       = agglo_handler->reinit(polytope, f);
                const auto &face_q_points = fe_face.get_quadrature_points();

                // Get normal vectors seen from each agglomeration.
                const auto &normals = fe_face.get_normal_vectors();

                if (polytope_is_in_Oseen_domain(polytope)) // Oseen domain
                  {
                    std::vector<Tensor<1, dim>> gD_O(face_q_points.size());
                    bcDirichlet_O->value_list(face_q_points, gD_O);
                    std::vector<Tensor<1, dim>> beta(face_q_points.size());
                    beta_function->value_list(face_q_points, beta);

                    // std::vector<Tensor<2, dim>>
                    // O_aver_grad_phi_v(current_dofs_per_cell);
                    std::vector<SymmetricTensor<2, dim>> O_aver_symgrad_phi_v(
                      current_dofs_per_cell);
                    std::vector<Tensor<1, dim>> O_jump_phi_v(
                      current_dofs_per_cell);
                    std::vector<double> O_aver_phi_p(current_dofs_per_cell);
                    std::vector<double> O_jump_phi_p(current_dofs_per_cell);

                    unsigned int deg_v_current =
                      polytope->get_fe().get_sub_fe(0, 1).degree;
                    // get_sub_fe(first_component, n_selected_components)
                    double tau_cell = (viscosity_nu) * (deg_v_current + 1) *
                                      (deg_v_current + dim) /
                                      std::fabs(polytope->diameter());
                    double sigma_v = penalty_constant_v * tau_cell;

                    for (unsigned int q_index :
                         fe_face.quadrature_point_indices())
                      {
                        double is_face_inflow_of_cell =
                          (beta[q_index] * normals[q_index]) < 0 ? 1.0 : 0.0;

                        for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                          {
                            O_aver_symgrad_phi_v[k] =
                              fe_face[velocities].symmetric_gradient(k,
                                                                     q_index);
                            O_jump_phi_v[k] =
                              fe_face[velocities].value(k, q_index);
                            O_aver_phi_p[k] =
                              fe_face[pressure_O].value(k, q_index);
                          }

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < current_dofs_per_cell;
                                 ++j)
                              {
                                if (true) // Dirichlet
                                  cell_matrix(i, j) +=
                                    (-2 * viscosity_nu * O_jump_phi_v[i] *
                                       (O_aver_symgrad_phi_v[j] *
                                        normals[q_index]) // - ν [v] · ({∇u} ·
                                                          // n)
                                     - 2 * viscosity_nu * O_jump_phi_v[j] *
                                         (O_aver_symgrad_phi_v[i] *
                                          normals[q_index]) // - ν [u] · ({∇v} ·
                                                            // n)
                                     + sigma_v * O_jump_phi_v[i] *
                                         O_jump_phi_v[j] // + σ_v [v] · [u]
                                     + O_aver_phi_p[j] * O_jump_phi_v[i] *
                                         normals[q_index] // + [v] · n · {p_O}
                                     - O_aver_phi_p[i] * O_jump_phi_v[j] *
                                         normals[q_index] // - [u] · n · {q_O}
                                     - is_face_inflow_of_cell * // inflow faces
                                                                // only
                                         (beta[q_index] * normals[q_index]) *
                                         O_jump_phi_v[j] *
                                         O_jump_phi_v[i] // - (β·n) v_down·[u]
                                                         // v_down = [v] at
                                                         // inflow boundary
                                     ) *
                                    fe_face.JxW(q_index); // ds

                                // - ν ∫    [v] · ({∇u} · n) ds
                                // - ν ∫    [u] · ({∇v} · n) ds
                                // + ∫     σ_v [v] · [u] ds
                                // + ∫     [v] · n · {p_O} ds
                                // - ∫     [u] · n · {q_O} ds
                              }
                            cell_rhs(i) +=
                              (-2 * viscosity_nu * gD_O[q_index] *
                                 (O_aver_symgrad_phi_v[i] *
                                  normals[q_index]) // - ν gD_O · ({∇v} · n)
                               + sigma_v * gD_O[q_index] *
                                   O_jump_phi_v[i] // + σ_v gD_O · [v]
                               - O_aver_phi_p[i] * gD_O[q_index] *
                                   normals[q_index] // - gD_O · n · {q_O}
                               - is_face_inflow_of_cell *
                                   (beta[q_index] * normals[q_index]) *
                                   gD_O[q_index] *
                                   O_jump_phi_v[i] // - (β·n) g · [v]
                               ) *
                              fe_face.JxW(q_index); // ds
                            // - ν ∫    gD_O · ({∇v} · n) ds
                            // + ∫     σ_v gD_O · [v] ds
                            // - ∫     gD_O · n · {q_O} ds
                          }

                        // where:
                        //   [·] = jump across the face; equals
                        //   value on the current cell at the boundary
                        //   {·} = average across the face; equals value on the
                        //   current cell at the boundary
                        //   gD_O = Dirichlet data on the interface (used as
                        //   boundary value of u_O)
                        //   σ_v = penalty parameter for velocity
                      }
                  }
                else if (polytope_is_in_Darcy_domain(polytope)) // Darcy domain
                  {
                    Assert(false, ExcMessage("No boundary Darcy xxx."));
                  }
                else
                  Assert(false,
                         ExcMessage(
                           "Polytope with index " +
                           std::to_string(polytope->index()) +
                           " should belong to either Oseen or Darcy domain."));
              }
            else
              {
                // Handle internal faces/interfaces
                const auto &neigh_polytope = polytope->neighbor(f);

                // This is necessary to loop over internal faces only once.
                if (polytope->index() < neigh_polytope->index())
                  {
                    const unsigned int neigh_dofs_per_cell =
                      neigh_polytope->get_fe().dofs_per_cell;

                    FullMatrix<double> M11(current_dofs_per_cell,
                                           current_dofs_per_cell);
                    FullMatrix<double> M12(current_dofs_per_cell,
                                           neigh_dofs_per_cell);
                    FullMatrix<double> M21(neigh_dofs_per_cell,
                                           current_dofs_per_cell);
                    FullMatrix<double> M22(neigh_dofs_per_cell,
                                           neigh_dofs_per_cell);
                    M11 = 0.;
                    M12 = 0.;
                    M21 = 0.;
                    M22 = 0.;
                    // During interface integrals, dofs from both
                    // adjacent cells are involved.
                    //
                    // M11 corresponds to test and trial functions both on the
                    // current cell. M12 corresponds to test functions on the
                    // current cell and trial functions on the neighbor cell.
                    // M21 and M22 correspond similarly, with test functions on
                    // the neighbor cell.
                    //
                    // When using hp::FECollection, the number of dofs may
                    // differ between cells, so M12 and M21 are generally not
                    // square.
                    //
                    //                 dof_current   dof_neighbor
                    //   dof_current       M11           M12
                    //   dof_neighbor      M21           M22

                    std::vector<types::global_dof_index>
                      local_dof_indices_neighbor(neigh_dofs_per_cell);
                    neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

                    unsigned int nofn =
                      polytope->neighbor_of_agglomerated_neighbor(f);
                    const auto &fe_faces = agglo_handler->reinit_interface(
                      polytope, neigh_polytope, f, nofn);
                    const auto &fe_faces0 = fe_faces.first;
                    const auto &fe_faces1 = fe_faces.second;

                    const auto &normals = fe_faces0.get_normal_vectors();


                    std::vector<Tensor<1, dim>> D_aver_grad_phi_p(
                      current_dofs_per_cell);
                    std::vector<double> D_jump_phi_p(current_dofs_per_cell);


                    // This is an interior face within the Oseen domain
                    if (polytope_is_in_Oseen_domain(polytope) &&
                        polytope_is_in_Oseen_domain(neigh_polytope))
                      {
                        // std::vector<Tensor<2, dim>>
                        // O_aver_grad_phi_v0(current_dofs_per_cell);
                        std::vector<Tensor<2, dim>> O_aver_symgrad_phi_v0(
                          current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> O_jump_phi_v0(
                          current_dofs_per_cell);
                        std::vector<double> O_aver_phi_p0(
                          current_dofs_per_cell);
                        std::vector<double> O_jump_phi_p0(
                          current_dofs_per_cell);
                        // std::vector<Tensor<2, dim>>
                        // O_aver_grad_phi_v1(neigh_dofs_per_cell);
                        std::vector<Tensor<2, dim>> O_aver_symgrad_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<Tensor<1, dim>> O_jump_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<double> O_aver_phi_p1(neigh_dofs_per_cell);
                        std::vector<double> O_jump_phi_p1(neigh_dofs_per_cell);

                        std::vector<Tensor<1, dim>> downwind_phi_v0(
                          current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> downwind_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<Tensor<1, dim>> beta(
                          fe_faces0.n_quadrature_points);
                        beta_function->value_list(
                          fe_faces0.get_quadrature_points(), beta);

                        unsigned int deg_v_current =
                          polytope->get_fe().get_sub_fe(0, 1).degree;
                        unsigned int deg_v_neigh =
                          neigh_polytope->get_fe().get_sub_fe(0, 1).degree;
                        double tau_current = (viscosity_nu) *
                                             (deg_v_current + 1) *
                                             (deg_v_current + dim) /
                                             std::fabs(polytope->diameter());
                        double tau_neigh =
                          (viscosity_nu) * (deg_v_neigh + 1) *
                          (deg_v_neigh + dim) /
                          std::fabs(neigh_polytope->diameter());
                        double sigma_v =
                          penalty_constant_v * std::max(tau_current, tau_neigh);
                        double zeta_current =
                          1. / (viscosity_nu / polytope->diameter());
                        double zeta_neigh =
                          1. / (viscosity_nu / neigh_polytope->diameter());
                        double sigma_p_O = penalty_constant_p_O *
                                           std::max(zeta_current, zeta_neigh);

                        for (unsigned int q_index = 0;
                             q_index < fe_faces0.n_quadrature_points;
                             ++q_index)
                          {
                            bool is_face_inflow_of_cell =
                              ((beta[q_index] * normals[q_index]) < 0);

                            for (unsigned int k = 0; k < current_dofs_per_cell;
                                 ++k)
                              {
                                O_aver_symgrad_phi_v0[k] =
                                  0.5 * fe_faces0[velocities]
                                          .symmetric_gradient(k, q_index);
                                O_jump_phi_v0[k] =
                                  fe_faces0[velocities].value(k, q_index);
                                O_aver_phi_p0[k] =
                                  0.5 * fe_faces0[pressure_O].value(k, q_index);
                                O_jump_phi_p0[k] =
                                  fe_faces0[pressure_O].value(k, q_index);
                                if (is_face_inflow_of_cell)
                                  downwind_phi_v0[k] =
                                    fe_faces0[velocities].value(k, q_index);
                                else
                                  downwind_phi_v0[k] =
                                    -fe_faces0[velocities].value(k, q_index);
                              }

                            for (unsigned int k = 0; k < neigh_dofs_per_cell;
                                 ++k)
                              {
                                O_aver_symgrad_phi_v1[k] =
                                  0.5 * fe_faces1[velocities]
                                          .symmetric_gradient(k, q_index);
                                O_jump_phi_v1[k] =
                                  -fe_faces1[velocities].value(k, q_index);
                                O_aver_phi_p1[k] =
                                  0.5 * fe_faces1[pressure_O].value(k, q_index);
                                O_jump_phi_p1[k] =
                                  -fe_faces1[pressure_O].value(k, q_index);
                                if (is_face_inflow_of_cell)
                                  downwind_phi_v1[k] =
                                    -fe_faces1[velocities].value(k, q_index);
                                else
                                  downwind_phi_v1[k] =
                                    fe_faces1[velocities].value(k, q_index);
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M11(i, j) +=
                                      (-2 * viscosity_nu * O_jump_phi_v0[i] *
                                         (O_aver_symgrad_phi_v0[j] *
                                          normals[q_index]) // - ν [v] · ({∇u} ·
                                                            // n)
                                       - 2 * viscosity_nu * O_jump_phi_v0[j] *
                                           (O_aver_symgrad_phi_v0[i] *
                                            normals[q_index]) // - ν [u] · ({∇v}
                                                              // · n)
                                       + sigma_v * O_jump_phi_v0[i] *
                                           O_jump_phi_v0[j] // + σ_v [v] · [u]
                                       + O_aver_phi_p0[j] * O_jump_phi_v0[i] *
                                           normals[q_index] // + [v] · n · {p_O}
                                       - O_aver_phi_p0[i] * O_jump_phi_v0[j] *
                                           normals[q_index] // - [u] · n · {q_O}
                                       + sigma_p_O * O_jump_phi_p0[i] *
                                           O_jump_phi_p0[j] // + σ_p_O [p_O] ·
                                                            // [q_O]
                                       - (beta[q_index] * normals[q_index]) *
                                           O_jump_phi_v0[j] *
                                           downwind_phi_v0[i] // - (β·n)
                                                              // v_down·[u]
                                       ) *
                                      fe_faces0.JxW(q_index); // ds

                                    // - ν ∫    [v] · ({∇u} · n) ds
                                    // - ν ∫    [u] · ({∇v} · n) ds
                                    // + ∫     σ_v [v] · [u] ds
                                    // + ∫     [v] · n · {p_O} ds
                                    // - ∫     [u] · n · {q_O} ds
                                    // + ∫     σ_p_O [p_O] · [q_O] ds
                                    //
                                    // where:
                                    //   [·]       = jump across face
                                    //   {·}       = average across face
                                    //   ∫         = integral over interior face
                                    //   σ_v       = velocity penalty parameter
                                    //   σ_p_O     = pressure penalty parameter
                                    //   for Oseen pressure
                                    //   σ_p_D     = pressure penalty parameter
                                    //   for Darcy pressure (used elsewhere)
                                    //
                                    // Note:
                                    //   O_ prefix denotes Oseen region
                                    //   Suffix '0' indicates basis functions of
                                    //   the current cell only (no neighbor)
                                  }
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M12(i, j) +=
                                      (-2 * viscosity_nu * O_jump_phi_v0[i] *
                                         (O_aver_symgrad_phi_v1[j] *
                                          normals[q_index]) -
                                       2 * viscosity_nu * O_jump_phi_v1[j] *
                                         (O_aver_symgrad_phi_v0[i] *
                                          normals[q_index]) +
                                       sigma_v * O_jump_phi_v0[i] *
                                         O_jump_phi_v1[j] +
                                       O_aver_phi_p1[j] * O_jump_phi_v0[i] *
                                         normals[q_index] -
                                       O_aver_phi_p0[i] * O_jump_phi_v1[j] *
                                         normals[q_index] +
                                       sigma_p_O * O_jump_phi_p0[i] *
                                         O_jump_phi_p1[j] -
                                       (beta[q_index] * normals[q_index]) *
                                         O_jump_phi_v1[j] *
                                         downwind_phi_v0[i]) *
                                      fe_faces0.JxW(q_index);
                                    // Same structure as M11; only the basis
                                    // functions differ.
                                    //
                                    // Suffix '1' refers to neighbor cell basis
                                    // functions, while suffix '0' refers to
                                    // current cell basis functions.
                                    //
                                    // Index [j] corresponds to trial functions,
                                    // and [i] to test functions.
                                    //
                                    // In M21, all [i] indices are associated
                                    // with suffix '1', indicating test
                                    // functions from the neighbor cell and
                                    // trial functions from the current cell.
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M21(i, j) +=
                                      (-2 * viscosity_nu * O_jump_phi_v1[i] *
                                         (O_aver_symgrad_phi_v0[j] *
                                          normals[q_index]) -
                                       2 * viscosity_nu * O_jump_phi_v0[j] *
                                         (O_aver_symgrad_phi_v1[i] *
                                          normals[q_index]) +
                                       sigma_v * O_jump_phi_v1[i] *
                                         O_jump_phi_v0[j] +
                                       O_aver_phi_p0[j] * O_jump_phi_v1[i] *
                                         normals[q_index] -
                                       O_aver_phi_p1[i] * O_jump_phi_v0[j] *
                                         normals[q_index] +
                                       sigma_p_O * O_jump_phi_p1[i] *
                                         O_jump_phi_p0[j] -
                                       (beta[q_index] * normals[q_index]) *
                                         O_jump_phi_v0[j] *
                                         downwind_phi_v1[i]) *
                                      fe_faces0.JxW(q_index);
                                    // In M21, [j] indices use suffix '0',
                                    // indicating trial functions from the
                                    // current cell, while [i] indices use
                                    // suffix '1', indicating test functions
                                    // from the neighbor cell.
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M22(i, j) +=
                                      (-2 * viscosity_nu * O_jump_phi_v1[i] *
                                         (O_aver_symgrad_phi_v1[j] *
                                          normals[q_index]) -
                                       2 * viscosity_nu * O_jump_phi_v1[j] *
                                         (O_aver_symgrad_phi_v1[i] *
                                          normals[q_index]) +
                                       sigma_v * O_jump_phi_v1[i] *
                                         O_jump_phi_v1[j] +
                                       O_aver_phi_p1[j] * O_jump_phi_v1[i] *
                                         normals[q_index] -
                                       O_aver_phi_p1[i] * O_jump_phi_v1[j] *
                                         normals[q_index] +
                                       sigma_p_O * O_jump_phi_p1[i] *
                                         O_jump_phi_p1[j] -
                                       (beta[q_index] * normals[q_index]) *
                                         O_jump_phi_v1[j] *
                                         downwind_phi_v1[i]) *
                                      fe_faces0.JxW(q_index);
                                    // In M22, both test and trial functions use
                                    // suffix '1', meaning they are associated
                                    // with the neighbor cell only.
                                  }
                              }
                          }
                      }

                    // This is an interior face within the Darcy domain
                    if (polytope_is_in_Darcy_domain(polytope) &&
                        polytope_is_in_Darcy_domain(neigh_polytope))
                      {
                        std::vector<Tensor<1, dim>> D_aver_grad_phi_p0(
                          current_dofs_per_cell);
                        std::vector<double> D_jump_phi_p0(
                          current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> D_aver_grad_phi_p1(
                          neigh_dofs_per_cell);
                        std::vector<double> D_jump_phi_p1(neigh_dofs_per_cell);

                        unsigned int deg_p_current =
                          polytope->get_fe().get_sub_fe(3, 1).degree;
                        unsigned int deg_p_neigh =
                          neigh_polytope->get_fe().get_sub_fe(3, 1).degree;
                        double tau_current = (permeability_scalar) *
                                             (deg_p_current + 1) *
                                             (deg_p_current + dim) /
                                             std::fabs(polytope->diameter());
                        double tau_neigh =
                          (permeability_scalar) * (deg_p_neigh + 1) *
                          (deg_p_neigh + dim) /
                          std::fabs(neigh_polytope->diameter());
                        double sigma_p_D = penalty_constant_p_D *
                                           std::max(tau_current, tau_neigh);

                        for (unsigned int q_index = 0;
                             q_index < fe_faces0.n_quadrature_points;
                             ++q_index)
                          {
                            for (unsigned int k = 0; k < current_dofs_per_cell;
                                 ++k)
                              {
                                D_aver_grad_phi_p0[k] =
                                  0.5 *
                                  fe_faces0[pressure_D].gradient(k, q_index);
                                D_jump_phi_p0[k] =
                                  fe_faces0[pressure_D].value(k, q_index);
                              }

                            for (unsigned int k = 0; k < neigh_dofs_per_cell;
                                 ++k)
                              {
                                D_aver_grad_phi_p1[k] =
                                  0.5 *
                                  fe_faces1[pressure_D].gradient(k, q_index);
                                D_jump_phi_p1[k] =
                                  -fe_faces1[pressure_D].value(k, q_index);
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M11(i, j) +=
                                      (-D_jump_phi_p0[i] *
                                         D_aver_grad_phi_p0[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p0[j] *
                                         D_aver_grad_phi_p0[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p0[i] *
                                         D_jump_phi_p0[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M12(i, j) +=
                                      (-D_jump_phi_p0[i] *
                                         D_aver_grad_phi_p1[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p1[j] *
                                         D_aver_grad_phi_p0[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p0[i] *
                                         D_jump_phi_p1[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M21(i, j) +=
                                      (-D_jump_phi_p1[i] *
                                         D_aver_grad_phi_p0[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p0[j] *
                                         D_aver_grad_phi_p1[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p1[i] *
                                         D_jump_phi_p0[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M22(i, j) +=
                                      (-D_jump_phi_p1[i] *
                                         D_aver_grad_phi_p1[j] *
                                         permeability_K * normals[q_index] -
                                       D_jump_phi_p1[j] *
                                         D_aver_grad_phi_p1[i] *
                                         permeability_K * normals[q_index] +
                                       sigma_p_D * D_jump_phi_p1[i] *
                                         D_jump_phi_p1[j]) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }
                          }
                      }

                    // This is an interface between Oseen and Darcy domains
                    if ((polytope_is_in_Oseen_domain(polytope) &&
                         polytope_is_in_Darcy_domain(neigh_polytope)) ||
                        (polytope_is_in_Darcy_domain(polytope) &&
                         polytope_is_in_Oseen_domain(neigh_polytope)))
                      {
                        bool is_stokes_side =
                          polytope_is_in_Oseen_domain(polytope);
                        Tensor<1, dim> normal;
                        Tensor<1, dim> tangential;

                        std::vector<Tensor<1, dim>> O_phi_v0(
                          current_dofs_per_cell);
                        std::vector<double> D_phi_p0(current_dofs_per_cell);
                        std::vector<Tensor<1, dim>> O_phi_v1(
                          neigh_dofs_per_cell);
                        std::vector<double> D_phi_p1(neigh_dofs_per_cell);

                        for (unsigned int q_index = 0;
                             q_index < fe_faces0.n_quadrature_points;
                             ++q_index)
                          {
                            if (is_stokes_side)
                              normal = normals[q_index];
                            else
                              normal = -normals[q_index];
                            tangential[0] = -normal[1];
                            tangential[1] = normal[0];

                            for (unsigned int k = 0; k < current_dofs_per_cell;
                                 ++k)
                              {
                                O_phi_v0[k] =
                                  fe_faces0[velocities].value(k, q_index);
                                D_phi_p0[k] =
                                  fe_faces0[pressure_D].value(k, q_index);
                              }

                            for (unsigned int k = 0; k < neigh_dofs_per_cell;
                                 ++k)
                              {
                                O_phi_v1[k] =
                                  fe_faces1[velocities].value(k, q_index);
                                D_phi_p1[k] =
                                  fe_faces1[pressure_D].value(k, q_index);
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M11(i, j) +=
                                      (D_phi_p0[j] * O_phi_v0[i] * normal - // 0
                                       D_phi_p0[i] * O_phi_v0[j] * normal + // 0
                                       nu_over_G * (O_phi_v0[i] * tangential) *
                                         (O_phi_v0[j] *
                                          tangential) // + ν/G(u·t)(v·t)
                                       ) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < current_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M12(i, j) +=
                                      (D_phi_p1[j] * O_phi_v0[i] * normal -
                                       D_phi_p0[i] * O_phi_v1[j] * normal +
                                       nu_over_G * (O_phi_v0[i] * tangential) *
                                         (O_phi_v1[j] * tangential) // 0
                                       ) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < current_dofs_per_cell;
                                     ++j)
                                  {
                                    M21(i, j) +=
                                      (D_phi_p0[j] * O_phi_v1[i] * normal -
                                       D_phi_p1[i] * O_phi_v0[j] * normal +
                                       nu_over_G * (O_phi_v1[i] * tangential) *
                                         (O_phi_v0[j] * tangential)) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }

                            for (unsigned int i = 0; i < neigh_dofs_per_cell;
                                 ++i)
                              {
                                for (unsigned int j = 0;
                                     j < neigh_dofs_per_cell;
                                     ++j)
                                  {
                                    M22(i, j) +=
                                      (D_phi_p1[j] * O_phi_v1[i] * normal -
                                       D_phi_p1[i] * O_phi_v1[j] * normal +
                                       nu_over_G * (O_phi_v1[i] * tangential) *
                                         (O_phi_v1[j] * tangential)) *
                                      fe_faces0.JxW(q_index);
                                  }
                              }
                          }
                      }


                    constraints.distribute_local_to_global(M11,
                                                           local_dof_indices,
                                                           system_matrix);
                    constraints.distribute_local_to_global(
                      M12,
                      local_dof_indices,
                      local_dof_indices_neighbor,
                      system_matrix);
                    constraints.distribute_local_to_global(
                      M21,
                      local_dof_indices_neighbor,
                      local_dof_indices,
                      system_matrix);
                    constraints.distribute_local_to_global(
                      M22, local_dof_indices_neighbor, system_matrix);
                  } // Loop only once trough internal faces
              }
          } // Loop over faces of current cell
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::solve()
  {
    constraints.condense(system_matrix);
    constraints.condense(system_rhs);

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);

    constraints.distribute(solution);

    std::cout << "   Interpolating..." << std::endl;
    PolyUtils::interpolate_to_fine_grid(*agglo_handler,
                                        interpolated_solution,
                                        solution,
                                        true /*on_the_fly*/);
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::zero_pressure_dof_constraint()
  {
    const FEValuesExtractors::Scalar pressure_O(dim);
    ComponentMask  pressure_O_mask = fe_collection.component_mask(pressure_O);
    const IndexSet pressure_dofs =
      DoFTools::extract_dofs(agglo_handler->agglo_dh, pressure_O_mask);
    const types::global_dof_index first_pressure_dof =
      pressure_dofs.nth_index_in_set(0);
    constraints.constrain_dof_to_zero(first_pressure_dof);
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::mean_pressure_to_zero()
  {
    const ComponentSelectFunction<dim> pressure_O_mask(dim, dim + 2);
    const ComponentSelectFunction<dim> pressure_D_mask(dim + 1, dim + 2);

    Vector<double> integral_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      Functions::ZeroFunction<dim>(dim + 2),
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_O_mask);
    const double global_pressure_O_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);
    VectorTools::integrate_difference(agglo_handler->output_dh,
                                      interpolated_solution,
                                      Functions::ZeroFunction<dim>(dim + 2),
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_D_mask);
    const double global_pressure_D_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);

    const double mean_pressure =
      (global_pressure_O_integral + global_pressure_D_integral) / domain_area;

    for (const auto &cell : agglo_handler->output_dh.active_cell_iterators())
      {
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int dof_component =
              cell->get_fe().system_to_component_index(i).first;
            if (dof_component == dim)
              interpolated_solution[local_dof_indices[i]] -= mean_pressure;
            else if (dof_component == dim + 1)
              interpolated_solution[local_dof_indices[i]] -= mean_pressure;
          }
      }
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::compute_errors()
  {
    hp::MappingCollection<dim> mapping_collection(*mapping);
    hp::FEValues<dim>          hp_fe_values(mapping_collection,
                                   agglo_handler->output_dh.get_fe_collection(),
                                   q_collection,
                                   update_JxW_values |
                                     update_quadrature_points |
                                     update_gradients | update_values);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure_O(dim);
    const FEValuesExtractors::Scalar pressure_D(dim + 1);

    std::vector<Tensor<1, dim>>   velocities_values;
    std::vector<Tensor<dim, dim>> velocities_gradients;
    std::vector<double>           pressure_O_values;
    std::vector<double>           pressure_D_values;
    std::vector<Tensor<1, dim>>   pressure_D_gradients;

    std::vector<Vector<double>>              exact_values;
    std::vector<std::vector<Tensor<1, dim>>> exact_gradients;

    for (auto cell : agglo_handler->output_dh.active_cell_iterators())
      {
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values  = hp_fe_values.get_present_fe_values();
        const unsigned int   n_q_points = fe_values.n_quadrature_points;

        exact_values.resize(n_q_points);
        exact_solution->vector_value_list(fe_values.get_quadrature_points(),
                                          exact_values);
        exact_gradients.resize(n_q_points);
        exact_solution->vector_gradient_list(fe_values.get_quadrature_points(),
                                             exact_gradients);

        if (cell->material_id() == 0) // Oseen
          {
            velocities_values.resize(n_q_points);
            fe_values[velocities].get_function_values(interpolated_solution,
                                                      velocities_values);
            velocities_gradients.resize(n_q_points);
            fe_values[velocities].get_function_gradients(interpolated_solution,
                                                         velocities_gradients);
            pressure_O_values.resize(n_q_points);
            fe_values[pressure_O].get_function_values(interpolated_solution,
                                                      pressure_O_values);

            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                error_velocity_L2 +=
                  ((velocities_values[q][0] - exact_values[q][0]) *
                     (velocities_values[q][0] - exact_values[q][0]) +
                   (velocities_values[q][1] - exact_values[q][1]) *
                     (velocities_values[q][1] - exact_values[q][1])) *
                  fe_values.JxW(q);
                error_velocity_H1 +=
                  ((velocities_gradients[q][0] - exact_gradients[q][0]) *
                     (velocities_gradients[q][0] - exact_gradients[q][0]) +
                   (velocities_gradients[q][1] - exact_gradients[q][1]) *
                     (velocities_gradients[q][1] - exact_gradients[q][1])) *
                  fe_values.JxW(q);
                error_pressure_O +=
                  (pressure_O_values[q] - exact_values[q][2]) *
                  (pressure_O_values[q] - exact_values[q][2]) *
                  fe_values.JxW(q);
              }
          }
        else if (cell->material_id() == 1) // Darcy
          {
            pressure_D_values.resize(n_q_points);
            fe_values[pressure_D].get_function_values(interpolated_solution,
                                                      pressure_D_values);
            pressure_D_gradients.resize(n_q_points);
            fe_values[pressure_D].get_function_gradients(interpolated_solution,
                                                         pressure_D_gradients);

            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                error_pressure_D_semiH1 +=
                  (pressure_D_gradients[q] - exact_gradients[q][3]) *
                  (pressure_D_gradients[q] - exact_gradients[q][3]) *
                  fe_values.JxW(q);
                error_pressure_D_L2 +=
                  (pressure_D_values[q] - exact_values[q][3]) *
                  (pressure_D_values[q] - exact_values[q][3]) *
                  fe_values.JxW(q);
              }
          }
      }

    error_velocity_global   = error_velocity_L2 + error_pressure_D_semiH1;
    error_pressure_global   = error_pressure_O + error_pressure_D_L2;
    error_velocity_L2       = std::sqrt(error_velocity_L2);
    error_velocity_H1       = std::sqrt(error_velocity_H1);
    error_pressure_O        = std::sqrt(error_pressure_O);
    error_pressure_D_L2     = std::sqrt(error_pressure_D_L2);
    error_pressure_D_semiH1 = std::sqrt(error_pressure_D_semiH1);
    error_velocity_global   = std::sqrt(error_velocity_global);
    error_pressure_global   = std::sqrt(error_pressure_global);

    std::cout << "     velocity L2 Error: " << error_velocity_L2 << std::endl
              << "     velocity H1 Error: " << error_velocity_H1 << std::endl
              << "     pressure_O L2 Error: " << error_pressure_O << std::endl
              << "     pressure_D L2 Error: " << error_pressure_D_L2
              << std::endl
              << "     pressure_D semiH1 Error: " << error_pressure_D_semiH1
              << std::endl;
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::output_results(unsigned int n_subdomains) const
  {
    std::string n_polytopes_str = Utilities::int_to_string(n_subdomains, /* digits = */ 5);

    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure_O");
    solution_names.emplace_back("pressure_D");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(agglo_handler->output_dh);

    data_out.add_data_vector(interpolated_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> agglo_idx(triangulation.n_active_cells());
    for (const auto &polytope : agglo_handler->polytope_iterators())
      {
        const types::global_cell_index polytope_index = polytope->index();
        const auto &patch_of_cells = polytope->get_agglomerate(); // fine cells
        // Flag them
        for (const auto &cell : patch_of_cells)
          agglo_idx[cell->active_cell_index()] = polytope_index;
      }
    data_out.add_data_vector(agglo_idx,
                             "agglo_idx",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches();

    std::string output_filename =
      output_prefix + "_solution_" + n_polytopes_str + ".vtk";
    std::ofstream output(output_filename);

    data_out.write_vtk(output);

    std::cout << "     Solution written to file: "
              << output_filename
              << std::endl;

    if (agglo_switch)
      {
        std::string polygon_boundaries =
          output_prefix + "_polygon_boundaries_" + n_polytopes_str;
        PolyUtils::export_polygon_to_csv_file(*agglo_handler,
                                              polygon_boundaries);
      }
  }

  template <int dim>
  void
  OseenDarcyProblem<dim>::run()
  {
    std::cout << "   Making grid..." << std::endl;
    if (agglo_switch)
      make_agglo_grid();
    else
      {
        make_base_grid();
        std::cout << "     Size of base grid: "
                  << triangulation.n_active_cells() << std::endl;
        cached_tria =
          std::make_unique<GridTools::Cache<dim>>(triangulation, *mapping);
        agglo_handler =
          std::make_unique<AgglomerationHandler<dim>>(*cached_tria);
        for (const auto &cell : triangulation.active_cell_iterators())
          agglo_handler->define_agglomerate({cell}, fe_collection.size());
        n_subdomains = agglo_handler->n_agglomerates();
        std::cout << "     N subdomains = " << n_subdomains << std::endl;
      }

    std::cout << "   Setting up agglomeration..." << std::endl;
    setup_agglomeration();

    std::cout << "   Assembling..." << std::endl;
    assemble_system();

    std::cout << "   Solving..." << std::endl;
    solve();

    std::cout << "   Modifying pressure..." << std::endl;
    mean_pressure_to_zero();

    std::cout << "   Error:" << std::endl;
    compute_errors();

    std::cout << "   Writing output..." << std::endl;
    output_results(n_subdomains);

    std::cout << std::endl;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_velocity_L2() const
  {
    return error_velocity_L2;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_velocity_H1() const
  {
    return error_velocity_H1;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_pressure_O() const
  {
    return error_pressure_O;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_pressure_D_L2() const
  {
    return error_pressure_D_L2;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_pressure_D_semiH1() const
  {
    return error_pressure_D_semiH1;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_velocity_global() const
  {
    return error_velocity_global;
  }

  template <int dim>
  double
  OseenDarcyProblem<dim>::get_error_pressure_global() const
  {
    return error_pressure_global;
  }

  template <int dim>
  unsigned int
  OseenDarcyProblem<dim>::get_n_dofs() const
  {
    return agglo_handler->n_dofs();
  }

  template <int dim>
  unsigned int
  OseenDarcyProblem<dim>::get_n_polytopes() const
  {
    return agglo_handler->n_agglomerates();
  }
} // namespace OseenDarcyNamespace

int
main()
{
  try
    {
      using namespace OseenDarcyNamespace;

      for (const MeshType mesh_type : {MeshType::Mesh9,
                                       MeshType::Mesh10,
                                       MeshType::Mesh11,
                                       MeshType::Mesh12})
        {
          const std::string type_str = "example4_" + Mesh_str(mesh_type);
          std::cout << "\n" << std::string(type_str.length(), '-') << std::endl;
          std::cout << type_str << std::endl;
          std::cout << std::string(type_str.length(), '-') << std::endl;

          ConvergenceTable   convergence_table;
          const unsigned int deg_v   = 3;
          const unsigned int deg_p_O = 2;
          const unsigned int deg_p_D = 3;

          for (unsigned int mesh_level = 3; mesh_level < 6; ++mesh_level)
            {
              std::cout << "level " << mesh_level << std::endl;
              convergence_table.add_value("level", mesh_level);

              if (mesh_level < 8)
                {
                  OseenDarcyProblem<2> SD_problem(
                    deg_v, deg_p_O, deg_p_D, mesh_level, mesh_type);

                  SD_problem.run();
                  convergence_table.add_value("polytopes",
                                              SD_problem.get_n_polytopes());
                  convergence_table.add_value("dofs", SD_problem.get_n_dofs());
                  convergence_table.add_value(
                    "velocity_H1", SD_problem.get_error_velocity_H1());
                  convergence_table.add_value(
                    "Ev", SD_problem.get_error_velocity_global());
                  convergence_table.add_value(
                    "Ep", SD_problem.get_error_pressure_global());
                  convergence_table.add_value(
                    "velocity_L2", SD_problem.get_error_velocity_L2());
                  convergence_table.add_value(
                    "pressure_O", SD_problem.get_error_pressure_O());
                  convergence_table.add_value(
                    "pressure_D_L2", SD_problem.get_error_pressure_D_L2());
                  convergence_table.add_value(
                    "pressure_D_semiH1",
                    SD_problem.get_error_pressure_D_semiH1());
                }
              else
                AssertThrow(
                  false,
                  ExcMessage(
                    "You need to refine the base grid to use higher mesh levels."
                    "Please modify make_base_grid()."));
            }

          convergence_table.set_precision("velocity_H1", 2);
          convergence_table.set_precision("Ev", 2);
          convergence_table.set_precision("Ep", 2);
          convergence_table.set_precision("velocity_L2", 2);
          convergence_table.set_precision("pressure_O", 2);
          convergence_table.set_precision("pressure_D_L2", 2);
          convergence_table.set_precision("pressure_D_semiH1", 2);
          convergence_table.set_scientific("velocity_H1", true);
          convergence_table.set_scientific("Ev", true);
          convergence_table.set_scientific("Ep", true);
          convergence_table.set_scientific("velocity_L2", true);
          convergence_table.set_scientific("pressure_O", true);
          convergence_table.set_scientific("pressure_D_L2", true);
          convergence_table.set_scientific("pressure_D_semiH1", true);

          convergence_table.evaluate_convergence_rates(
            "velocity_H1",
            "polytopes",
            ConvergenceTable::reduction_rate_log2,
            2);
          convergence_table.evaluate_convergence_rates(
            "Ev", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
          convergence_table.evaluate_convergence_rates(
            "Ep", "polytopes", ConvergenceTable::reduction_rate_log2, 2);
          convergence_table.evaluate_convergence_rates(
            "velocity_L2",
            "polytopes",
            ConvergenceTable::reduction_rate_log2,
            2);
          convergence_table.evaluate_convergence_rates(
            "pressure_O",
            "polytopes",
            ConvergenceTable::reduction_rate_log2,
            2);
          convergence_table.evaluate_convergence_rates(
            "pressure_D_L2",
            "polytopes",
            ConvergenceTable::reduction_rate_log2,
            2);
          convergence_table.evaluate_convergence_rates(
            "pressure_D_semiH1",
            "polytopes",
            ConvergenceTable::reduction_rate_log2,
            2);

          std::cout << "(deg_v, deg_p_O, deg_p_D) = (" << deg_v << ", "
                    << deg_p_O << ", " << deg_p_D << ")," << std::endl;
          convergence_table.write_text(std::cout);
          std::ofstream output("output/example4/" + type_str + "_convergence_table.vtk");
          convergence_table.write_text(output);
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
