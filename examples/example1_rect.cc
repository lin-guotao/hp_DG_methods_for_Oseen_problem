#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "Oseen_solver.h"

namespace example1_rect
{
  using namespace dealii;
  using std::cos;
  using std::exp;
  using std::pow;
  using std::sin;
  using std::sqrt;

  enum class MeshType
  {
    Mesh1,
    Mesh2,
    Mesh3,
    Mesh4
  };
  std::string
  Mesh_str(MeshType type)
  {
    static constexpr std::array<std::string_view, 4> names = {
      {"Mesh1", "Mesh2", "Mesh3", "Mesh4"}};

    return std::string(names[static_cast<size_t>(type)]);
  }

  enum class FEType
  {
    P1P1,
    P2P1,
    P2P2,
    P3P2
  };
  std::string
  FE_str(FEType type)
  {
    static constexpr std::array<std::string_view, 5> names = {
      {"P1P1", "P2P1", "P2P2", "P3P2"}};

    return std::string(names[static_cast<size_t>(type)]);
  }

  std::string
  Re_str(double Re)
  {
    std::string s = dealii::Utilities::to_string<double>(Re, 5);

    size_t pos = s.find('.');
    if (pos != std::string::npos)
      {
        s[pos] = 'p';
      }

    return "Re" + s;
  }

  /*============================================================================
     ExactSolution (Kovasznay flow)
     Represents the analytical solution (u, p) to the Oseen problem.

     Re: Reynolds number
     λ = Re/2 - sqrt(Re²/4 + 4π²)
     ν = 1 / Re

     The exact solution is:
       u₁(x,y) = 1 - exp(λx) * cos(2πy)
       u₂(x,y) = (λ / 2π) * exp(λx) * sin(2πy)
       p(x,y)  = 0.5 * exp(2λx) + C, where C ensures ∫_Ω p dx = 0

  --- Reference ---
    B. Cockburn, G. Kanschat, and D. Schötzau,
    "The Local Discontinuous Galerkin Method for the Oseen Equations",
    Mathematics of Computation, Vol. 73, No. 246, pp. 569–593, 2003.
  ============================================================================*/
  template <int dim>
  class DataFunctions
  {
    static constexpr double PI = dealii::numbers::PI;

  public:
    DataFunctions(const double Re)
      : Re(Re)
    {
      lambda        = Re / 2. - sqrt(Re * Re / 4. + 4. * PI * PI);
      mean_pressure = 1. / (8. * lambda) * (exp(3. * lambda) - exp(-lambda));
      nu            = 1. / Re;

      exact_solution =
        std::make_unique<const ExactSolution>(Re, lambda, mean_pressure);
      rhs_function  = std::make_unique<const RightHandSide>(Re, lambda, nu);
      bcDirichlet   = std::make_unique<const BoundaryDirichlet>(Re, lambda);
      beta_function = std::make_unique<const BetaFunction>(Re, lambda);

      data.rhs_function  = rhs_function.get();
      data.bcDirichlet   = bcDirichlet.get();
      data.beta_function = beta_function.get();
    }

    const PolyOseenSolver::Data<dim> &
    get_data() const
    {
      return data;
    }
    const Function<dim> &
    get_exactsol() const
    {
      return *exact_solution;
    }

  private:
    const double Re;
    double       lambda;
    double       mean_pressure;
    double       nu;

    std::unique_ptr<const Function<dim>>                  exact_solution;
    std::unique_ptr<const TensorFunction<1, dim, double>> rhs_function;
    std::unique_ptr<const TensorFunction<1, dim, double>> bcDirichlet;
    std::unique_ptr<const TensorFunction<1, dim, double>> beta_function;

    PolyOseenSolver::Data<dim> data;

    // DataFunctions::ExactSolution
    class ExactSolution : public Function<dim>
    {
    public:
      ExactSolution(const double Re,
                    const double lambda,
                    const double mean_pressure)
        : Function<dim>(dim + 1)
        , Re(Re)
        , lambda(lambda)
        , mean_pressure(mean_pressure)
      {}

      virtual void
      vector_value_list(const std::vector<Point<dim>> &points,
                        std::vector<Vector<double>> &value_list) const override
      {
        AssertDimension(points.size(), value_list.size());

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            const double x = points[i][0];
            const double y = points[i][1];

            value_list[i][0] = 1. - exp(lambda * x) * cos(2. * PI * y);
            value_list[i][1] =
              lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
            value_list[i][2] = 0.5 * exp(2. * lambda * x) - mean_pressure;
          }
      }

      virtual void
      vector_gradient_list(
        const std::vector<Point<dim>>            &points,
        std::vector<std::vector<Tensor<1, dim>>> &gradient_list) const override
      {
        AssertDimension(points.size(), gradient_list.size());

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            const double x = points[i][0];
            const double y = points[i][1];

            gradient_list[i][0][0] =
              -lambda * exp(lambda * x) * cos(2. * PI * y);
            gradient_list[i][0][1] =
              2. * PI * exp(lambda * x) * sin(2. * PI * y);

            gradient_list[i][1][0] =
              lambda * lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
            gradient_list[i][1][1] =
              lambda * exp(lambda * x) * cos(2. * PI * y);

            gradient_list[i][2][0] = lambda * exp(2. * lambda * x);
            gradient_list[i][2][1] = 0.;
          }
      }

    private:
      const double Re;
      const double lambda;
      const double mean_pressure;
    };

    // DataFunctions::RightHandSide
    class RightHandSide : public TensorFunction<1, dim, double>
    {
    public:
      RightHandSide(const double Re, const double lambda, const double nu)
        : TensorFunction<1, dim, double>(dim)
        , Re(Re)
        , lambda(lambda)
        , nu(nu)
      {}

      void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<1, dim>>   &value_list) const override
      {
        AssertDimension(points.size(), value_list.size());

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            const auto  &p = points[i];
            const double x = p[0];
            const double y = p[1];

            value_list[i][0] = nu * (lambda * lambda - 4. * PI * PI) *
                                 exp(lambda * x) * cos(2. * PI * y) +
                               lambda * exp(lambda * x) *
                                 (2. * exp(lambda * x) - cos(2. * PI * y));

            value_list[i][1] =
              nu * (2. * PI * lambda - pow(lambda, 3.) / (2. * PI)) *
                exp(lambda * x) * sin(2. * PI * y) +
              lambda * lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
          }
      }

    private:
      const double Re;
      const double lambda;
      const double nu;
    };

    // DataFunctions::BoundaryDirichlet
    class BoundaryDirichlet : public TensorFunction<1, dim, double>
    {
    public:
      BoundaryDirichlet(const double Re, const double lambda)
        : TensorFunction<1, dim, double>(dim)
        , Re(Re)
        , lambda(lambda)
      {}

      void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<1, dim>>   &value_list) const override
      {
        AssertDimension(points.size(), value_list.size());

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            const auto  &p = points[i];
            const double x = p[0];
            const double y = p[1];

            value_list[i][0] = 1. - exp(lambda * x) * cos(2. * PI * y);
            value_list[i][1] =
              lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
          }
      }

    private:
      const double Re;
      const double lambda;
    };

    // DataFunctions::BetaFunction
    class BetaFunction : public TensorFunction<1, dim, double>
    {
    public:
      BetaFunction(const double Re, const double lambda)
        : TensorFunction<1, dim, double>(dim)
        , Re(Re)
        , lambda(lambda)
      {}

      void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<1, dim>>   &value_list) const override
      {
        AssertDimension(points.size(), value_list.size());

        for (unsigned int i = 0; i < points.size(); ++i)
          {
            const auto  &p = points[i];
            const double x = p[0];
            const double y = p[1];

            value_list[i][0] = 1. - exp(lambda * x) * cos(2. * PI * y);
            value_list[i][1] =
              lambda / (2. * PI) * exp(lambda * x) * sin(2. * PI * y);
          }
      }

    private:
      const double Re;
      const double lambda;
    };
  }; // class DataFunctions

  template <int dim>
  class OseenProblem
  {
  public:
    OseenProblem(const MeshType mesh_type = MeshType::Mesh1,
                 const FEType   fe_type   = FEType::P3P2,
                 const double   Re        = 1.0);

    // Run the simulation
    void
    run();

  private:
    void
    make_base_grid();
    void
    make_agglo_grid(const unsigned int mesh_level);
    void
    setup_agglo_fe_space();

    const std::string example_name = "example1_rect";

    const MeshType mesh_type;
    const FEType   fe_type;
    const double   Re;
    const double   viscosity_nu;
    std::string    type_str;
    std::string    output_prefix;
    unsigned int   degree_v = 3;
    unsigned int   degree_p = 2;


    Triangulation<dim>                         triangulation;
    const MappingQ<dim>                        mapping;
    std::unique_ptr<AgglomerationHandler<dim>> agglo_handler;
    std::unique_ptr<GridTools::Cache<dim>>     cached_tria;

    hp::FECollection<dim>    fe_collection;
    hp::QCollection<dim>     q_collection;
    hp::QCollection<dim - 1> face_q_collection;

    DataFunctions<dim> data_functions;

    std::unique_ptr<PolyOseenSolver::Solver<dim>> oseen_solver;
  };

  template <int dim>
  OseenProblem<dim>::OseenProblem(const MeshType mesh_type,
                                  const FEType   fe_type,
                                  const double   Re)
    : mesh_type(mesh_type)
    , fe_type(fe_type)
    , Re(Re)
    , viscosity_nu(1. / Re)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , mapping(1)
    , data_functions(Re)
  {
    switch (fe_type)
      {
        case FEType::P1P1:
          degree_v = 1;
          degree_p = 1;
          break;
        case FEType::P2P1:
          degree_v = 2;
          degree_p = 1;
          break;
        case FEType::P2P2:
          degree_v = 2;
          degree_p = 2;
          break;
        case FEType::P3P2:
          degree_v = 3;
          degree_p = 2;
          break;
        default:
          AssertThrow(
            false, ExcMessage("Only P1P1, P2P1, P2P2 and P3P2 are supported."));
      }

    FESystem<dim> Oseen_fe(FE_AggloDGP<dim>(degree_v) ^ dim,
                           FE_AggloDGP<dim>(degree_p));
    fe_collection.push_back(Oseen_fe);

    const QGauss<dim> quadrature(degree_v);
    q_collection.push_back(quadrature);
    const QGauss<dim - 1> face_quadrature(degree_v + 1);
    face_q_collection.push_back(face_quadrature);

    type_str = example_name + "_" + Mesh_str(mesh_type) + "_" +
               FE_str(fe_type) + "_" + Re_str(Re);

    const std::string output_path = "output/" + example_name;
    if (!std::filesystem::exists(output_path))
      std::filesystem::create_directories(output_path);
    output_prefix = output_path + "/" + type_str;
  }

  /*============================================================================
    Create a fine background grid for cell agglomeration.
  ============================================================================*/
  template <int dim>
  void
  OseenProblem<dim>::make_base_grid()
  {
    std::cout << "   Making base grid..." << std::endl;
    Point<2>                  bottom_left(-0.5, 0.0);
    Point<2>                  top_right(1.5, 2.0);
    std::vector<unsigned int> subdivisions = {2, 2};
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              subdivisions,
                                              bottom_left,
                                              top_right);

    const std::string cachefilename = output_prefix + "_cached_base";
    if (std::filesystem::exists(cachefilename + "_triangulation.data"))
      {
        std::cout << "     Loading cached base grid from " << cachefilename
                  << " ..." << std::endl;
        triangulation.load(cachefilename);
        if (mesh_type == MeshType::Mesh2 || mesh_type == MeshType::Mesh4)
          {
            std::cout << "     Setting curved manifold..." << std::endl;
            triangulation.set_manifold(
              1, PolarManifold<2>(Point<2>(-0.5, 1.5))); // left top
            triangulation.set_manifold(
              2, PolarManifold<2>(Point<2>(1.5, 0.5))); // right bottom
            triangulation.set_manifold(
              3, PolarManifold<2>(Point<2>(1., 2.))); // right top
            triangulation.set_manifold(
              4, PolarManifold<2>(Point<2>(0., 0.))); // left bottom
          }
      }
    else
      {
        std::cout << "     Cached base grid not found. Generating new grid..."
                  << std::endl;

        if (mesh_type == MeshType::Mesh2 || mesh_type == MeshType::Mesh4)
          {
            std::cout << "     Setting curved manifold..." << std::endl;
            triangulation.set_manifold(1,
                                       PolarManifold<2>(Point<2>(-0.5, 1.5)));
            triangulation.set_manifold(2, PolarManifold<2>(Point<2>(1.5, 0.5)));
            triangulation.set_manifold(3, PolarManifold<2>(Point<2>(1., 2.)));
            triangulation.set_manifold(4, PolarManifold<2>(Point<2>(0., 0.)));
          }

        for (const auto &cell : triangulation.active_cell_iterators())
          {
            if (mesh_type == MeshType::Mesh1 || mesh_type == MeshType::Mesh3)
              cell->set_material_id(0);
            else
              {
                if (cell->center()[0] < 0.5 && cell->center()[1] > 1.)
                  cell->set_material_id(0);
                if (cell->center()[0] > 0.5 && cell->center()[1] > 1.)
                  cell->set_material_id(1);
                if (cell->center()[0] < 0.5 && cell->center()[1] < 1.)
                  cell->set_material_id(2);
                if (cell->center()[0] > 0.5 && cell->center()[1] < 1.)
                  cell->set_material_id(3);
              }

            if (mesh_type == MeshType::Mesh2 || mesh_type == MeshType::Mesh4)
              for (unsigned int f = 0; f < 4; ++f)
                if (!cell->at_boundary(f))
                  {
                    if ((cell->face(f)->center()[0] > 0.1) &&
                        (cell->face(f)->center()[0] < 0.9))
                      {
                        if (cell->face(f)->center()[1] > 1.)
                          cell->face(f)->set_all_manifold_ids(1);
                        else
                          cell->face(f)->set_all_manifold_ids(2);
                      }

                    if ((cell->face(f)->center()[1] > 0.6) &&
                        (cell->face(f)->center()[1] < 1.4))
                      {
                        if (cell->face(f)->center()[0] > 0.5)
                          cell->face(f)->set_all_manifold_ids(3);
                        else
                          cell->face(f)->set_all_manifold_ids(4);
                      }
                  }
          }
        triangulation.refine_global(7);

        triangulation.save(cachefilename);
        std::cout << "     Saved grid to " << cachefilename << std::endl;
      }

    std::cout << "     Size of base grid: " << triangulation.n_active_cells()
              << std::endl;
  }

  /*============================================================================
    Generate an agglomerated mesh with different partitioning strategies.
  ============================================================================*/
  template <int dim>
  void
  OseenProblem<dim>::make_agglo_grid(const unsigned int mesh_level)
  {
    std::cout << "   Making agglo grid..." << std::endl;
    cached_tria =
      std::make_unique<GridTools::Cache<dim>>(triangulation, mapping);
    agglo_handler = std::make_unique<AgglomerationHandler<dim>>(*cached_tria);

    if (mesh_type == MeshType::Mesh1 || mesh_type == MeshType::Mesh2)
      {
        std::cout << "     Partition with Rtree." << std::endl;
        namespace bgi = boost::geometry::index;
        static constexpr unsigned int max_elem_per_node =
          PolyUtils::constexpr_pow(2, dim);

        const unsigned int num_domain = (mesh_type == MeshType::Mesh1) ? 1 : 4;
        std::vector<std::vector<
          std::pair<BoundingBox<dim>,
                    typename Triangulation<dim>::active_cell_iterator>>>
          all_boxes(num_domain);
        // To preserve the curved boundaries of the domains, we use separate
        // R-trees for each domain. A "boxes" is a collection of bounding boxes
        // for a single domain. "all_boxes" is a collection of all such "boxes".
        for (const auto &cell : triangulation.active_cell_iterators())
          all_boxes[cell->material_id()].emplace_back(
            mapping.get_bounding_box(cell), cell);

        for (unsigned int i = 0; i < num_domain; ++i)
          {
            auto tree = pack_rtree<bgi::rstar<max_elem_per_node>>(all_boxes[i]);

            std::cout << "     Total number of available levels in domain_" << i
                      << ": " << n_levels(tree) << std::endl;

            const unsigned int extraction_level_sub =
              std::min(mesh_level - num_domain / max_elem_per_node,
                       n_levels(tree));

            CellsAgglomerator<dim, decltype(tree)> agglomerator{
              tree, extraction_level_sub};
            const auto vec_agglomerates = agglomerator.extract_agglomerates();
            for (const auto &agglo : vec_agglomerates)
              agglo_handler->define_agglomerate(agglo, fe_collection.size());
          }
      }
    else if (mesh_type == MeshType::Mesh3 || mesh_type == MeshType::Mesh4)
      {
        std::cout << "     Partition with Metis." << std::endl;
        const unsigned int n_subdomains = (int)pow(4, mesh_level);

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
        for (const auto &cell : triangulation.active_cell_iterators())
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

        // Finalize the connectivity graph and call METIS to partition it.
        // Each cell is assigned to one of the n_subdomains.
        SparsityPattern sp_cell_connectivity;
        sp_cell_connectivity.copy_from(cell_connectivity);
        std::vector<unsigned int> partition_indices(
          triangulation.n_active_cells());
        SparsityTools::partition(sp_cell_connectivity,
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

        // For every subdomain, agglomerate elements together
        for (std::size_t i = 0; i < n_subdomains; ++i)
          agglo_handler->define_agglomerate(cells_per_subdomain[i],
                                            fe_collection.size());
      }
    else
      {
        AssertThrow(false,
                    ExcMessage("Only Mesh1 through Mesh4 are supported."));
      }

    std::cout << "     N subdomains = " << agglo_handler->n_agglomerates()
              << std::endl;
  }

  template <int dim>
  void
  OseenProblem<dim>::setup_agglo_fe_space()
  {
    std::cout << "   Setting up FE space..." << std::endl;
    for (const auto &polytope : agglo_handler->polytope_iterators())
      polytope->set_active_fe_index(0);

    agglo_handler->distribute_agglomerated_dofs(fe_collection);
  }

  template <int dim>
  void
  OseenProblem<dim>::run()
  {
    ConvergenceTable convergence_table;
    std::cout << std::scientific << std::setprecision(2);

    std::cout << "\n" << std::string(type_str.length(), '-') << std::endl;
    std::cout << type_str << std::endl;
    std::cout << std::string(type_str.length(), '-') << std::endl;

    make_base_grid();
    for (unsigned int mesh_level = 5; mesh_level < 8; ++mesh_level)
      {
        std::cout << "level " << mesh_level << std::endl;
        convergence_table.add_value("level", mesh_level);
        convergence_table.add_value("polytopes", (int)std::pow(4, mesh_level));

        double error_velocity_L2 = 0.;
        double error_velocity_H1 = 0.;
        double error_pressure_L2 = 0.;
        if (mesh_level < 8)
          {
            make_agglo_grid(mesh_level);
            setup_agglo_fe_space();

            oseen_solver =
              std::make_unique<PolyOseenSolver::Solver<dim>>(*agglo_handler,
                                                   viscosity_nu,
                                                   data_functions.get_data(),
                                                   output_prefix,
                                                   q_collection,
                                                   face_q_collection);
            oseen_solver->run();
            oseen_solver->compute_errors(data_functions.get_exactsol(),
                                         error_velocity_L2,
                                         error_velocity_H1,
                                         error_pressure_L2);
          }
        else
          AssertThrow(
            false,
            ExcMessage(
              "You need to refine the base grid to use higher mesh levels."
              "Please modify make_base_grid()."));

        convergence_table.add_value("dofs", oseen_solver->get_n_dofs());
        convergence_table.add_value("velocity_L2", error_velocity_L2);
        convergence_table.add_value("velocity_H1", error_velocity_H1);
        convergence_table.add_value("pressure_L2", error_pressure_L2);
      }

    for (const std::string col_name :
         {"velocity_L2", "velocity_H1", "pressure_L2"})
      {
        convergence_table.set_precision(col_name, 2);
        convergence_table.set_scientific(col_name, true);
        convergence_table.evaluate_convergence_rates(
          col_name, "polytopes", ConvergenceTable::reduction_rate_log2, 2);
      }

    std::cout << "(deg_v, deg_p) = (" << degree_v << ", " << degree_p << "),"
              << std::endl
              << std::fixed << std::setprecision(1) << "Re = " << Re << ","
              << std::endl;
    convergence_table.write_text(std::cout);
    std::ofstream output(output_prefix + "_convergence_table.vtk");
    convergence_table.write_text(output);
  }
} // namespace example1_rect

int
main()
{
  try
    {
      using namespace dealii;
      using example1_rect::FEType;
      using example1_rect::MeshType;

      for (const MeshType mesh_type :
           {MeshType::Mesh1, MeshType::Mesh2, MeshType::Mesh3, MeshType::Mesh4})
        {
          example1_rect::OseenProblem<2> oseen_problem(mesh_type,
                                                  FEType::P3P2,
                                                  /*Re =*/1.0);
          oseen_problem.run();
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
