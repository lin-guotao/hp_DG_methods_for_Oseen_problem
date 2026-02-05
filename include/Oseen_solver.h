#ifndef Oseen_solver_h
#define Oseen_solver_h

#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <agglomeration_handler.h>

namespace PolyOseenSolver
{
  using namespace dealii;

  template <int dim>
  struct Data
  {
    const TensorFunction<1, dim, double> *rhs_function  = nullptr;
    const TensorFunction<1, dim, double> *bcDirichlet   = nullptr;
    const TensorFunction<1, dim, double> *beta_function = nullptr;
  };

  template <int dim>
  class Solver
  {
  public:
    /**
     * @param agglo_handler The dof handler for agglomerated mesh.
     * @note Requires `define_agglomerate()` to be executed and `set_active_fe_index()`
     * to be set on all iterators beforehand.
     * @param viscosity_nu  The kinematic viscosity ($\nu$) of the fluid.
     * @param data          A structure containing physical functions (RHS, BC, Beta).
     * This is a mandatory argument. You must ensure that the function pointers
     * required for your simulation (e.g., boundary conditions) are valid and
     * assigned.
     * @param output_prefix Prefix for the output files.
     * @param q_c           The quadrature collection for cell integration.
     * Defaults to `QGauss<dim>(degree)` based on the FE index if not specified.
     * @param f_q_c         The quadrature collection for face integration.
     * Defaults to `QGauss<dim-1>(degree+1)` if not specified.
     */
    Solver(AgglomerationHandler<dim>     &agglo_handler,
           const double                   viscosity_nu,
           const Data<dim>               &data,
           const std::string              output_prefix = "Oseen",
           const hp::QCollection<dim>     q_c   = hp::QCollection<dim>(),
           const hp::QCollection<dim - 1> f_q_c = hp::QCollection<dim - 1>());

    // Run the simulation
    void
    run();

    /**
     * @brief Computes error norms against an exact solution.
     * @param exact_solution    The analytical solution function.
     * @param error_velocity_L2 Output: L2 norm of velocity error.
     * @param error_velocity_H1 Output: H1 semi-norm of velocity error.
     * @param error_pressure_L2 Output: L2 norm of pressure error.
     */
    void
    compute_errors(const Function<dim> &exact_solution,
                   double              &error_velocity_L2,
                   double              &error_velocity_H1,
                   double              &error_pressure_L2) const;

    /**
     * @brief Returns the total number of degrees of freedom.
     */
    inline unsigned int
    get_n_dofs() const;

  private:
    // Distributes DoFs and initializes sparsity patterns and matrices
    void
    setup_agglo_system();
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
    // Output results
    void
    output_results() const;

    // References to external large objects
    AgglomerationHandler<dim> &agglo_handler;
    const Triangulation<dim>  &triangulation;
    const Mapping<dim>        &mapping;

    // Parameters and pointers to functions
    const double                          viscosity_nu;
    const TensorFunction<1, dim, double> *rhs_function;
    const TensorFunction<1, dim, double> *bcDirichlet;
    const TensorFunction<1, dim, double> *beta_function;
    const std::string                     output_prefix;

    // Discretization components
    const hp::FECollection<dim> fe_collection;
    hp::QCollection<dim>        q_collection;
    hp::QCollection<dim - 1>    face_q_collection;

    // Linear Algebra Objects
    AffineConstraints<double> constraints;
    SparsityPattern           sparsity;
    SparseMatrix<double>      system_matrix;
    Vector<double>            solution;
    Vector<double>            system_rhs;

    // Constants
    const double penalty_constant_v = 40.0;
    const double penalty_constant_p = 1.0;

    // Solution vector used for error computation and visualization
    Vector<double> interpolated_solution;

    double domain_volume = 0.;

    bool is_solved = false;
  };

  template <int dim>
  inline unsigned int
  Solver<dim>::get_n_dofs() const
  {
    return agglo_handler.n_dofs();
  }
} // namespace PolyOseenSolver

#endif