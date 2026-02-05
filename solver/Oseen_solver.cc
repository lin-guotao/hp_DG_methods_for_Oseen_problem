#include "Oseen_solver.h"

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <agglomeration_handler.h>
#include <fe_agglodgp.h>
#include <poly_utils.h>

#include <algorithm>
#include <fstream>
#include <iostream>


/*============================================================================
   Oseen Problem Formulation (steady, incompressible)

   Given a domain Ω ⊂ ℝ^d, find a velocity field u : Ω → ℝ^d
   and a pressure field p : Ω → ℝ such that:

       -ν Δu + (β · ∇)u + ∇p = f      in Ω       (momentum equation)
                        ∇ · u = 0      in Ω       (mass conservation)
                             u = g     on ∂Ω       (Dirichlet boundary)

   where:
       - u: velocity vector field
       - p: pressure scalar field, with ∫_Ω p dx = 0
       - f: external body force (RightHandSide)
       - g: prescribed boundary velocity (BoundaryDirichlet)
       - ν: kinematic viscosity
       - β: prescribed convection (advection) velocity field (BetaFunction)
       - Δu: Laplacian of u, i.e., component-wise second derivatives
============================================================================*/
namespace PolyOseenSolver
{
  using namespace dealii;

  template <int dim>
  Solver<dim>::Solver(AgglomerationHandler<dim>     &agglo_handler,
                      const double                   viscosity_nu,
                      const Data<dim>               &data,
                      const std::string              output_prefix,
                      const hp::QCollection<dim>     q_c,
                      const hp::QCollection<dim - 1> f_q_c)
    : agglo_handler(agglo_handler)
    , triangulation(agglo_handler.get_triangulation())
    , mapping(agglo_handler.get_mapping())
    , viscosity_nu(viscosity_nu)
    , rhs_function(data.rhs_function)
    , bcDirichlet(data.bcDirichlet)
    , beta_function(data.beta_function)
    , output_prefix(output_prefix)
    , fe_collection(agglo_handler.get_fe_collection())
    , q_collection(q_c)
    , face_q_collection(f_q_c)
  {
    if (q_collection.size() == 0)
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        {
          const unsigned int degree = fe_collection[i].degree;
          q_collection.push_back(QGauss<dim>(degree));
        }

    if (face_q_collection.size() == 0)
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        {
          const unsigned int degree = fe_collection[i].degree;
          face_q_collection.push_back(QGauss<dim - 1>(degree + 1));
        }

    AssertThrow(
      (q_collection.size() == fe_collection.size() ||
       q_collection.size() == 1) &&
        (face_q_collection.size() == fe_collection.size() ||
         face_q_collection.size() == 1),
      ExcMessage(
        "Both cell and face quadrature collections must have a size of 1 "
        "or match the number of finite elements."));

    const MappingQ<dim> higher_order_mapping(3); // Use a higher-order mapping for volume computation
    domain_volume = GridTools::volume(triangulation, higher_order_mapping);
  }

  template <int dim>
  void
  Solver<dim>::setup_agglo_system()
  {
    DynamicSparsityPattern dsp;
    constraints.clear();
    zero_pressure_dof_constraint();
    constraints.close();
    agglo_handler.create_agglomeration_sparsity_pattern(dsp, constraints);
    sparsity.copy_from(dsp);
  }

  template <int dim>
  void
  Solver<dim>::assemble_system()
  {
    system_matrix.reinit(sparsity);
    solution.reinit(agglo_handler.n_dofs());
    system_rhs.reinit(agglo_handler.n_dofs());

    agglo_handler.initialize_fe_values(q_collection,
                                       update_gradients | update_JxW_values |
                                         update_quadrature_points |
                                         update_JxW_values | update_values,
                                       face_q_collection);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    // Loop over all agglomerated polytopes (cells)
    for (const auto &polytope : agglo_handler.polytope_iterators())
      {
        const unsigned int current_dofs_per_cell =
          polytope->get_fe().dofs_per_cell;
        FullMatrix<double>                   cell_matrix(current_dofs_per_cell,
                                       current_dofs_per_cell);
        Vector<double>                       cell_rhs(current_dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(
          current_dofs_per_cell);

        cell_matrix              = 0;
        cell_rhs                 = 0;
        const auto &agglo_values = agglo_handler.reinit(polytope);
        polytope->get_dof_indices(local_dof_indices);

        const auto        &q_points  = agglo_values.get_quadrature_points();
        const unsigned int n_qpoints = q_points.size();
        std::vector<Tensor<1, dim>> rhs(n_qpoints);
        std::vector<Tensor<1, dim>> beta(n_qpoints);
        rhs_function->value_list(q_points, rhs);
        beta_function->value_list(q_points, beta);

        std::vector<Tensor<1, dim>> phi_u(current_dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(current_dofs_per_cell);
        std::vector<double>         div_phi_u(current_dofs_per_cell);
        std::vector<double>         phi_p(current_dofs_per_cell);

        for (unsigned int q_index : agglo_values.quadrature_point_indices())
          {
            for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
              {
                phi_u[k]      = agglo_values[velocities].value(k, q_index);
                grad_phi_u[k] = agglo_values[velocities].gradient(k, q_index);
                div_phi_u[k]  = agglo_values[velocities].divergence(k, q_index);
                phi_p[k]      = agglo_values[pressure].value(k, q_index);
              }

            for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                  {
                    cell_matrix(i, j) +=
                      (viscosity_nu * scalar_product(grad_phi_u[i],
                                                     grad_phi_u[j]) // + ν ∇v:∇u
                       - div_phi_u[i] * phi_p[j]                    // - ∇·v p
                       + phi_p[i] * div_phi_u[j]                    // + ∇·u q
                       - (grad_phi_u[i] * beta[q_index]) *
                           phi_u[j] // - u · (β·∇)v
                       ) *
                      agglo_values.JxW(q_index); // dx
                    // + ν ∫ ∇v : ∇u dx
                    // -   ∫ (∇·v) p dx
                    // +   ∫ (∇·u) q dx
                    // +   ∫ v · (β·∇u) dx
                  }
                cell_rhs(i) += phi_u[i] * rhs[q_index] *
                               agglo_values.JxW(q_index); // ∫ v·f dx
              }
          }

        // Loop over faces of the current polytope
        const unsigned int n_faces = polytope->n_faces();
        AssertThrow(n_faces > 0,
                    ExcMessage(
                      "Invalid element: at least 4 faces are required."));

        auto polygon_boundary_vertices = polytope->polytope_boundary();

        for (unsigned int f = 0; f < n_faces; ++f)
          {
            if (polytope->at_boundary(f))
              {
                // Handle boundary faces
                const auto &fe_face       = agglo_handler.reinit(polytope, f);
                const auto &face_q_points = fe_face.get_quadrature_points();

                std::vector<Tensor<2, dim>> aver_grad_phi_v(
                  current_dofs_per_cell);
                std::vector<Tensor<1, dim>> jump_phi_v(current_dofs_per_cell);
                std::vector<double>         aver_phi_p(current_dofs_per_cell);
                std::vector<double>         jump_phi_p(current_dofs_per_cell);

                std::vector<Tensor<1, dim>> g(face_q_points.size());
                bcDirichlet->value_list(face_q_points, g);
                std::vector<Tensor<1, dim>> beta(face_q_points.size());
                beta_function->value_list(face_q_points, beta);

                // Get normal vectors seen from each agglomeration.
                const auto &normals = fe_face.get_normal_vectors();

                unsigned int deg_v_current =
                  polytope->get_fe().get_sub_fe(0, 1).degree;
                double tau_cell = (viscosity_nu) * (deg_v_current + 1) *
                                  (deg_v_current + dim) /
                                  std::fabs(polytope->diameter());
                double sigma_v = penalty_constant_v * tau_cell;

                for (unsigned int q_index : fe_face.quadrature_point_indices())
                  {
                    // double is_face_inflow_of_cell =
                    //   (beta[q_index] * normals[q_index]) < 0 ? 1.0 : 0.0;

                    for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                      {
                        aver_grad_phi_v[k] =
                          fe_face[velocities].gradient(k, q_index);
                        jump_phi_v[k] = fe_face[velocities].value(k, q_index);
                        aver_phi_p[k] = fe_face[pressure].value(k, q_index);
                      }

                    for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                      {
                        for (unsigned int j = 0; j < current_dofs_per_cell; ++j)
                          {
                            cell_matrix(i, j) +=
                              (-viscosity_nu * jump_phi_v[i] *
                                 (aver_grad_phi_v[j] *
                                  normals[q_index]) // - ν [v]·({∇u}·n)
                               - viscosity_nu * jump_phi_v[j] *
                                   (aver_grad_phi_v[i] *
                                    normals[q_index]) // - ν [u]·({∇v}·n)
                               + sigma_v * jump_phi_v[i] *
                                   jump_phi_v[j] // + σ_v [v]·[u]
                               + aver_phi_p[j] * jump_phi_v[i] *
                                   normals[q_index] // + [v]·n {p}
                               - aver_phi_p[i] * jump_phi_v[j] *
                                   normals[q_index] // - [u]·n {q}
                               //  - is_face_inflow_of_cell * // inflow faces
                               //  only
                               //      (beta[q_index] * normals[q_index]) *
                               //      jump_phi_v[j] *
                               //      jump_phi_v[i] // - (β·n) v_down·[u]
                               //                    // v_down = [v] at inflow
                               //                    // boundary
                               + 0.5 *
                                   (beta[q_index] * normals[q_index] +
                                    beta[q_index].norm()) *
                                   jump_phi_v[j] *
                                   jump_phi_v[i] // + 0.5 (β·n + Cb) [v]·[u]
                               ) *
                              fe_face.JxW(q_index); // ds
                            // - ν ∫    [v] · ({∇u} · n) ds
                            // - ν ∫    [u] · ({∇v} · n) ds
                            // +   ∫    σ_v [v] · [u] ds
                            // +   ∫    [v] · n · {p} ds
                            // -   ∫    [u] · n · {q} ds
                            // -   ∫_in (β · n) v_down · [u] ds
                            //
                            // where:
                            //   [·]      = jump across face; equals value
                            //              on current cell at boundary
                            //   {·}      = average across face; equals
                            //              value on current cell at boundary
                            //   v_down   = value from downwind side,
                            //              taken as v_current at inflow
                            //              boundary
                            //   ∫_in     = integral over inflow faces
                            //              where (β · n) < 0
                            //
                            // Note:
                            //  Although inflow (upwind) directions usually
                            //  matter more for stability,
                            // here we integrate over faces, not cells.
                            // The downwind-side cell regards the face as an
                            // inflow face. Thus, convection terms on each face
                            // should couple to the downwind-side cell.
                          }
                        cell_rhs(i) +=
                          (-viscosity_nu * g[q_index] *
                             (aver_grad_phi_v[i] *
                              normals[q_index]) // - ν g · ({∇v} · n)
                           +
                           sigma_v * g[q_index] * jump_phi_v[i] // + σ_v g · [v]
                           - aver_phi_p[i] * g[q_index] *
                               normals[q_index] // - {q} (g · n)
                                                //  - is_face_inflow_of_cell *
                           //      (beta[q_index] * normals[q_index]) *
                           //      g[q_index] * jump_phi_v[i] // - (β·n) g · [v]
                           - 0.5 *
                               (beta[q_index] * normals[q_index] -
                                beta[q_index].norm()) *
                               g[q_index] *
                               jump_phi_v[i] // - 0.5 (β·n - Cb) g · [v]
                           ) *
                          fe_face.JxW(q_index); // ds
                        // - ∫     ν g · ({∇v} · n) ds
                        // + ∫     σ_v g · [v] ds
                        // - ∫     {q} (g · n) ds
                        // - ∫_in  (β · n) g · [v] ds
                      }
                  }
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

                    unsigned int nofn =
                      polytope->neighbor_of_agglomerated_neighbor(f);

                    const auto &fe_faces = agglo_handler.reinit_interface(
                      polytope, neigh_polytope, f, nofn);

                    const auto &fe_faces0 = fe_faces.first;
                    const auto &fe_faces1 = fe_faces.second;

                    std::vector<types::global_dof_index>
                      local_dof_indices_neighbor(neigh_dofs_per_cell);

                    // Next, we define the four dofsxdofs matrices needed to
                    // assemble jumps and averages.
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

                    const auto &normals = fe_faces0.get_normal_vectors();

                    std::vector<Tensor<2, dim>> aver_grad_phi_v0(
                      current_dofs_per_cell);
                    std::vector<Tensor<1, dim>> aver_phi_v0(
                      current_dofs_per_cell);
                    std::vector<Tensor<1, dim>> jump_phi_v0(
                      current_dofs_per_cell);
                    std::vector<double> aver_phi_p0(current_dofs_per_cell);
                    std::vector<double> jump_phi_p0(current_dofs_per_cell);
                    // std::vector<Tensor<1, dim>> downwind_phi_v0(
                    //   current_dofs_per_cell);

                    std::vector<Tensor<2, dim>> aver_grad_phi_v1(
                      neigh_dofs_per_cell);
                    std::vector<Tensor<1, dim>> aver_phi_v1(
                      neigh_dofs_per_cell);
                    std::vector<Tensor<1, dim>> jump_phi_v1(
                      neigh_dofs_per_cell);
                    std::vector<double> aver_phi_p1(neigh_dofs_per_cell);
                    std::vector<double> jump_phi_p1(neigh_dofs_per_cell);
                    // std::vector<Tensor<1, dim>> downwind_phi_v1(
                    //   neigh_dofs_per_cell);

                    std::vector<Tensor<1, dim>> beta(
                      fe_faces0.n_quadrature_points);
                    beta_function->value_list(fe_faces0.get_quadrature_points(),
                                              beta);

                    unsigned int deg_v_current =
                      polytope->get_fe().get_sub_fe(0, 1).degree;
                    unsigned int deg_v_neigh =
                      neigh_polytope->get_fe().get_sub_fe(0, 1).degree;
                    double tau_current = (viscosity_nu) * (deg_v_current + 1) *
                                         (deg_v_current + dim) /
                                         std::fabs(polytope->diameter());
                    double tau_neigh = (viscosity_nu) * (deg_v_neigh + 1) *
                                       (deg_v_neigh + dim) /
                                       std::fabs(neigh_polytope->diameter());
                    double sigma_v =
                      penalty_constant_v * std::max(tau_current, tau_neigh);

                    double beta_max = 0;
                    for (unsigned int q_index = 0;
                         q_index < fe_faces0.n_quadrature_points;
                         ++q_index)
                      {
                        double beta_max0 = beta[q_index].norm();
                        if (beta_max0 > beta_max)
                          beta_max = beta_max0;
                      }
                    double zeta_current =
                      1. / (viscosity_nu / polytope->diameter() + beta_max);
                    double zeta_neigh =
                      1. /
                      (viscosity_nu / neigh_polytope->diameter() + beta_max);
                    double sigma_p =
                      penalty_constant_p * std::max(zeta_current, zeta_neigh);

                    for (unsigned int q_index = 0;
                         q_index < fe_faces0.n_quadrature_points;
                         ++q_index)
                      {
                        // bool is_face_inflow_of_cell =
                        //   ((beta[q_index] * normals[q_index]) < 0);

                        for (unsigned int k = 0; k < current_dofs_per_cell; ++k)
                          {
                            aver_grad_phi_v0[k] =
                              0.5 * fe_faces0[velocities].gradient(k, q_index);
                            aver_phi_v0[k] =
                              0.5 * fe_faces0[velocities].value(k, q_index);
                            jump_phi_v0[k] =
                              fe_faces0[velocities].value(k, q_index);
                            aver_phi_p0[k] =
                              0.5 * fe_faces0[pressure].value(k, q_index);
                            jump_phi_p0[k] =
                              fe_faces0[pressure].value(k, q_index);
                            // if (is_face_inflow_of_cell)
                            //   downwind_phi_v0[k] =
                            //     fe_faces0[velocities].value(k, q_index);
                            // else
                            //   downwind_phi_v0[k] =
                            //     -fe_faces0[velocities].value(k, q_index);
                          }

                        for (unsigned int k = 0; k < neigh_dofs_per_cell; ++k)
                          {
                            aver_grad_phi_v1[k] =
                              0.5 * fe_faces1[velocities].gradient(k, q_index);
                            aver_phi_v1[k] =
                              0.5 * fe_faces1[velocities].value(k, q_index);
                            jump_phi_v1[k] =
                              -fe_faces1[velocities].value(k, q_index);
                            aver_phi_p1[k] =
                              0.5 * fe_faces1[pressure].value(k, q_index);
                            jump_phi_p1[k] =
                              -fe_faces1[pressure].value(k, q_index);
                            // if (is_face_inflow_of_cell)
                            //   downwind_phi_v1[k] =
                            //     -fe_faces1[velocities].value(k, q_index);
                            // else
                            //   downwind_phi_v1[k] =
                            //     fe_faces1[velocities].value(k, q_index);
                          }

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < current_dofs_per_cell;
                                 ++j)
                              {
                                M11(i, j) +=
                                  (-viscosity_nu * jump_phi_v0[i] *
                                     (aver_grad_phi_v0[j] *
                                      normals[q_index]) // - ν [v] · ({∇u} · n)
                                   -
                                   viscosity_nu * jump_phi_v0[j] *
                                     (aver_grad_phi_v0[i] *
                                      normals[q_index]) // - ν [u] · ({∇v} · n)
                                   + sigma_v * jump_phi_v0[i] *
                                       jump_phi_v0[j] // + σ_v [v] · [u]
                                   + aver_phi_p0[j] * jump_phi_v0[i] *
                                       normals[q_index] // + [v] · n · {p}
                                   - aver_phi_p0[i] * jump_phi_v0[j] *
                                       normals[q_index] // - [u] · n · {q}
                                   + sigma_p * jump_phi_p0[i] *
                                       jump_phi_p0[j] // + σ_p [p] · [q]
                                   //  - (beta[q_index] * normals[q_index]) *
                                   //      jump_phi_v0[j] *
                                   //      downwind_phi_v0[i] // - (β·n)
                                   //      v_down·[u]
                                   + (beta[q_index] * normals[q_index]) *
                                       aver_phi_v0[j] * jump_phi_v0[i] +
                                   0.5 * beta[q_index].norm() * jump_phi_v0[j] *
                                     jump_phi_v0[i] // + numerical flux
                                                    // (β·n){u}·[v] + 0.5 Cb
                                                    // [u]·[v]
                                   ) *
                                  fe_faces0.JxW(q_index); // ds
                                // - ν ∫    [v] · ({∇u} · n) ds
                                // - ν ∫    [u] · ({∇v} · n) ds
                                // + ∫     σ_v [v] · [u] ds
                                // + ∫     [v] · n · {p} ds
                                // - ∫     [u] · n · {q} ds
                                // + ∫     σ_p [p] · [q] ds
                                // - ∫_in  (β · n) v_down · [u] ds
                                //
                                // where:
                                //   [·]      = jump across face; equals value
                                //              on current cell at boundary
                                //   {·}      = average across face; equals
                                //              value on current cell at
                                //              boundary
                                //   v_down   = value from downwind side,
                                //              taken as v_current at inflow
                                //              boundary
                                //   ∫_in     = integral over inflow faces
                                //              where (β · n) < 0
                                //   σ_v      = velocity penalty parameter
                                //   σ_p      = pressure penalty parameter
                                //
                                // Note:
                                //   Suffix '0' denotes basis functions of the
                                //   current cell. M11 involves only basis
                                //   functions from the current cell.
                              }
                          }

                        for (unsigned int i = 0; i < current_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < neigh_dofs_per_cell;
                                 ++j)
                              {
                                M12(i, j) +=
                                  (-viscosity_nu * jump_phi_v0[i] *
                                     (aver_grad_phi_v1[j] * normals[q_index]) -
                                   viscosity_nu * jump_phi_v1[j] *
                                     (aver_grad_phi_v0[i] * normals[q_index]) +
                                   sigma_v * jump_phi_v0[i] * jump_phi_v1[j] +
                                   aver_phi_p1[j] * jump_phi_v0[i] *
                                     normals[q_index] -
                                   aver_phi_p0[i] * jump_phi_v1[j] *
                                     normals[q_index] +
                                   sigma_p * jump_phi_p0[i] * jump_phi_p1[j]
                                   //  - (beta[q_index] * normals[q_index]) *
                                   //    jump_phi_v1[j] * downwind_phi_v0[i]
                                   + (beta[q_index] * normals[q_index]) *
                                       aver_phi_v1[j] * jump_phi_v0[i] +
                                   0.5 * beta[q_index].norm() * jump_phi_v1[j] *
                                     jump_phi_v0[i]) *
                                  fe_faces0.JxW(q_index);
                                // Same structure as M11; only the basis
                                // functions differ.
                                //
                                // Suffix '1' refers to neighbor cell basis
                                // functions, while suffix '0' refers to current
                                // cell basis functions.
                                //
                                // Index [j] corresponds to trial functions,
                                // and [i] to test functions.
                                //
                                // In M21, all [i] indices are associated with
                                // suffix '1', indicating test functions from
                                // the neighbor cell and trial functions from
                                // the current cell.
                              }
                          }

                        for (unsigned int i = 0; i < neigh_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < current_dofs_per_cell;
                                 ++j)
                              {
                                M21(i, j) +=
                                  (-viscosity_nu * jump_phi_v1[i] *
                                     (aver_grad_phi_v0[j] * normals[q_index]) -
                                   viscosity_nu * jump_phi_v0[j] *
                                     (aver_grad_phi_v1[i] * normals[q_index]) +
                                   sigma_v * jump_phi_v1[i] * jump_phi_v0[j] +
                                   aver_phi_p0[j] * jump_phi_v1[i] *
                                     normals[q_index] -
                                   aver_phi_p1[i] * jump_phi_v0[j] *
                                     normals[q_index] +
                                   sigma_p * jump_phi_p1[i] * jump_phi_p0[j]
                                   //  - (beta[q_index] * normals[q_index]) *
                                   //    jump_phi_v0[j] * downwind_phi_v1[i]
                                   + (beta[q_index] * normals[q_index]) *
                                       aver_phi_v0[j] * jump_phi_v1[i] +
                                   0.5 * beta[q_index].norm() * jump_phi_v0[j] *
                                     jump_phi_v1[i]) *
                                  fe_faces0.JxW(q_index);
                                // Same structure as M11; only the basis
                                // functions differ.
                                //
                                // Suffix '1' refers to neighbor cell basis
                                // functions, while suffix '0' refers to current
                                // cell basis functions.
                                //
                                // Index [j] corresponds to trial functions,
                                // and [i] to test functions.
                                //
                                // In M21, [j] indices use suffix '0',
                                // indicating trial functions from the current
                                // cell, while [i] indices use suffix '1',
                                // indicating test functions from the neighbor
                                // cell.
                              }
                          }

                        for (unsigned int i = 0; i < neigh_dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < neigh_dofs_per_cell;
                                 ++j)
                              {
                                M22(i, j) +=
                                  (-viscosity_nu * jump_phi_v1[i] *
                                     (aver_grad_phi_v1[j] * normals[q_index]) -
                                   viscosity_nu * jump_phi_v1[j] *
                                     (aver_grad_phi_v1[i] * normals[q_index]) +
                                   sigma_v * jump_phi_v1[i] * jump_phi_v1[j] +
                                   aver_phi_p1[j] * jump_phi_v1[i] *
                                     normals[q_index] -
                                   aver_phi_p1[i] * jump_phi_v1[j] *
                                     normals[q_index] +
                                   sigma_p * jump_phi_p1[i] * jump_phi_p1[j]
                                   //  - (beta[q_index] * normals[q_index]) *
                                   //    jump_phi_v1[j] * downwind_phi_v1[i]
                                   + (beta[q_index] * normals[q_index]) *
                                       aver_phi_v1[j] * jump_phi_v1[i] +
                                   0.5 * beta[q_index].norm() * jump_phi_v1[j] *
                                     jump_phi_v1[i]) *
                                  fe_faces0.JxW(q_index);
                                // Same structure as M11; only the basis
                                // functions differ.
                                //
                                // Suffix '1' refers to neighbor cell basis
                                // functions, while suffix '0' refers to current
                                // cell basis functions.
                                //
                                // Index [j] corresponds to trial functions,
                                // and [i] to test functions.
                                //
                                // In M22, both test and trial functions use
                                // suffix '1', meaning they are associated with
                                // the neighbor cell only.
                              }
                          }
                      }

                    neigh_polytope->get_dof_indices(local_dof_indices_neighbor);

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
  Solver<dim>::solve()
  {
    constraints.condense(system_matrix);
    constraints.condense(system_rhs);

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);

    constraints.distribute(solution);

    std::cout << "   Interpolating..." << std::endl;
    PolyUtils::interpolate_to_fine_grid(agglo_handler,
                                        interpolated_solution,
                                        solution,
                                        true /*on_the_fly*/);
  }

  template <int dim>
  void
  Solver<dim>::zero_pressure_dof_constraint()
  {
    const FEValuesExtractors::Scalar pressure(dim);
    ComponentMask  pressure_mask = fe_collection.component_mask(pressure);
    const IndexSet pressure_dofs =
      DoFTools::extract_dofs(agglo_handler.agglo_dh, pressure_mask);
    const types::global_dof_index first_pressure_dof =
      pressure_dofs.nth_index_in_set(0);
    constraints.constrain_dof_to_zero(first_pressure_dof);
  }

  template <int dim>
  void
  Solver<dim>::mean_pressure_to_zero()
  {
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    Vector<double> integral_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(agglo_handler.output_dh,
                                      interpolated_solution,
                                      Functions::ZeroFunction<dim>(dim + 1),
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_mask);
    const double global_pressure_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);

    const double mean_pressure = global_pressure_integral / domain_volume;

    for (const auto &cell : agglo_handler.output_dh.active_cell_iterators())
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
          }
      }
  }

  template <int dim>
  void
  Solver<dim>::compute_errors(const Function<dim> &exact_solution,
                              double              &error_velocity_L2,
                              double              &error_velocity_H1,
                              double              &error_pressure_L2) const
  {
    AssertThrow(
      is_solved,
      ExcMessage("Cannot compute errors: the system has not been solved yet."));

    error_velocity_L2 = 0.;
    error_velocity_H1 = 0.;
    error_pressure_L2 = 0.;

    Vector<float> difference_per_cell(triangulation.n_active_cells());
    Vector<float> integral_per_cell(triangulation.n_active_cells());
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);

    VectorTools::integrate_difference(agglo_handler.output_dh,
                                      interpolated_solution,
                                      exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    error_velocity_L2 = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::L2_norm);

    VectorTools::integrate_difference(agglo_handler.output_dh,
                                      interpolated_solution,
                                      exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::H1_norm,
                                      &velocity_mask);
    error_velocity_H1 = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::H1_norm);

    VectorTools::integrate_difference(agglo_handler.output_dh,
                                      interpolated_solution,
                                      exact_solution,
                                      integral_per_cell,
                                      q_collection,
                                      VectorTools::mean,
                                      &pressure_mask);
    const double global_ep_integral =
      -VectorTools::compute_global_error(triangulation,
                                         integral_per_cell,
                                         VectorTools::mean);
    const double mean_ep = global_ep_integral / domain_volume;

    Vector<double> corrected_solution(interpolated_solution);
    corrected_solution.add(-mean_ep);
    VectorTools::integrate_difference(agglo_handler.output_dh,
                                      corrected_solution,
                                      exact_solution,
                                      difference_per_cell,
                                      q_collection,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    error_pressure_L2 = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::L2_norm);

    std::cout << "     velocity L2 error: " << error_velocity_L2 << std::endl
              << "     velocity H1 error: " << error_velocity_H1 << std::endl
              << "     pressure L2 error: " << error_pressure_L2 << std::endl;
  }

  template <int dim>
  void
  Solver<dim>::output_results() const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(agglo_handler.output_dh);

    data_out.add_data_vector(interpolated_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> agglo_idx(triangulation.n_active_cells());
    for (const auto &polytope : agglo_handler.polytope_iterators())
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

    std::string n_polytopes_str =
      Utilities::int_to_string(agglo_handler.n_agglomerates(),
                               /* digits = */ 5);

    std::string output_filename =
      output_prefix + "_solution_" + n_polytopes_str + ".vtk";
    std::ofstream output(output_filename);

    data_out.write_vtk(output);

    std::cout << "     Solution written to file: " << output_filename
              << std::endl;

    if constexpr (dim == 2)
      {
        // Export polygon boundaries to CSV file
        std::string polygon_boundaries =
          output_prefix + "_polygon_boundaries_" + n_polytopes_str;
        PolyUtils::export_polygon_to_csv_file(agglo_handler,
                                              polygon_boundaries);
      }
  }

  template <int dim>
  void
  Solver<dim>::run()
  {
    std::cout << "   Setting up system..." << std::endl;
    setup_agglo_system();

    std::cout << "   Assembling..." << std::endl;
    assemble_system();

    std::cout << "   Solving..." << std::endl;
    solve();

    std::cout << "   Modifying pressure..." << std::endl;
    mean_pressure_to_zero();

    std::cout << "   Writing output..." << std::endl;
    output_results();

    is_solved = true;
  }

  template class Solver<2>;
  template class Solver<3>;
} // namespace PolyOseenSolver