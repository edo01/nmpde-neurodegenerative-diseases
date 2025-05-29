#ifndef NDTHETASOLVER_HPP
#define NDTHETASOLVER_HPP

#include <fstream>
#include <iostream>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_out.h>

#include "NDProblem.hpp"
#include "WhiteGrayPartition.hpp"


using namespace dealii;

template<unsigned int DIM>
class NDThetaSolver
{
public:
  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
    NDThetaSolver(NDProblem<DIM> &problem_,
                  double theta_,
                  double deltat_,
                  double T_,
                  unsigned int &r_,
                  const std::string &output_directory_ = "./",
                  const std::string &output_filename_ = "output")
    :
      problem(problem_)
    , diffusion_tensor(problem_.get_diffusion_tensor())
    , theta(theta_)
    , deltat(deltat_)
    , T(T_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , r(r_)
    , output_directory(output_directory_)
    , output_filename(output_filename_)
    , mesh(MPI_COMM_WORLD)
  {
    if(theta < 0.0 || theta > 1.0)
      throw std::runtime_error("Theta parameter must be in the interval [0, 1].");
  }

  virtual ~NDThetaSolver() = default;

  // Initialization.
  virtual void setup();

  // Solve the problem.
  virtual void solve();

protected:
  // Assemble the tangent problem.
  virtual void assemble_system();

  // Solve the linear system associated to the tangent problem.
  virtual void solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  virtual void solve_newton();

  // Output.
  void output(const unsigned int &time_step) const;

  // Problem references
  const NDProblem<DIM> &problem;
  const typename NDProblem<DIM>::DiffusionTensor &diffusion_tensor;

  // Theta method parameter.
  const double theta;

  // Current time and time step.
  double time;
  double deltat;

  // Final time.
  double T;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Polynomial degree.
  const unsigned int r;

  // directory where the output files will be written
  std::string output_directory;

  // output filename
  std::string output_filename;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Mesh.
  parallel::fullydistributed::Triangulation<DIM> mesh;
  Triangulation<DIM> mesh_serial;

  // Finite element space.
  std::unique_ptr<FiniteElement<DIM>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<DIM>> quadrature;

  // DoF handler.
  DoFHandler<DIM> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;

  // Global concentration.
  double global_concentration;

  // Total domain volume.
  double total_domain_volume;

  private: 

  // Write the fiber field to the output file.
  void write_fiber_field_to_file() const;

  // Integrate the current solution on whole domain to compute the global concentration.
  void compute_current_global_concentration();

};

template<unsigned int DIM>
void
NDThetaSolver<DIM>::assemble_system()
{

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q           = this->quadrature->size();

  FEValues<DIM> fe_values(*(this->fe),
                          *(this->quadrature),
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  this->jacobian_matrix = 0.0;
  this->residual_vector = 0.0;

  // Value and gradient of the solution on current cell.
  std::vector<double>         solution_loc(n_q);
  std::vector<double> solution_old_loc(n_q);

  std::vector<Tensor<1, DIM>> solution_gradient_loc(n_q);
  std::vector<Tensor<1, DIM>> solution_old_gradient_loc(n_q);

  std::vector<Tensor<2, DIM>> diffusion_coefficent_loc(n_q);

  // get the alpha value from the problem
  const double alpha_white_matter = this->problem.get_alpha();
  const double alpha_gray_matter = this->problem.get_alpha() / 2.0;

  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      //query material id of the current cell and set variable parameters
      double alpha;
      if(cell->material_id() != 1) // we are on a white matter cell
      {
        for(unsigned int q = 0; q < n_q; ++q) diffusion_coefficent_loc[q] = diffusion_tensor.white_matter_value(fe_values.quadrature_point(q));
        alpha = alpha_white_matter;
      }
      else // we are on a gray cell
      {
        for(unsigned int q = 0; q < n_q; ++q) diffusion_coefficent_loc[q] = diffusion_tensor.gray_matter_value();
        alpha = alpha_gray_matter;
      }

      cell_matrix   = 0.0;
      cell_residual = 0.0;

      fe_values.get_function_values(this->solution, solution_loc);
      fe_values.get_function_values(this->solution_old, solution_old_loc);

      fe_values.get_function_gradients(this->solution, solution_gradient_loc);
      fe_values.get_function_gradients(this->solution_old, solution_old_gradient_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          double theta_comb = (1 - theta) * solution_old_loc[q] + theta * solution_loc[q];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass matrix. 
                  // phi_i * phi_j/deltat * dx
                  cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) / this->deltat *
                                       fe_values.JxW(q);

                  // Non-linear stiffness matrix, first term.
                  // D*grad(phi_i) * grad(phi_j) * dx
                  cell_matrix(i, j) += theta * diffusion_coefficent_loc[q] 
                                   * fe_values.shape_grad(j, q) *
                    fe_values.shape_grad(i, q) * fe_values.JxW(q);

                  // Non-linear stiffness matrix, second term.
                  // alpha * (1-2*c) * phi_i * phi_j * dx

                  cell_matrix(i, j) -=
                    theta * alpha * (1-2.0 * theta_comb) * fe_values.shape_value(j, q) *
                    fe_values.shape_value(i, q) * fe_values.JxW(q);
                    
                }

              // Assemble the residual vector (with changed sign).

              // Time derivative term.
              // phi_i * (c - c_old)/deltat * dx
              cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                                  this->deltat * fe_values.shape_value(i, q) *
                                  fe_values.JxW(q);

              // Diffusion term.
              //(1-theta) * D*grad(c_old) * grad(phi_i) * dx
              cell_residual(i) -= (1-theta) * diffusion_coefficent_loc[q] *
                  solution_old_gradient_loc[q] * fe_values.shape_grad(i, q) * fe_values.JxW(q);

                  // Diffusion term.
              //(1-theta) * D*grad(c_old) * grad(phi_i) * dx
              cell_residual(i) -= theta * diffusion_coefficent_loc[q] *
                  solution_gradient_loc[q] * fe_values.shape_grad(i, q) * fe_values.JxW(q);

              // Reaction term. (Non-linear)
              // alpha * (theta_comb) * (1-theta_comb) * phi_i * dx
              cell_residual(i) -=
                alpha * theta_comb * (theta_comb-1) * fe_values.shape_value(i, q) *
                fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      this->jacobian_matrix.add(dof_indices, cell_matrix);
      this->residual_vector.add(dof_indices, cell_residual);
    }

  this->jacobian_matrix.compress(VectorOperation::add);
  this->residual_vector.compress(VectorOperation::add);
}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;
    pcout << "Mesh file name = " << problem.get_mesh_file_name() << std::endl;

    GridIn<DIM> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(problem.get_mesh_file_name());
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "-----------------------------------------------" << std::endl;
  }

    //print mesh info and partition into white and gray matter
  {

    pcout << "Mesh information" << std::endl;
    pcout << "  Number of global cells = " << mesh.n_global_active_cells() << std::endl;
    pcout << "  Number of cells " << mesh.n_active_cells() << std::endl;
    pcout << "  Number of locally owned cells = " << mesh.n_locally_owned_active_cells() << std::endl;
    pcout << "  Number of vertices " << mesh.n_vertices() << std::endl;
    
    const double distance_threshold = problem.get_gray_matter_distance_threshold();
    if(distance_threshold > 0.0)
    {
        pcout << "Partitioning the mesh into white and gray matter" << std::endl;
        WhiteGrayPartition::set_white_gray_material(mesh_serial, mesh, distance_threshold);     

        pcout << "Writing the partition to file" << std::endl;
        WhiteGrayPartition::write_partition_to_pvtu(mesh, output_directory, output_filename);
    }
    else
    {
        pcout << "Partitioning into white and gray matter is disabled. All cells are white matter." << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

  }

  // Calculate and store the total domain volume (needed for global concentration calculation)
  {
    double local_volume = 0.0;
    for (const auto &cell : this->mesh.active_cell_iterators()) 
    {
        if (cell->is_locally_owned())
            local_volume += cell->measure();
    }
    this->total_domain_volume = Utilities::MPI::sum(local_volume, MPI_COMM_WORLD);
    pcout << "Total domain volume = " << this->total_domain_volume << std::endl;
    pcout << "-----------------------------------------------" << std::endl;
  }

  // FINITE ELEMENTS SPACE INITIALIZATION 
  {
    pcout << "Initializing the finite element space" << std::endl;

    if(DIM == 1)
      // Finite elements in one dimensions are obtained with the FE_Q class. 
      fe = std::make_unique<FE_Q<DIM>>(r);
    else
      // Triangular finite elements in higher dimensions are obtained w/
      // FE_SimplexP, while FE_Q would provide hexahedral elements. 
      fe = std::make_unique<FE_SimplexP<DIM>>(r);



    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<DIM>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // DOF HANDLER INITIALIZATION 
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // LINEAR SYSTEM INITIALIZATION 
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    jacobian_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    residual_vector.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution;
  }


}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::solve_linear_system()
{
  SolverControl solver_control(20000, 1e-12 * residual_vector.l2_norm());

  //SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);;

//   TrilinosWrappers::PreconditionSSOR      preconditioner;
//   preconditioner.initialize(jacobian_matrix,
//                             TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  TrilinosWrappers::PreconditionILU      preconditioner;
    preconditioner.initialize(jacobian_matrix,
                                 TrilinosWrappers::PreconditionILU::AdditionalData(1.0));
                 
//   TrilinosWrappers::PreconditionAMG preconditioner;
//   preconditioner.initialize(jacobian_matrix);

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;
}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::solve_newton()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-12;

  unsigned int n_iter        = 0;
  double       residual_norm = residual_tolerance + 1;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance)
    {

      assemble_system();
      residual_norm = residual_vector.l2_norm();

      pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
            << " - ||r|| = " << std::scientific << std::setprecision(6)
            << residual_norm << std::flush;

      // We actually solve the system only if the residual is larger than the
      // tolerance.
      if (residual_norm > residual_tolerance)
        {
          //At each iteration of Newton's method, we solve the linear system for the increment of the solution.
          solve_linear_system();

          solution_owned += delta_owned;
          solution = solution_owned;
        }
      else
        {
          pcout << " < tolerance" << std::endl;
        }

      ++n_iter;
    }
}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::write_fiber_field_to_file() const
{

    auto &fiber_field = diffusion_tensor.get_fiber_field();
    std::array<Vector<double>, DIM> fiber_field_values;

    for (unsigned int i = 0; i < DIM; ++i)
    {
        fiber_field_values[i].reinit(mesh.n_active_cells());
    }

    for (const auto &cell : mesh.active_cell_iterators())
    {
        if(!cell->is_locally_owned())
          continue;

        const unsigned int cell_idx = cell->active_cell_index();
        const auto p = cell->center();

        Vector<double> fiber(DIM);
        fiber_field.vector_value(p, fiber);

        for (unsigned int i = 0; i < DIM; ++i)
        {
            fiber_field_values[i][cell_idx] = fiber[i];
        }

    }

    DataOut<DIM> data_out;

    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> fiber_field_names = {"fiber_field_x", "fiber_field_y", "fiber_field_z"};

    for (unsigned int i = 0; i < DIM; ++i)
    {
        data_out.add_data_vector(fiber_field_values[i], fiber_field_names[i]);
    }

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
        output_directory, output_filename + "_fiber_field", 0, MPI_COMM_WORLD, 0);
}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::output(const unsigned int &time_step) const
{
  DataOut<DIM> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  pcout << std::endl << "  Numerical range of solution u: \n" << std::endl;

  pcout << "  Min: " << solution.min() << std::endl;
  pcout << "  Max: " << solution.max() << std::endl;
  pcout << "  Global concentration: " << this->global_concentration << std::endl;

  //pcout << "..............................................." << std::endl;
  pcout << std::endl << "<+><+><+><+><+><+><+><+><+><+><+><+><+><+><+><+><+><+><+><+>" << std::endl;
   
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    output_directory, output_filename, time_step, MPI_COMM_WORLD, 3);
}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::compute_current_global_concentration()
{
    double local_integral = 0.0;
    FEValues<DIM> fe_values_for_integration(*(this->fe),
                                            *(this->quadrature),
                                            update_values | update_quadrature_points | update_JxW_values);
    std::vector<double> u_values_at_quadrature_points(this->quadrature->size());

    for (const auto &cell : this->dof_handler.active_cell_iterators())
        {
        if (cell->is_locally_owned())
            {
            fe_values_for_integration.reinit(cell);
            fe_values_for_integration.get_function_values(this->solution, u_values_at_quadrature_points);
            for (unsigned int q_index = 0; q_index < this->quadrature->size(); ++q_index)
                {
                local_integral += u_values_at_quadrature_points[q_index] * fe_values_for_integration.JxW(q_index);
                }
            }
        }
    this->global_concentration = Utilities::MPI::sum(local_integral, MPI_COMM_WORLD) / this->total_domain_volume;
}

template<unsigned int DIM>
void
NDThetaSolver<DIM>::solve()
{
  pcout << "===============================================" << std::endl;


  write_fiber_field_to_file();

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, problem.get_initial_concentration(),
                             solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0);
    //pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      // Store the old solution (previous time-step), so that it is available for assembly.
      solution_old = solution;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << std::fixed << time << std::endl;

      // At every time step, we invoke Newton's method to solve the non-linear
      // problem.
      solve_newton();

      // Calculate global concentration by integrating the current solution over the whole domain
      compute_current_global_concentration();

      output(time_step);

      pcout << std::endl;
    }

    pcout << "===============================================" << std::endl;
}

// Cranck-Nicolson solver
template<unsigned int DIM>
class NDCrankNicolsonSolver : public NDThetaSolver<DIM>
{
  public:
    NDCrankNicolsonSolver(NDProblem<DIM> &problem_,
                          double deltat_,
                          double T_,
                          unsigned int &r_,
                          const std::string &output_directory_ = "./",
                          const std::string &output_filename_ = "output")
      : NDThetaSolver<DIM>(problem_, 0.5, deltat_, T_, r_, output_directory_, output_filename_)
    {}

    virtual ~NDCrankNicolsonSolver() = default;
};

// Backward Euler solver
template<unsigned int DIM>
class NDBackwardEulerSolver : public NDThetaSolver<DIM>
{
  public:
    NDBackwardEulerSolver(NDProblem<DIM> &problem_,
                          double deltat_,
                          double T_,
                          unsigned int &r_,
                          const std::string &output_directory_ = "./",
                          const std::string &output_filename_ = "output")
      : NDThetaSolver<DIM>(problem_, 1.0, deltat_, T_, r_, output_directory_, output_filename_)
    {}

    virtual ~NDBackwardEulerSolver() = default;
};


#endif // NDTHETASOLVER_HPP

/*
Question: I am doing a Newton iteration on the theta method fully discretized PDE (non-linear) parabolic PDE. But Newton iteration converges only if the initial guess (which is the initial condition) is "close enough" to the solution at the successive timestep. So it is clear that this scheme cannot work for any arbitrary big time-step, even for theta >= 0.5 (when it would be unconditionaly stable for linear parabolic PDEs). I can think of large enough time-steps for which the solution at next time-step is very different from the solution at the previous time-step, too far from the initial condition that Newton iteration cannot converge. So time-step should be small enough. Apparently, on this eqution, Backward Euler (theta = 1) seems to work good even for large time-steps. 
*/