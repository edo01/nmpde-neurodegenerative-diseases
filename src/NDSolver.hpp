#ifndef ND_SOLVER_HPP
#define ND_SOLVER_HPP 

#define ANYSOTROPIC true
#define SAVE_FIBER_FIELD_TO_FILE true

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

#include <fstream>
#include <iostream>

#include "NDProblem.hpp"
#include "AnisotropicEvaluator.hpp"

using namespace dealii;

template<unsigned int DIM>
class NDSolver
{
public:
  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  NDSolver(NDProblem<DIM> &problem_,
                const double deltat_,
                const double T_,
                const unsigned int &r_,
                const std::string &output_directory_ = "./",
                const std::string &output_filename_ = "output")
    :
      problem(problem_)
    , deltat(deltat_)
    , T(T_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , r(r_)
    , output_directory(output_directory_)
    , output_filename(output_filename_)
    , mesh(MPI_COMM_WORLD)
    , anisotropic_evaluator(problem_, mesh_serial, mesh, mpi_rank, mpi_size, pcout)
  {}

  virtual ~NDSolver() = default;

  // Initialization.
  virtual void setup();

  // Solve the problem.
  virtual void solve();

protected:
  // Assemble the tangent problem.
  virtual void assemble_system() = 0;

  // Solve the linear system associated to the tangent problem.
  virtual void solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  virtual void solve_newton();

  // Output.
  void output(const unsigned int &time_step) const;

  Tensor<2, DIM> evaluate_diffusion_coeff(const types::global_cell_index &global_active_cell_index, const Point<DIM> &p) const;

  // Problem definition.
  const NDProblem<DIM> &problem;
        
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

  private: 

  // Anisotropic evaluator
  AnisotropicEvaluator<DIM> anisotropic_evaluator;

  // Write the fiber field to the output file.
  void write_fiber_field_to_file() const;
};


template<unsigned int DIM>
Tensor<2, DIM> NDSolver<DIM>::evaluate_diffusion_coeff(const types::global_cell_index &global_cell_index, const Point<DIM> &p) const{
  const auto &diffusion_tensor = problem.get_diffusion_tensor();

  if constexpr (ANYSOTROPIC)
  {
    return anisotropic_evaluator.cells_domain[global_cell_index] == 0 ? 
        diffusion_tensor.white_matter_value(p) : diffusion_tensor.gray_matter_value();
  }
  else
  {
    return diffusion_tensor.white_matter_value(p);
  }
 
};

template<unsigned int DIM>
void
NDSolver<DIM>::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;


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


  // ANYSOTROPIC EVALUATION
  if constexpr (ANYSOTROPIC)
  {
    pcout << "-----------------------------------------------" << std::endl;
    pcout << "  Anysotropic evaluation" << std::endl;
    anisotropic_evaluator.compute_cells_domain();
  }

}

template<unsigned int DIM>
void
NDSolver<DIM>::solve_linear_system()
{
  SolverControl solver_control(20000, 1e-12 * residual_vector.l2_norm());

  //SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);;

    TrilinosWrappers::PreconditionSSOR      preconditioner;
    //TrilinosWrappers::PreconditionAMG preconditioner;
  preconditioner.initialize(
    jacobian_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta_owned, residual_vector, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}


template<unsigned int DIM>
void
NDSolver<DIM>::solve_newton()
{
  const unsigned int n_max_iters        = 1000;
  const double       residual_tolerance = 1e-9;

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
NDSolver<DIM>::write_fiber_field_to_file() const
{

    auto &fiber_field = problem.get_diffusion_tensor().get_fiber_field();
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
NDSolver<DIM>::output(const unsigned int &time_step) const
{
  DataOut<DIM> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  pcout << std::endl << "  Numerical range of solution u: \n" << std::endl;

  pcout << "  Min: " << solution.min() << std::endl;
  pcout << "  Max: " << solution.max() << std::endl;

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
NDSolver<DIM>::solve()
{
  pcout << "===============================================" << std::endl;

  if constexpr (SAVE_FIBER_FIELD_TO_FILE)
  {
    write_fiber_field_to_file();
  }

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

      // Store the old solution, so that it is available for assembly.
      solution_old = solution;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << std::fixed << time << std::endl;

      // At every time step, we invoke Newton's method to solve the non-linear
      // problem.
      solve_newton();

      output(time_step);

      pcout << std::endl;
    }

    pcout << "===============================================" << std::endl;
}

#endif // ND_SOLVER_HPP