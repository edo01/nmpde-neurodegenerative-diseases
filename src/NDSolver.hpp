#ifndef ND_SOLVER_HPP
#define ND_SOLVER_HPP 

#define ANYSOTROPIC true 

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
#include <vector>

#include "NDProblem.hpp"

using namespace dealii;

template<unsigned int DIM>
class NDSolver
{
public:


  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  NDSolver(NDProblem<DIM> &problem_,
                const unsigned int &r_,
                const std::string &output_directory_ = "./",
                const std::string &output_filename_ = "output")
    :
      problem(problem_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , r(r_)
    , output_directory(output_directory_)
    , output_filename(output_filename_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void setup();

  // Solve the problem.
  void solve();

protected:
  // Assemble the tangent problem.
  void assemble_system();

  // Solve the linear system associated to the tangent problem.
  void solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void solve_newton();

  // Output.
  void output(const unsigned int &time_step) const;

  Tensor<2, DIM> evaluate_diffusion_coeff(const unsigned int &quadrature_point_index, const Point<DIM> &p) const;

  void compute_quadrature_points_domain();

  // Problem definition.
  NDProblem<DIM> problem;
        
  // Current time.
  double time;

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

  // Coordinates of the mesh bounding box center
  Point<DIM> box_center;

  // Vector which maps every quadrature point to a part of the brain denoted by a value:
  // 0 if it is in the white portion or 1 for the gray portion. @TODO: could be boolean but some problems with file storing and loading
  std::vector<int> quadrature_points_domain;
};

void printLoadingBar(int current, int total, int barLength = 50) {
    float progress = (float)current / total;
    int pos = barLength * progress;

    std::cout << "[";
    for (int i = 0; i < barLength; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

template<typename T>
void saveVectorToFile(std::vector<T>& vec, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
    outfile.close();
}

// Fix messages print in parallel
template<typename T>
bool loadVectorFromFile(std::vector<T>& vec, const std::string& filename) {
    std::cout<< "Trying to read quadrature points domain file '" << filename << "'" <<  std::endl;
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile) {
        std::cout<< "Failed to open file\n";
        return false;
    }

    infile.seekg(0, std::ios::end);
    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    vec.resize(size / sizeof(T));
    infile.read(reinterpret_cast<char*>(vec.data()), size);

    // Check if the read operation was successful
    if (infile.fail()) {
        std::cout<< "Read operation failed\n";
        return false;
    }

    infile.close();
    return true;
}

/**
 * Compute the position of every quadrature point with respect
 * to the white and gray partion of the brain, and save the resulting
 * boolean vector.
 *  
*/
template<unsigned int DIM>
void NDSolver<DIM>::compute_quadrature_points_domain(){

  // @TODO: Assign custom name file based on mesh file name 
  const std::string file_name = "quadrature_points_domain_vec";

  // Tries to load existing file
  if(loadVectorFromFile(quadrature_points_domain, file_name)){
    pcout << "Found valid file.\n";
    return;
  }
  pcout << "It could be take a while.\n";

  // For each cell, there will be a number of quadrature points specified by the quadrature
  unsigned n_points = mesh_serial.n_global_active_cells()*quadrature->size();

  // Initialize vector to map every quadrature point to the brain portion
  // N.B. Could be <bool> since there are only two domains, but writing to file
  // is a little tricker with boolean vectors.
  quadrature_points_domain = std::vector<int>(n_points, 0);


  DoFHandler<DIM> dof_handler_serial;
  dof_handler_serial.reinit(mesh_serial);
  dof_handler_serial.distribute_dofs(*fe);

  // @TODO: Parallelize?
  if(mpi_rank == 0){
  // Retrieve all vertices on the boundary 
  std::map<unsigned int, Point<DIM>> boundary_vertices = GridTools::get_all_vertices_at_boundary(mesh_serial);

  // Retrieve all vertices from the triangulation 
  std::vector<Point<DIM>> triangulation_vertices=mesh_serial.get_vertices();
  
  // Create a vector to mark every triangulation vertix if they are on boundary
  std::vector<bool> triangulation_boundary_mask = std::vector<bool>(triangulation_vertices.size(), false);
  for(const auto [key, value] : boundary_vertices){
    triangulation_boundary_mask[key] = true;
  }

  // Only need to retrieve quadrature points from cells
  FEValues<DIM> fe_values(*fe, *quadrature,update_quadrature_points);

  double white_coeff = problem.get_white_matter_portion();

    // Current global quadrature point id.
  unsigned quadrature_point_id = 0;


  // Iterate through the cells  
  for (const auto &cell : dof_handler_serial.active_cell_iterators()){

    // NO PARALLEL YET
    /*
    if (!cell->is_locally_owned()){
      continue;
    }
    */

    // Set focus on current cell for the FEValues object 
    fe_values.reinit(cell);

    // For each quadrature point in cell
    for (const Point<DIM> &cell_quadrature_point : fe_values.get_quadrature_points())
    {
      // VERY COMPUTING INTENSIVE 
      // Find the closest vertex point on the boundary 
      unsigned closest_boundary_id = GridTools::find_closest_vertex(mesh_serial, cell_quadrature_point, triangulation_boundary_mask);
      Point<DIM> closest_boundary_point = triangulation_vertices[closest_boundary_id];
      // -------------------------

      // If the quadrature point is nearest to the center of the mesh box than its closest vertex on the white boundary,
      // it is in the white portion. 
      if(cell_quadrature_point.distance(box_center) < white_coeff*closest_boundary_point.distance(box_center))
        quadrature_points_domain[quadrature_point_id]=0;
      else
        quadrature_points_domain[quadrature_point_id]=1;


      printLoadingBar(quadrature_point_id, n_points);
      quadrature_point_id ++;
    }

  }

  // Save to file computed vector
  saveVectorToFile(quadrature_points_domain, file_name);

  }

  MPI_Bcast(quadrature_points_domain.data(), quadrature_points_domain.size(), MPI_INT, 0, MPI_COMM_WORLD);
}

template<unsigned int DIM>
Tensor<2, DIM> NDSolver<DIM>::evaluate_diffusion_coeff(const unsigned int &quadrature_point_index, const Point<DIM> &p) const{
  auto gray_diffusion_tensor = problem.get_gray_diffusion_tensor();
  auto white_diffusion_tensor = problem.get_white_diffusion_tensor();

#if ANYSOTROPIC
  return quadrature_points_domain[quadrature_point_index] == 0 ? 
    white_diffusion_tensor.value(p) :
    gray_diffusion_tensor.value(p);
#else
    return gray_diffusion_tensor.value(p);
#endif
     
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
   
    // MESH INFORMATIONS
    {

      pcout << "-----------------------------------------------" << std::endl;

      pcout << "  Mesh file informations:" << std::endl<<std::endl;
      pcout << "  Bounding box sides lenght:" << std::endl;

      auto box = GridTools::compute_bounding_box(mesh_serial);

      box_center = box.center();
      
      static const char labels[3] = {'x', 'y', 'z'}; 
      for(unsigned i=0; i<DIM; i++){
        pcout << "  " << labels[i] << ": " << box.side_length(i) << std::endl;
      }

      pcout << "  Center:  " << box_center << std::endl ;
      pcout << "  Box volume:  " << box.volume()<< std::endl;


      pcout << "  Number of elements = " << mesh.n_global_active_cells()
            << std::endl;
      
    }

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
#if ANYSOTROPIC
  {
    pcout << std::endl << "Evaluating quadrature points domain." << std::endl;
    compute_quadrature_points_domain();
  }
#endif
}

template<unsigned int DIM>
void
NDSolver<DIM>::assemble_system()
{


  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<DIM> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_residual(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  residual_vector = 0.0;

  // Value and gradient of the solution on current cell.
  std::vector<double>         solution_loc(n_q);
  std::vector<Tensor<1, DIM>> solution_gradient_loc(n_q);

  // Value of the solution at previous timestep (un) on current cell.
  std::vector<double> solution_old_loc(n_q);

  // get the parameters of the problem once for all
  const double alpha = problem.get_alpha();
  const double deltat = problem.get_deltat();
  //const auto diffusion_tensor = problem.get_diffusion_tensor();

  int quadrature_point_id=0; 
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix   = 0.0;
      cell_residual = 0.0;

      fe_values.get_function_values(solution, solution_loc);
      fe_values.get_function_gradients(solution, solution_gradient_loc);
      fe_values.get_function_values(solution_old, solution_old_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {

          // Evaluate the Diffusion term on the current quadrature point.
          // Pass also the global quadrature point index to check in which part of the brain it is located. 
          const Tensor<2, DIM> diffusion_coefficent_loc = evaluate_diffusion_coeff(quadrature_point_id, fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass matrix. 
                  // phi_i * phi_j/deltat * dx
                  cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) / deltat *
                                       fe_values.JxW(q);

                  // Non-linear stiffness matrix, first term.
                  // D*grad(phi_i) * grad(phi_j) * dx
                  cell_matrix(i, j) += (diffusion_coefficent_loc 
                                   * fe_values.shape_grad(j, q)) *
                    fe_values.shape_grad(i, q) * fe_values.JxW(q);

                  // Non-linear stiffness matrix, second term.
                  // alpha * (1-2*c) * phi_i * phi_j * dx
                  cell_matrix(i, j) -=
                    alpha * (1-2.0*solution_loc[q]) * fe_values.shape_value(j, q) *
                    fe_values.shape_value(i, q) * fe_values.JxW(q);
                    
                }

              // Assemble the residual vector (with changed sign).

              // Time derivative term.
              // phi_i * (c - c_old)/deltat * dx
              cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                                  deltat * fe_values.shape_value(i, q) *
                                  fe_values.JxW(q);

              // Diffusion term.
              // D*grad(c) * grad(phi_i) * dx
              cell_residual(i) -= (diffusion_coefficent_loc *
                  solution_gradient_loc[q]) * fe_values.shape_grad(i, q) * fe_values.JxW(q);

              // Reaction term. (Non-linear)
              // alpha * c * (1-c) * phi_i * dx
              cell_residual(i) +=
                alpha * solution_loc[q] * (1-solution_loc[q]) * fe_values.shape_value(i, q) *
                fe_values.JxW(q);

            }

            quadrature_point_id ++;
        }

      cell->get_dof_indices(dof_indices);

      jacobian_matrix.add(dof_indices, cell_matrix);
      residual_vector.add(dof_indices, cell_residual);

    }

  jacobian_matrix.compress(VectorOperation::add);
  residual_vector.compress(VectorOperation::add);
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
NDSolver<DIM>::output(const unsigned int &time_step) const
{
  DataOut<DIM> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  pcout << std::endl << "  Numerical range of solution u: \n" << std::endl;

  pcout << "  Min: " << solution_owned.min() << std::endl;
  pcout << "  Max: " << solution_owned.max() << std::endl;
  //pcout << "  L2: " << solution_owned.l2_norm() << std::endl;

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

  const double T = problem.get_T();
  const double deltat = problem.get_deltat();

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
}


#endif
