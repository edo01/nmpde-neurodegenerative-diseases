#ifndef FE_SOLVER_HPP
#define FE_SOLVER_HPP 

#include "NDSolver.hpp"

template<unsigned int DIM>
class FESolver : public NDSolver<DIM>
{
public:

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
     FESolver(NDProblem<DIM> &problem_,
                 const double deltat_,
                 const double T_,
                 const unsigned int &r_,
                 const bool adaptive_ = false,
                 const double err_tol_ = 1e-9,
                 const double min_step_ = 1e-6,
                 const std::string &output_directory_ = "./",
                 const std::string &output_filename_ = "output")
     : NDSolver<DIM>(problem_, deltat_, T_, r_, output_directory_, output_filename_),
       adaptive(adaptive_),
       err_tol(err_tol_),
       min_step(min_step_)
   {}

   virtual ~FESolver() = default;

  // Solve the problem.
  virtual void solve() override;

protected:
  // Assemble the tangent problem.
  virtual void assemble_system() override;

private:
  // Flag to indicate if we are using adaptive time stepping.
  const bool adaptive;

  // Tolerance for the error.
  const double err_tol;

  // Minimum time step.
  const double min_step;

};

/**
 * Solve the problem using the Forward Euler method or the adaptive Forward Euler method.
 * 
 * Scheme for the adaptive Forward Euler method:
 * 
 * 1. Apply the initial condition.
 * 2. Compute the solution at time t+dt.
 * 3. Compute the solution at time t+0.5*dt.
 * 4. Compute the error ||u_{dt/2} - u_{dt}||/||u_{dt/2}||.
 * 5. If the error is less than 3*err_tol, accept the solution and move to the next time step.
 * 6. If the error is greater than 3*err_tol, reduce the time step by half and repeat the process.
*/
template<unsigned int DIM>
void FESolver<DIM>::solve()
{
  // If the adaptive flag is not set, we use the standard time stepping.
  if(!this->adaptive){
    NDSolver<DIM>::solve();
    return;
  }

  this->pcout << "===============================================" << std::endl;
  this->pcout << "\t\tADAPTIVE FORWARD EULER" << std::endl;

  this->time = 0.0;

  // Apply the initial condition.
  {
    this->pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(this->dof_handler, this->problem.get_initial_concentration(),
                             this->solution_owned);
    this->solution = this->solution_owned;

    // Output the initial solution.
    this->output(0);
    //pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  double dt      = this->deltat;
  double dt_half = this->deltat / 2.0;

  // Store the dt_half solution.
  TrilinosWrappers::MPI::Vector solution_dt; // store the solution at time t+dt


  /*
  At each timestep we compute the solution at time t+dt and t+0.5*dt.

  */
  while (this->time < this->T - 0.5 * this->deltat)
    {
      this->time += dt_half;
      ++time_step;

      // Store the old solution, so that it is available for assembly.
      this->solution_old = this->solution;

      this->pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << std::fixed << this->time << std::endl;

      // Check if the time step is less than the minimum time step.
      while(dt_half>2*this->min_step)
      {

        this->pcout << "Computing the solution at time t+dt" << std::endl;
        // Compute the solution at time t+dt
        this->solve_newton();
        // Save the solution at time t+dt
        solution_dt = this->solution;

        this->pcout << "Computing the solution at time t+0.5*dt" << std::endl;
        // Set the time step to dt/2
        this->deltat = dt_half;
        // Compute the solution at time t+0.5*dt
        this->solve_newton();


        /*
        At this point we have:
        - solution: u_{dt}
        - solution_dt: u_{dt+dt}
        */

        // Compute the error
        // u_{dt/2} - u_{dt}
        TrilinosWrappers::MPI::Vector error;
        error = this->solution;
        error -= solution_dt;


        // ||u_{dt/2} - u_{dt}||
        double error_norm = error.l2_norm();
        // ||u_{dt/2}||
        double solution_norm = this->solution.l2_norm();

        // ||u_{dt/2} - u_{dt}||/||u_{dt/2}|| < 3*err_tol
        if((error_norm/solution_norm) < 3*this->err_tol)
        {
          this->pcout << "[INFO] Accepting the solution. err: "<< error_norm/solution_norm << std::endl;

          // Accept the solution
          // the solution with dt/2 is already stored in this->solution
          // Reset the time step
          this->deltat = dt;

          // output the solution
          this->output(time_step);

          this->pcout << std::endl;
          break;
        }
        else
        {
          // discard the solution
          this->solution = this->solution_old;

          // reduce the time step
          double temp = dt_half;
          dt_half /= 2;
          dt = temp;

          // reduce the time step
          this->pcout << "[INFO] Reducing the time step. new deltat: " << dt_half 
              << ", err: "<< error_norm/solution_norm << std::endl;

          // this->deltat is already set to dt_half          
        }
      }
    }
}

template<unsigned int DIM>
void
FESolver<DIM>::assemble_system()
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
  
  // Value of the solution at previous timestep (un) on current cell.
  std::vector<double> solution_old_loc(n_q);
  std::vector<Tensor<1, DIM>> solution_old_gradient_loc(n_q);

  // get the parameters of the problem once for all
  const double alpha = this->problem.get_alpha();
  const auto diffusion_tensor = this->problem.get_diffusion_tensor();
  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      fe_values.reinit(cell);
      cell_matrix   = 0.0;
      cell_residual = 0.0;
      fe_values.get_function_values(this->solution, solution_loc);
      fe_values.get_function_values(this->solution_old, solution_old_loc);
      fe_values.get_function_gradients(this->solution_old, solution_old_gradient_loc);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          //evaluate the Diffusion term on the current quadrature point
          const Tensor<2, DIM> diffusion_coefficent_loc =
            diffusion_tensor.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass matrix. 
                  // phi_i * phi_j/deltat * dx
                  cell_matrix(i, j) += fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) / this->deltat *
                                       fe_values.JxW(q);
                    
                }
              // Assemble the residual vector (with changed sign).
              // Time derivative term.
              // phi_i * (c - c_old)/deltat * dx
              cell_residual(i) -= (solution_loc[q] - solution_old_loc[q]) /
                                  this->deltat * fe_values.shape_value(i, q) *
                                  fe_values.JxW(q);
              // Diffusion term.
              // D*grad(c_old) * grad(phi_i) * dx
              cell_residual(i) -= (diffusion_coefficent_loc *
                  solution_old_gradient_loc[q]) * fe_values.shape_grad(i, q) * fe_values.JxW(q);

              // Reaction term. (Non-linear)
              // alpha * c_old * (1-c_old) * phi_i * dx
              cell_residual(i) +=
                alpha * solution_old_loc[q] * (1-solution_old_loc[q]) * fe_values.shape_value(i, q) *
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

#endif // FE_SOLVER_HPP