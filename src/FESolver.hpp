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
                 const std::string &output_directory_ = "./",
                 const std::string &output_filename_ = "output")
     : NDSolver<DIM>(problem_, deltat_, T_, r_, output_directory_, output_filename_)
   {}

protected:
  // Assemble the tangent problem.
  virtual void assemble_system() override;
};

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