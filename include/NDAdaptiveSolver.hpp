#ifndef NDADAPTIVESOLVER_HPP
#define NDADAPTIVESOLVER_HPP

#include "NDThetaSolver.hpp"

template <unsigned int DIM>
class NDAdaptiveSolver : public NDThetaSolver<DIM>
{
  public:
    NDAdaptiveSolver(NDProblem<DIM> &problem_,
                  double theta_,
                  double deltat_,
                  double T_,
                  unsigned int &r_,
                  double err_tol_ = 1e-9,
                  double err_perc = 0.1,
                  double min_step_ = 1e-6,
                  const std::string &output_directory_ = "./",
                  const std::string &output_filename_ = "output")
      : NDThetaSolver<DIM>(problem_, theta_, deltat_, T_, r_, output_directory_, output_filename_),
        err_tol(err_tol_),
        err_perc(err_perc),
        min_step(min_step_)
    {
      if(this->theta >= 0.5)
        throw std::runtime_error("NDAdaptiveSolver is only implemented for theta < 0.5.");
    }

  virtual ~NDAdaptiveSolver() = default;

  // Solve the problem.
  virtual void solve() override;

  private:
    // Tolerance for the error.
    const double err_tol;

    // Error percentage.
    const double err_perc;

    // Minimum time step.
    const double min_step;
};

template <unsigned int DIM>
void
NDAdaptiveSolver<DIM>::solve()
{

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

        // calc error
        double relative_error = error_norm/(3*solution_norm);
        double max_error = err_perc * err_tol;

        // ||u_{dt/2} - u_{dt}||/||u_{dt/2}|| < err_tol-max_error -> increase the time step
        if( relative_error < err_tol-max_error)
        {
          // keep the solution already calculated
          
          //increase the time step
          double temp = dt_half;
          dt_half *= 2;
          dt = temp;

          this->pcout << "[INFO] Increasing the time step. new deltat: " << dt_half 
              << ", err: "<< relative_error << std::endl;

        }
        else if (relative_error > err_tol+max_error)
        {
          // discard the solution
          this->solution = this->solution_old;

          // reduce the time step
          double temp = dt_half;
          dt_half /= 2;
          dt = temp;

          // reduce the time step
          this->pcout << "[INFO] Reducing the time step. new deltat: " << dt_half 
              << ", err: "<< relative_error << std::endl;

          // this->deltat is already set to dt_half 
        }else{
          this->pcout << "[INFO] Accepting the solution. err: "<< relative_error << std::endl;

          // Accept the solution
          // the solution with dt/2 is already stored in this->solution
          // Reset the time step
          this->deltat = dt;

          // output the solution
          this->output(time_step);

          this->pcout << std::endl;
          break;
        }
      }
    }
}

template <unsigned int DIM>
class NDForwardAdaptiveSolver : public NDAdaptiveSolver<DIM>
{
  public:
    NDForwardAdaptiveSolver(NDProblem<DIM> &problem_,
                            double deltat_,
                            double T_,
                            unsigned int &r_,
                            const double err_tol_ = 1e-9,
                            const double err_perc = 0.1,
                            const double min_step_ = 1e-6,
                            const std::string &output_directory_ = "./",
                            const std::string &output_filename_ = "output")
      : NDAdaptiveSolver<DIM>(problem_, 0.0, deltat_, T_, r_, err_tol_, err_perc, min_step_, output_directory_, output_filename_)
    {}

    virtual ~NDForwardAdaptiveSolver() = default;
};



#endif // NDADAPTIVESOLVER_HPP