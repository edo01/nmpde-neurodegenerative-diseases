#include "NDProblem.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "BESolver.hpp"
#include "NDConfig.hpp"

static NDConfig config_line = {
    .dim = 1,
    .T = 7.0,
    .alpha = 1.8,
    .deltat = 0.01,
    .degree = 1,
    .d_ext = 0.00,
    .d_axn = 0.2,
    .C_0 = 0.4,
    .mesh = "../meshes/mesh-40.msh",
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  NDConfig config = config_line;

  config.parse(argc, argv);

  ExponentialInitialCondition<1> initial_condition;
  RadialFiberField<1> fiber_field;

  NDProblem<1> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, initial_condition, fiber_field);
  problem.export_problem(std::string(config.output_dir) + config.output_filename + ".problem");

  BESolver<1> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);
  solver.setup();
  solver.solve();

  return EXIT_SUCCESS;
}