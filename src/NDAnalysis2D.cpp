#include "NDProblem.hpp"
#include "NDConfig.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "NDThetaSolver.hpp"
#include "NDAdaptiveSolver.hpp"

static const Point<2> square_origin = Point<2>(0.5, 0.5);

static NDConfig config_square = {
    .dim = 2,
    .T = 7.0,
    .alpha = 1.8,
    .deltat = 0.01,
    .degree = 1,
    .d_ext = 0.00,
    .d_axn = 0.2,
    .C_0 = 0.95,
    .mesh = "../meshes/mesh-square-40.msh",
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  NDConfig config = config_square;

  config.parse(argc, argv);

  const Point<2> random_point(0.7, 0.7);
  ConstantInitialCondition<2> initial_condition(config.C_0, random_point, 0.1);
  //AxonBasedFiberField<2> fiber_field(0.3, square_origin);
  //RadialFiberField<2> fiber_field(square_origin);
  CircumferentialFiberField<2> fiber_field(square_origin);

  NDProblem<2> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, initial_condition, fiber_field);
  NDBackwardEulerSolver<2> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);
  //NDCrankNicolsonSolver<2> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);

  problem.export_problem(config.output_dir + config.output_filename + ".problem");
  solver.setup();
  solver.solve();

  return EXIT_SUCCESS;
}