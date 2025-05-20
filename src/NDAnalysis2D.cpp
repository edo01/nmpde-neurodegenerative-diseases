#include "NDProblem.hpp"
#include "NDConfig.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "ThetaSolver.hpp"

static const Point<2> square_origin = Point<2>(0.5, 0.5);

static NDConfig config_square = {
    .dim = 2,
    .T = 7.0,
    .alpha = 1.8,
    // .alpha = 0.0,
    .deltat = 0.01,
    .degree = 1,
    .d_ext = 0.00,
    .d_axn = 0.2,
    .C_0 = 0.4,
    // .mesh = "../meshes/mesh-square-40.msh",
    .mesh = "../meshes/mesh-square-200.msh",
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  NDConfig config = config_square;

  config.parse(argc, argv);

  const Point<2> random_point(0.7, 0.7);
//   ConstantInitialCondition<2> initial_condition(1.0, random_point, 0.1);
  //ExponentialInitialCondition<2> initial_condition(random_point);
  QuadraticInitialCondition<2> initial_condition(random_point, 0.9, 0.1);
  AxonBasedFiberField<2> fiber_field(square_origin, Point<2>(0.25, 0.2));

  NDProblem<2> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, initial_condition, fiber_field);
  //BESolver<2> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);
  ThetaSolver<2> solver(problem, config.deltat, config.T, config.degree, 1.0, config.output_dir, config.output_filename);

  problem.export_problem(config.output_dir + config.output_filename + ".problem");
  solver.setup();
  solver.solve();

  return EXIT_SUCCESS;
}