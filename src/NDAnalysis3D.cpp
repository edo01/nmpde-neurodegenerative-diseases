#include "NDConfig.hpp"
#include "NDProblem.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "NDThetaSolver.hpp"
#include "SeedingRegions.hpp"

static const Point<3> brain_origin = Point<3>(48.0, 73.0, 60.0);
static const Point<3> cube_origin = Point<3>(0.5, 0.5, 0.5);

static NDConfig config_cube = {
    .dim = 3,
    .T = 20.0,
    .alpha = 0.45,
    .deltat = 0.2,
    .degree = 1,
    .d_ext = 0.0,
    .d_axn = 20.0,
    .C_0 = 0.4,
    .mesh = "../meshes/mesh-cube-40.msh",
};

static NDConfig config_brain = {
    .dim = 3,
    .T = 24.0,
    .alpha = 0.5,
    .deltat = 0.24,
    .degree = 1,
    .d_ext = 1.5,
    .d_axn = 3.0,
    .C_0 = 0.95,
    .mesh = "../meshes/brain-h3.03D.msh",
    .seeding_region_type = SeedingRegionType::AmyloidBeta,
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  //choose the configuration
  NDConfig config = config_brain;
  //NDConfig config = config_cube;

  config.parse(argc, argv);

  //brain mesh
  SeedingRegion sr = SeedingRegion::create(config.seeding_region_type, config.C_0);
  AxonBasedFiberField<3> fiber_field(20, brain_origin);
  NDProblem<3> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, sr, fiber_field);

  // cube mesh 
  // AxonBasedFiberField<3> fiber_field_cube(0.3, cube_origin);
  // const Point<3> random_point(0.7, 0.7, 0.7);
  // ExponentialInitialCondition<3> initial_condition_cube(random_point, 0.1, 1.0, 0.1); 
  // NDProblem<3> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, initial_condition_cube, fiber_field_cube);
  NDBackwardEulerSolver<3> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);

  problem.export_problem(std::string(config.output_dir) + config.output_filename + ".problem");
  solver.setup();
  solver.solve();

  return EXIT_SUCCESS;
}