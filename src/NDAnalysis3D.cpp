#include "NDConfig.hpp"
#include "NDProblem.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "NDThetaSolver.hpp"
#include "SeedingRegions.hpp"

static const Point<3> brain_origin = Point<3>(80.0, 78.0, 70.0);
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
    .fiber_field_type = FiberFieldType::AxonBased
};

static NDConfig config_brain_baseline = {
    .dim = 3,
    .T = 48.0,
    .alpha = 0.6,
    .deltat = 0.24,
    .degree = 1,
    .d_ext = 1.5,
    .d_axn = 3.0,
    .C_0 = 0.95,
    .mesh = "../meshes/brain-h3.03D.msh",
    .seeding_region_type = SeedingRegionType::Tau,
    .fiber_field_type = FiberFieldType::AxonBased
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  NDConfig config = config_brain_baseline;

  config.parse(argc, argv);

  //brain mesh
  auto sr = SeedingRegion<3>::create(config.seeding_region_type, config.C_0);
  
  // Create appropriate fiber field using factory
  auto fiber_field = FiberFieldFactory<3>::create(
    config.fiber_field_type,
    brain_origin,
    Point<3>(25, 35, 20)  // Default semi-axes for axon-based field
  );

  NDProblem<3> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, sr, *fiber_field);

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