#include "NDProblem.hpp"
#include "NDConfig.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "NDThetaSolver.hpp"
#include "NDAdaptiveSolver.hpp"
#include "SeedingRegions.hpp"

static const Point<2> square_origin = Point<2>(0.5, 0.5);

static const Point<2> sagittal_origin = Point<2>(70.0, 73.0);

static NDConfig config_square = {
    .dim = 2,
    .T = 7.0,
    .alpha = 1.8,
    // .alpha = 0.0,
    .deltat = 0.01,
    .degree = 1,
    .d_ext = 0.00,
    .d_axn = 0.2,
    .C_0 = 0.9,
    .mesh = "../meshes/mesh-square-300.msh",
    //.mesh = "../meshes/slice_generated.msh",
};

static NDConfig config_sagittal = {
    .dim = 2,
    .T = 48.0,
    .alpha = 0.6,
    .deltat = 0.24,
    .degree = 1,
    .d_ext = 1.5,
    .d_axn = 3.0,
    .C_0 = 0.2,
    .mesh = "../meshes/slice_generated.msh",
    .gray_matter_distance_threshold = 4.0,
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

   NDConfig config = config_sagittal;
    // NDConfig config = config_square;

  config.parse(argc, argv);

  // square old config
  const Point<2> random_point(0.7, 0.7);
  //ConstantInitialCondition<2> initial_condition(config.C_0, random_point, 0.1);
  //CircumferentialFiberField<2> fiber_field(square_origin);

  // sagittal section config
  const Point<2> seeding_center(87.0, 58.0);
  // SmoothBumpInitialCondition<2> initial_condition(seeding_center, config.C_0, 5.0); // Old hardcoded

  // Create seeding region based on parsed type
  auto initial_condition = SeedingRegion<2>::create(config.seeding_region_type, config.C_0);

  Point<2> default_semi_axes(35.0, 15.0); // Default semi-axes, adjust if necessary
  auto fiber_field = FiberFieldFactory<2>::create(
    config.fiber_field_type,
    sagittal_origin, // Using sagittal_origin as the center
    default_semi_axes  // Default semi-axes
  );

  NDProblem<2> problem(config.mesh, config.alpha, config.d_ext, config.d_axn, initial_condition, *fiber_field, config.gray_matter_distance_threshold);
  NDBackwardEulerSolver<2> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);
  //NDCrankNicolsonSolver<2> solver(problem, config.deltat, config.T, config.degree, config.output_dir, config.output_filename);

  problem.export_problem(config.output_dir + config.output_filename + ".problem");
  solver.setup();
  solver.solve();

  return EXIT_SUCCESS;
}