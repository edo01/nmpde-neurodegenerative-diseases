#include "NeurodegenerativeDisease.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 1;

  const double T      = 0.01;
  const double deltat = 0.001;

  NeurodegenerativeDisease::FiberField fiber_field;

  NeurodegenerativeDisease problem("../meshes/brain-h3.0.msh", degree, T, deltat, fiber_field);

  problem.setup();
  problem.solve();
  return 0;
}