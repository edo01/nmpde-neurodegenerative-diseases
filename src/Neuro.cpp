#include "NeurodegenerativeDisease.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 2;


  const double T      = 0.00010;
  const double deltat = 0.000001;

  NeurodegenerativeDisease::FiberField fiber_field;

  NeurodegenerativeDisease problem("../meshes/mesh-square-40.msh", degree, T, deltat, fiber_field);

  problem.setup();
  problem.solve();
  return 0;
}
