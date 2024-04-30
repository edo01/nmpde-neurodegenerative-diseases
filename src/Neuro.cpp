#include "NeurodegenerativeDisease.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 2;

  const double T      = 10;
//  const double deltat = 0.0000001;  // for square-40
    const double deltat = 0.1;

  NeurodegenerativeDisease::FiberField fiber_field;

  NeurodegenerativeDisease problem("../meshes/brain-h3.03D.msh", degree, T, deltat, fiber_field);

  problem.setup();

  problem.solve();
  return 0;
}
