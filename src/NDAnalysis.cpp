#include <getopt.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "ThetaSolver.hpp"
#include "BESolver.hpp"
#include "FESolver.hpp"
#include "InitialConditions.hpp"
#include "FiberFields.hpp"
#include "SeedingRegions.hpp"

using namespace dealii;

int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Variables to hold parameter values
  unsigned int dim = 2;
  double T = 1.0;
  double alpha = 0.45;
  double deltat = 0.01;
  unsigned int degree = 1;
  double d_ext = 1.5;
  double d_axn = 3.0;
  double C_0 = 0.4;
  const char *mesh = "../meshes/mesh-square-40.msh";
  const char *output_filename = "output";
  const char *output_dir = "./";
  int seeding_region_type = 0;

  // getopt options
  int opt;
  while ((opt = getopt(argc, argv, "d:T:a:t:e:x:m:g:h:D:o:s:c:")) != -1) {
    switch (opt) {
      case 'D':
        dim = std::atoi(optarg);
        break;
      case 'T':
        T = std::atof(optarg);
        break;
      case 'a':
        alpha = std::atof(optarg);
        break;
      case 't':
        deltat = std::atof(optarg);
        break;
      case 'g':
        degree = std::atoi(optarg);
        break;
      case 'e':
        d_ext = std::atof(optarg);
        break;
      case 'x':
        d_axn = std::atof(optarg);
        break;
      case 'c':
        C_0 = std::atof(optarg);
        break;
      case 'm':
        mesh = optarg;
        break;
      case 'o':
        output_filename = optarg;
        break;
      case 'd':
        output_dir = optarg;
        break;
      case 's':
        seeding_region_type = std::atoi(optarg);
        break;
      case 'h':
        std::cerr << "Usage: " << argv[0] << " [-D dim] [-T T] [-a alpha] [-t deltat] [-g degree] [-e d_ext] [-x d_axn] [-m mesh] [-o output_filename] [-d output_dir] [-s seeding_region_type]\n";
        std::cerr << "  -m: mesh file\n";
        std::cerr << "  -D: dimension of the problem\n";
        std::cerr << "  -a: growth factor\n";
        std::cerr << "  -e: extracellular diffusion coefficient\n";
        std::cerr << "  -x: axonal diffusion coefficient\n";
        std::cerr << "  -g: degree of the finite element\n";
        std::cerr << "  -t: time step\n";
        std::cerr << "  -T: final time\n";
        std::cerr << "  -o: output filename\n";
        std::cerr << "  -d: output directory\n";
        std::cerr << "  -s: seeding region type (0: AmyloidBetaDeposit, 1: TauInclusions, 2: TDP43Inclusions, 3: AlphaSynucleinInclusions)\n";
        exit(EXIT_SUCCESS);
      default:
        std::cerr << "Usage: " << argv[0] << " [-D dim] [-T T] [-a alpha] [-t deltat] [-g degree] [-e d_ext] [-x d_axn] [-m mesh] [-o output_filename] [-d output_dir] [-s seeding_region_type]\n";
        exit(EXIT_FAILURE);
    }
  }

  switch(dim){
    case 1:
      {
        ExponentialInitialCondition<1> initial_condition;
        RadialFiberField<1> fiber_field;
        NDProblem<1> problem(mesh, alpha, d_ext, d_axn, initial_condition, fiber_field);
        BESolver<1> solver(problem, deltat, T, degree, output_dir, output_filename);
        problem.export_problem(std::string(output_dir) + output_filename + ".problem");
        solver.setup();
        solver.solve();
      }
      break;
    case 2:
      {
        const Point<2> random_point(0.7, 0.7);
        ConstantInitialCondition<2> initial_condition(1.0, random_point, 0.1);
        AxonBasedFiberField_2D fiber_field(0.3, square_origin);
        NDProblem<2> problem(mesh, alpha, d_ext, d_axn, initial_condition, fiber_field);
        BESolver<2> solver(problem, deltat, T, degree, output_dir, output_filename);
        problem.export_problem(std::string(output_dir) + output_filename + ".problem");
        solver.setup();
        solver.solve();
      }
      break;
    case 3:
      {
        std::unique_ptr<SeedingRegion> sr;
        switch(seeding_region_type) {
            case 0: sr = std::make_unique<AlphaSynucleinInclusions>(C_0); break;
            case 1: sr = std::make_unique<AmyloidBetaDeposit>(C_0); break;
            case 2: sr = std::make_unique<TauInclusions>(C_0); break;
            case 3: sr = std::make_unique<TDP43Inclusions>(C_0); break;
            default: sr = std::make_unique<AlphaSynucleinInclusions>(C_0);
        }
        CircumferentialFiberField3D fiber_field(brain_origin);
        NDProblem<3> problem(mesh, alpha, d_ext, d_axn, *sr, fiber_field);
        BESolver<3> solver(problem, deltat, T, degree, output_dir, output_filename);
        problem.export_problem(std::string(output_dir) + output_filename + ".problem");
        solver.setup();
        solver.solve();
      }
      break;
    default:
      std::cerr << "Invalid dimension\n";
      exit(EXIT_FAILURE);
  }
  return 0;
}