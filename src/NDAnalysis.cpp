#include "NDSolver.hpp"

using namespace dealii;
constexpr unsigned int dim = 2;

class RadialFiberField: public NDProblem::FiberField
{
    public:
        virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
          double distance_from_origin = p.distance(_origin);
          for (unsigned int i = 0; i < dim; ++i)
            values[i] = p[i] / (distance_from_origin + 1e-10);
        }
    
        virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            double distance_from_origin = p.distance(_origin);
            return p[component] / (distance_from_origin + 1e-10);
        }

        RadialFiberField(Point<dim> origin = Point<dim>()
        ): _origin(origin){}
        
    private:
      Point<dim> _origin;

};

class ExponentialInitialCondition: public NDProblem::InitialConcentration
{
    public:
        virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
        {
            double distance_from_origin = p.distance(_origin);
            if(distance_from_origin > _ray)
              return 0.0;
            return _C_0*std::exp(-distance_from_origin*distance_from_origin/(2*sigma*sigma));
        }
      
      ExponentialInitialCondition(Point<dim> origin = Point<dim>(), double sigma = 0.1, double C_0=0.4, double ray=4)
        : _origin(origin), sigma(sigma), _C_0(C_0), _ray(ray) {}
        
    private:
      Point<dim> _origin;
      double sigma;
      double _C_0;
      double _ray;
};

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree = 2;

  const double T      = 10;
  const double deltat = 0.1;
  const double alpha = 0.5;
  const double d_ext = 0.1;
  const double d_axn = 0.3;

  RadialFiberField fiber_field;
  ExponentialInitialCondition initial_condition;

  NDProblem problem("../meshes/mesh-square-40.msh", deltat,
          T, alpha, d_ext, d_axn, initial_condition, fiber_field);

  NDSolver solver(problem, degree);

  solver.setup();

  solver.solve();
  return 0;
}
