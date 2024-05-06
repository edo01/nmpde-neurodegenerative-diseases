#include "NDSolver.hpp"

using namespace dealii;

template<unsigned int DIM>
class RadialFiberField: public NDProblem<DIM>::FiberField
{
    public:
        virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override
        {
          double distance_from_origin = p.distance(_origin);
          for (unsigned int i = 0; i < DIM; ++i)
            values[i] = p[i] / (distance_from_origin + 1e-10);
        }
    
        virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override
        {
            double distance_from_origin = p.distance(_origin);
            return p[component] / (distance_from_origin + 1e-10);
        }

        RadialFiberField(Point<DIM> origin = Point<DIM>()
        ): _origin(origin){}
        
    private:
      Point<DIM> _origin;

};


template<unsigned int DIM>
class ExponentialInitialCondition: public NDProblem<DIM>::InitialConcentration
{
    public:
        virtual double value(const Point<DIM> &p, const unsigned int /*component*/ = 0) const override
        {
            double distance_from_origin = p.distance(_origin);
            if(distance_from_origin > _ray)
              return 0.0;
            return _C_0*std::exp(-distance_from_origin*distance_from_origin/(2*sigma*sigma));
        }
      
      ExponentialInitialCondition(Point<DIM> origin = Point<DIM>(), double sigma = 0.1, double C_0=0.4, double ray=4)
        : _origin(origin), sigma(sigma), _C_0(C_0), _ray(ray) {}
        
    private:
      Point<DIM> _origin;
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
  const unsigned int dim = 3;

  const double T      = 1;
  const double deltat = 0.1;
  const double alpha = 0.5;
  const double d_ext = 0.1;
  const double d_axn = 0.3;

  RadialFiberField<dim> fiber_field;
  ExponentialInitialCondition<dim> initial_condition;

  NDProblem<dim> problem("../meshes/mesh-cube-10.msh", deltat,
          T, alpha, d_ext, d_axn, initial_condition, fiber_field);

  NDSolver<dim> solver(problem, degree);

  solver.setup();

  solver.solve();
  return 0;
}
