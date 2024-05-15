#ifndef INITIAL_CONDITIONS_HPP
#define INITIAL_CONDITIONS_HPP

#include "NDSolver.hpp"

using namespace dealii;

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

#endif // INITIAL_CONDITIONS_HPP