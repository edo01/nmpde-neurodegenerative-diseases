#ifndef FIBERFIELDS_HPP
#define FIBERFIELDS_HPP

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


#endif // FIBERFIELDS_HPP