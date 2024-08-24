#ifndef INITIAL_CONDITIONS_HPP
#define INITIAL_CONDITIONS_HPP

#include "NDSolver.hpp"
#include "SeedingRegions.hpp"

using namespace dealii;

/**
 * @brief Constant initial condition.
 * 
 * @note If ray = 0 (default value), the initial condition is constant on the whole domain.
 * 
 * @tparam DIM Dimension of the problem.
 */

template<unsigned int DIM>
class ConstantInitialCondition: public NDProblem<DIM>::InitialConcentration
{
    public:
        virtual double value(const Point<DIM> &p, const unsigned int /*component*/ = 0) const override
        {
          if (_sr.is_inside(p))
            return _C_0;
          return 0.0;
        }
      
      ConstantInitialCondition(double C_0=0.4, SeedingRegion<DIM> sr = SeedingRegion<DIM>())
        : _C_0(C_0), _sr(sr) {}
        
    private:
      double _C_0;
      SeedingRegion<DIM> _sr;
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
      
      ExponentialInitialCondition(double sigma = 0.1, double C_0=0.4, )
        : _C_0(C_0), _origin(origin), _ray(ray), sigma(sigma) {}
        
    private:
      double _C_0;
      Point<DIM> _origin;
      double _ray;
      double sigma;
};

template<unsigned int DIM>
class QuadraticInitialCondition: public NDProblem<DIM>::InitialConcentration
{
    public:
        virtual double value(const Point<DIM> &p, const unsigned int /*component*/ = 0) const override
        {
            double distance_from_origin_squared = p.distance_square(_origin);
            if(distance_from_origin_squared > _ray_squared)
              return 0.0;
            return _C_0*(1 - distance_from_origin_squared/(_ray_squared));
        }

        QuadraticInitialCondition(Point<DIM> origin = Point<DIM>(), double C_0=0.9, double ray=5)
        : _C_0(C_0), _origin(origin), _ray_squared(ray*ray) {}

    private:
        double _C_0;
        Point<DIM> _origin;
        double _ray_squared;
};

#endif // INITIAL_CONDITIONS_HPP