#ifndef INITIAL_CONDITIONS_HPP
#define INITIAL_CONDITIONS_HPP

#include "NDSolver.hpp"

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
            if(_ray == 0.0)
              return _C_0;
            if(p.distance(_origin) > _ray)
                return 0.0;
            return _C_0;
        }
      
      ConstantInitialCondition(double C_0=0.4, Point<DIM> origin = Point<DIM>(), double ray=0)
        : _C_0(C_0), _origin(origin), _ray(ray) {}
        
    private:
      double _C_0;
      Point <DIM> _origin;
      double _ray;
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