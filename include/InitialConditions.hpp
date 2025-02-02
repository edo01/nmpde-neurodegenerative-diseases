#ifndef INITIAL_CONDITIONS_HPP
#define INITIAL_CONDITIONS_HPP

#include "NDProblem.hpp"

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
            if(ray == 0.0)
              return C_0;
            if(p.distance(origin) > ray)
                return 0.0;
            return C_0;
        }
      
      ConstantInitialCondition(double C_0_, Point<DIM> origin_, double ray_)
        : C_0(C_0_), origin(origin_), ray(ray_) {}
        
    private:
      double C_0;
      Point <DIM> origin;
      double ray;
};

template<unsigned int DIM>
class ExponentialInitialCondition: public NDProblem<DIM>::InitialConcentration
{
    public:
        virtual double value(const Point<DIM> &p, const unsigned int /*component*/ = 0) const override
        {
            double distance_from_origin = p.distance(origin);
            if(distance_from_origin > ray)
              return 0.0;
            return C_0*std::exp(-distance_from_origin*distance_from_origin/(2*sigma*sigma));
        }
      
      ExponentialInitialCondition(Point<DIM> origin_ = Point<DIM>(), double sigma_ = 0.1, double C_0_ = 0.4, double ray_ = 4)
        : C_0(C_0_), origin(origin_), ray(ray_), sigma(sigma_) {}
        
    private:
      double C_0;
      Point<DIM> origin;
      double ray;
      double sigma;
};

template<unsigned int DIM>
class QuadraticInitialCondition: public NDProblem<DIM>::InitialConcentration
{
    public:
        virtual double value(const Point<DIM> &p, const unsigned int /*component*/ = 0) const override
        {
            double distance_from_origin_squared = p.distance_square(origin);
            if(distance_from_origin_squared > ray_squared)
              return 0.0;
            return C_0*(1 - distance_from_origin_squared/(ray_squared));
        }

        QuadraticInitialCondition(double C_0_, Point<DIM> origin_, double ray_)
        : C_0(C_0_), origin(origin_), ray_squared(ray_ * ray_) {}

    private:
        double C_0;
        Point<DIM> origin;
        double ray_squared;
};

#endif // INITIAL_CONDITIONS_HPP