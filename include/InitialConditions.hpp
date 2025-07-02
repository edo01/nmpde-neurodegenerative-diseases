#ifndef INITIAL_CONDITIONS_HPP
#define INITIAL_CONDITIONS_HPP

#include "NDProblem.hpp"

// small value to be set as initial condition in the region where the initial condition is 0 to avoid the solution to go below zero
//#define EPSILON 5e-3
#define EPSILON 0

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
                // return 0.0;
                return EPSILON;
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
              // return 0.0;
              return EPSILON;
            return C_0*std::exp(-distance_from_origin*distance_from_origin/(2*sigma*sigma)) + EPSILON;
        }
      
      ExponentialInitialCondition(Point<DIM> origin_ = Point<DIM>(), double sigma_ = 0.1, double C_0_ = 0.4, double ray_ = 4)
        : C_0(C_0_), origin(origin_), ray(ray_), sigma(sigma_) {}
        
    private:
      double C_0;
      Point<DIM> origin;
      double ray;
      double sigma;
};

template <unsigned int DIM>
class SmoothBumpInitialCondition : public NDProblem<DIM>::InitialConcentration {
public:
    virtual double value(const Point<DIM>& p, const unsigned int /*component*/ = 0) const override {
        double distance_from_origin_sq = p.distance(origin) * p.distance(origin);
        double r_sq = distance_from_origin_sq / (ray * ray); // Normalized squared radius

        if (r_sq >= 1.0) {
            return EPSILON; // Or 0.0 if your solver handles it and you want strict zero
        } else {
            // Standard bump function form: C_0 * exp(1 - 1/(1 - r^2))
            // The exp(1) can be absorbed into C_0 if desired for a slightly cleaner look
            // or use exp(-1 / (1 - r_sq)) and adjust C_0 accordingly
            double bump_val = C_0 * std::exp(1.0 - 1.0 / (1.0 - r_sq));
            return bump_val + EPSILON; // Still good practice for numerical stability
            // Or: return std::max(bump_val, EPSILON);
        }
    }

    SmoothBumpInitialCondition(Point<DIM> origin_ = Point<DIM>(),
                               double C_0_ = 0.4,
                               double ray_ = 4.0) 
        : C_0(C_0_), origin(origin_), ray(ray_) {
        }

private:
    double C_0;      // Amplitude
    Point<DIM> origin;
    double ray;      // Radius of the bump's support
    // static const double EPSILON = 1e-9; // Or define globally
};

template<unsigned int DIM>
class QuadraticInitialCondition: public NDProblem<DIM>::InitialConcentration
{
    public:
        virtual double value(const Point<DIM> &p, const unsigned int /*component*/ = 0) const override
        {
            double distance_from_origin_squared = p.distance_square(origin);
            if(distance_from_origin_squared > ray_squared)
              // return 0.0;
              return EPSILON;
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