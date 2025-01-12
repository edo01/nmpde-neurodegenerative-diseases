#ifndef FIBERFIELDS_HPP
#define FIBERFIELDS_HPP

#include "NDSolver.hpp"

using namespace dealii;

template <unsigned int DIM>
class RadialFiberField : public NDProblem<DIM>::FiberField
{
  public:
    virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override
    {
      for(unsigned int i = 0; i < DIM; ++i)
        values[i] = (p[i] - _origin[i]) / (p.distance(_origin) + 1e-5);
    }

    virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override
    {
      double distance_from_origin = p.distance(_origin);
      return (p[component] - _origin[component]) / (distance_from_origin + 1e-5);
    }

    RadialFiberField(Point<DIM> origin = Point<DIM>()) : _origin(origin) {}

  private:
    Point<DIM> _origin;
};

template <unsigned int DIM>
class CircumferentialFiberField : public NDProblem<DIM>::FiberField
{
  static_assert(DIM == 2 || DIM == 3, "CircumferentialFiberField is only defined for 2D and 3D");
  public:
    CircumferentialFiberField(Point<DIM> center = Point<DIM>());

    virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override;
    virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override;

  private:
    Point<2> _center_proj;
};

template <unsigned int DIM>
class AxonBasedFiberField : public NDProblem<DIM>::FiberField
{
  static_assert(DIM == 2 || DIM == 3, "AxonBasedFiberField is only defined for 2D and 3D");

  public:
    AxonBasedFiberField(double radius = 20, Point<DIM> center = Point<DIM>())
      : _axon_radius(radius)
      , _axon_center(center)
      , _radial_fiber_field(center)
      , _circumferential_fiber_field(center) 
    {}

    virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override
    {
      auto distance_from_axon_center = p.distance(_axon_center);
      if(distance_from_axon_center < _axon_radius)
      {
        _circumferential_fiber_field.vector_value(p, values);
      }
      else
      {
        _radial_fiber_field.vector_value(p, values);
      }
    }

    virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override
    {
      auto distance_from_axon_center = p.distance(_axon_center);
      if(distance_from_axon_center < _axon_radius)
      {
        return _circumferential_fiber_field.value(p, component);
      }
      else
      {
        return _radial_fiber_field.value(p, component);
      }
    }

  private:
    double _axon_radius;
    Point<DIM> _axon_center;
    RadialFiberField<DIM> _radial_fiber_field;
    CircumferentialFiberField<DIM> _circumferential_fiber_field;
};

#endif // FIBERFIELDS_HPP
