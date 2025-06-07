#ifndef FIBERFIELDS_HPP
#define FIBERFIELDS_HPP

#include "NDProblem.hpp" // Assuming this contains NDProblem and deal.II includes
#include <cmath>        // For std::sqrt
#include <numeric>      // For std::inner_product if needed, not here.
#include <array>        // Potentially for semi_axes, but Point<DIM> is fine.

using namespace dealii;

template <unsigned int DIM>
class RadialFiberField : public NDProblem<DIM>::FiberField
{
  public:
    virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override
    {
      double distance = p.distance(_origin);
      double normalizer = distance + 1e-5; 

      for(unsigned int i = 0; i < DIM; ++i)
        values[i] = (p[i] - _origin[i]) / normalizer;
    }

    virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override
    {
      double distance = p.distance(_origin);
      double normalizer = distance + 1e-5;
      return (p[component] - _origin[component]) / normalizer;
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
    // Constructor for CircumferentialFiberField
    // semi_axis_proj_1:
    //   For DIM=2: semi-axis for the first coordinate of _center_proj (effectively X-axis).
    //   For DIM=3: semi-axis for the first coordinate of _center_proj (Y-axis in YZ projection).
    // semi_axis_proj_2:
    //   For DIM=2: semi-axis for the second coordinate of _center_proj (effectively Y-axis).
    //   For DIM=3: semi-axis for the second coordinate of _center_proj (Z-axis in YZ projection).
    CircumferentialFiberField(Point<DIM> center = Point<DIM>(), 
                              double semi_axis_proj_1 = 1.0, 
                              double semi_axis_proj_2 = 1.0);

    virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override;
    virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override;

  private:
    Point<2> _center_proj;      
    double _semi_axis_proj_1; 
    double _semi_axis_proj_2; 
};

template <unsigned int DIM>
class AxonBasedFiberField : public NDProblem<DIM>::FiberField
{
  static_assert(DIM == 2 || DIM == 3, "AxonBasedFiberField is only defined for 2D and 3D");

  public:
    // Primary constructor:
    // center: Center of the axon structure.
    // axon_semi_axes: Defines the semi-axes of the boundary ellipsoid/ellipse.
    //   For DIM=2: Point<2>(semi_axis_x, semi_axis_y)
    //   For DIM=3: Point<3>(semi_axis_x, semi_axis_y, semi_axis_z)
    AxonBasedFiberField(Point<DIM> center, Point<DIM> axon_semi_axes)
      : _axon_center(center)
      , _axon_semi_axes(axon_semi_axes)
      , _radial_fiber_field(center)
      // Initialize CircumferentialFiberField with appropriate semi-axes:
      // For DIM=2, CircField uses X and Y axes from axon_semi_axes.
      // For DIM=3, CircField (projects to YZ) uses Y and Z axes from axon_semi_axes.
      , _circumferential_fiber_field(
            center,
            /* semi_axis_proj_1 for CircField */ (DIM == 2 ? axon_semi_axes[0] : axon_semi_axes[1]),
            /* semi_axis_proj_2 for CircField */ (DIM == 2 ? axon_semi_axes[1] : axon_semi_axes[2])
        )
    {
        for (unsigned int i = 0; i < DIM; ++i)
        {
            // Add assertions if using deal.II's AssertThrow
            // AssertThrow(_axon_semi_axes[i] > 1e-9, ExcMessage("Semi-axis must be positive."));
            if (_axon_semi_axes[i] <= 1e-9)
                throw std::runtime_error("Semi-axis must be positive.");
        }
    }

    // Convenience constructor with default semi-axes (sphere/circle of radius 20)
    AxonBasedFiberField(Point<DIM> center = Point<DIM>())
      : AxonBasedFiberField(center, default_axon_boundary_semi_axes())
    {}


    virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override
    {
      if(is_inside_boundary_ellipsoid(p))
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
      if(is_inside_boundary_ellipsoid(p))
      {
        return _circumferential_fiber_field.value(p, component);
      }
      else
      {
        return _radial_fiber_field.value(p, component);
      }
    }

  private:
    // Helper to provide default semi-axes for the boundary (sphere/circle of radius 20.0)
    static Point<DIM> default_axon_boundary_semi_axes()
    {
        Point<DIM> axes;
        for (unsigned int i = 0; i < DIM; ++i)
            axes[i] = 20.0;
        return axes;
    }

    bool is_inside_boundary_ellipsoid(const Point<DIM>& p) const
    {
        double sum_of_normalized_squares = 0.0;
        for (unsigned int i = 0; i < DIM; ++i)
        {
            const double delta_coord = p[i] - _axon_center[i];
            // _axon_semi_axes[i] is guaranteed to be > 1e-9 by constructor check
            const double term = delta_coord / _axon_semi_axes[i];
            sum_of_normalized_squares += term * term;
        }
        return sum_of_normalized_squares < 1.0;
    }

    Point<DIM> _axon_center;
    Point<DIM> _axon_semi_axes; // Stores semi-axes for the boundary ellipsoid/ellipse
                                // DIM=2: [ax, ay], DIM=3: [ax, ay, az]
    RadialFiberField<DIM> _radial_fiber_field;
    CircumferentialFiberField<DIM> _circumferential_fiber_field;
};

// Forward declarations for template specializations of CircumferentialFiberField
// (assumed to be implemented in a .cpp file)
template <>
CircumferentialFiberField<2>::CircumferentialFiberField(Point<2> center, double semi_axis_proj_1, double semi_axis_proj_2);
template <>
void CircumferentialFiberField<2>::vector_value(const Point<2> &p, Vector<double> &values) const;
template <>
double CircumferentialFiberField<2>::value(const Point<2> &p, const unsigned int component) const;

template <>
CircumferentialFiberField<3>::CircumferentialFiberField(Point<3> center, double semi_axis_proj_1, double semi_axis_proj_2);
template <>
void CircumferentialFiberField<3>::vector_value(const Point<3> &p, Vector<double> &values) const;
template <>
double CircumferentialFiberField<3>::value(const Point<3> &p, const unsigned int component) const;

enum class FiberFieldType
{
  Radial = 0,
  Circumferential = 1,
  AxonBased = 2
};

template <unsigned int DIM>
class FiberFieldFactory {
public:
  static std::unique_ptr<typename NDProblem<DIM>::FiberField> create(FiberFieldType type, const Point<DIM>& center = Point<DIM>(), const Point<DIM>& semi_axes = Point<DIM>()) {
    switch (type) {
      case FiberFieldType::Radial:
        return std::make_unique<RadialFiberField<DIM>>(center);
      case FiberFieldType::Circumferential:
        return std::make_unique<CircumferentialFiberField<DIM>>(center, semi_axes[0], semi_axes[1]);
      case FiberFieldType::AxonBased:
        return std::make_unique<AxonBasedFiberField<DIM>>(center, semi_axes);
      default:
        throw std::invalid_argument("Unknown fiber field type");
    }
  }
};

#endif // FIBERFIELDS_HPP