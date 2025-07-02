#include "FiberFields.hpp"
#include <cmath> // For std::sqrt

//Circumferential fiber field in 2D

// Constructor for 2D
template <>
CircumferentialFiberField<2>::CircumferentialFiberField(Point<2> center, double semi_axis_1, double semi_axis_2)
    : _center_proj(center), _semi_axis_proj_1(semi_axis_1), _semi_axis_proj_2(semi_axis_2) {}

template <>
void CircumferentialFiberField<2>::vector_value(const Point<2> &p, Vector<double> &values) const
{
    // dx = p[0] (p_x) - _center_proj[0] (center_x)
    const double dx = p[0] - _center_proj[0];
    // dy = p[1] (p_y) - _center_proj[1] (center_y)
    const double dy = p[1] - _center_proj[1];

    // For an ellipse (x/a)^2 + (y/b)^2 = 1, centered at origin,
    // a tangent vector is (-y/b^2, x/a^2).
    // Here, 'a' corresponds to _semi_axis_proj_1 (for x-direction)
    // and 'b' corresponds to _semi_axis_proj_2 (for y-direction).
    const double val0_unnormalized = -dy / (_semi_axis_proj_2 * _semi_axis_proj_2);
    const double val1_unnormalized =  dx / (_semi_axis_proj_1 * _semi_axis_proj_1);

    const double magnitude = std::sqrt(val0_unnormalized * val0_unnormalized + val1_unnormalized * val1_unnormalized);
    const double normalizer = magnitude + 1e-5; // Add epsilon to avoid division by zero

    // If magnitude is effectively zero (e.g. at the center point AND semi-axes are non-zero),
    // unnormalized components will also be zero. Then 0.0/1e-5 = 0.0.
    values[0] = val0_unnormalized / normalizer;
    values[1] = val1_unnormalized / normalizer;
}

template <>
double CircumferentialFiberField<2>::value(const Point<2> &p, const unsigned int component) const
{
    const double dx = p[0] - _center_proj[0];
    const double dy = p[1] - _center_proj[1];

    const double val0_unnormalized = -dy / (_semi_axis_proj_2 * _semi_axis_proj_2);
    const double val1_unnormalized =  dx / (_semi_axis_proj_1 * _semi_axis_proj_1);
    
    const double magnitude = std::sqrt(val0_unnormalized * val0_unnormalized + val1_unnormalized * val1_unnormalized);
    const double normalizer = magnitude + 1e-5;

    if (component == 0)
    {
        return val0_unnormalized / normalizer;
    }
    else // component == 1
    {
        return val1_unnormalized / normalizer;
    }
}

//Circumferential fiber field in 3D

// Constructor for 3D
template <>
CircumferentialFiberField<3>::CircumferentialFiberField(Point<3> center, double semi_axis_1, double semi_axis_2)
    : _center_proj(Point<2>(center[1], center[2])), // Project center onto YZ plane: (center_y, center_z)
      _semi_axis_proj_1(semi_axis_1), // Semi-axis for the first projected coordinate (Y)
      _semi_axis_proj_2(semi_axis_2)  // Semi-axis for the second projected coordinate (Z)
{}

template <>
void CircumferentialFiberField<3>::vector_value(const Point<3> &p, Vector<double> &values) const
{
    // p_projection[0] is p[1] (p_y), _center_proj[0] is center_y_proj
    const double d_proj_coord1 = p[1] - _center_proj[0]; // displacement along projected Y-axis
    // p_projection[1] is p[2] (p_z), _center_proj[1] is center_z_proj
    const double d_proj_coord2 = p[2] - _center_proj[1]; // displacement along projected Z-axis

    // Ellipse in YZ plane. Semi-axis for Y-projection is _semi_axis_proj_1 ('a')
    // Semi-axis for Z-projection is _semi_axis_proj_2 ('b')
    // Tangent vector (in YZ plane) components (v_y, v_z) are (-d_proj_coord2/b^2, d_proj_coord1/a^2)
    const double val1_unnormalized = -d_proj_coord2 / (_semi_axis_proj_2 * _semi_axis_proj_2); // Fiber Y-component
    const double val2_unnormalized =  d_proj_coord1 / (_semi_axis_proj_1 * _semi_axis_proj_1); // Fiber Z-component

    const double magnitude_proj = std::sqrt(val1_unnormalized * val1_unnormalized + val2_unnormalized * val2_unnormalized);
    const double normalizer = magnitude_proj + 1e-5;

    values[0] = 0.0; // Fiber is in YZ plane, so X-component is 0
    values[1] = val1_unnormalized / normalizer;
    values[2] = val2_unnormalized / normalizer;
}

template <>
double CircumferentialFiberField<3>::value(const Point<3> &p, const unsigned int component) const
{
    if (component == 0)
    {
        return 0.0;
    }

    const double d_proj_coord1 = p[1] - _center_proj[0]; 
    const double d_proj_coord2 = p[2] - _center_proj[1]; 

    const double val1_unnormalized = -d_proj_coord2 / (_semi_axis_proj_2 * _semi_axis_proj_2);
    const double val2_unnormalized =  d_proj_coord1 / (_semi_axis_proj_1 * _semi_axis_proj_1);

    const double magnitude_proj = std::sqrt(val1_unnormalized * val1_unnormalized + val2_unnormalized * val2_unnormalized);
    const double normalizer = magnitude_proj + 1e-5;
    
    if (component == 1)
    {
        return val1_unnormalized / normalizer;
    }
    else // component == 2
    {
        return val2_unnormalized / normalizer;
    }
}