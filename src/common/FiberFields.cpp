#include "FiberFields.hpp"

//Circumferential fiber field in 2D

template <>
CircumferentialFiberField<2>::CircumferentialFiberField(Point<2> center_proj) : _center_proj(center_proj) {}

template <>
void CircumferentialFiberField<2>::vector_value(const Point<2> &p, Vector<double> &values) const
{
  auto distance_from_origin = p.distance(_center_proj);
  values[0] = -(p[1] - _center_proj[0]) / (distance_from_origin + 1e-5);
  values[1] = (p[0] - _center_proj[1]) / (distance_from_origin + 1e-5);
}

template <>
double CircumferentialFiberField<2>::value(const Point<2> &p, const unsigned int component) const
{
  auto distance_from_origin = p.distance(_center_proj);
  if(component == 0)
  {
    return -(p[1] - _center_proj[0]) / (distance_from_origin + 1e-5);
  }
  else
  {
    return (p[0] - _center_proj[1]) / (distance_from_origin + 1e-5);
  }
}

//Circumferential fiber field in 3D

template <>
CircumferentialFiberField<3>::CircumferentialFiberField(Point<3> center) : _center_proj(Point<2>(center[1], center[2])) {}

template <>
void CircumferentialFiberField<3>::vector_value(const Point<3> &p, Vector<double> &values) const
{
  Point<2> p_projection(p[1], p[2]);

  auto distance_from_projection = p_projection.distance(_center_proj);
  values[0] = 0;
  values[1] = -(p[2] - _center_proj[1]) / (distance_from_projection + 1e-5);
  values[2] = (p[1] - _center_proj[0]) / (distance_from_projection + 1e-5);
}

template <>
double CircumferentialFiberField<3>::value(const Point<3> &p, const unsigned int component) const
{
  Point<2> p_projection(p[1], p[2]);

  auto distance_from_projection = p_projection.distance(_center_proj);
  if(component == 0)
  {
    return 0;
  }
  else if(component == 1)
  {
    return -(p[2] - _center_proj[1]) / (distance_from_projection + 1e-5);
  }
  else
  {
    return (p[1] - _center_proj[0]) / (distance_from_projection + 1e-5);
  }
}