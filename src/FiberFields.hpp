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

          for (unsigned int i = 0; i < DIM; ++i)
            values[i] = (p[i] - _origin[i]) / (p.distance(_origin) + 1e-5);
        }
    
        virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override
        {
            double distance_from_origin = p.distance(_origin);
            return (p[component] - _origin[component]) / (distance_from_origin + 1e-5);

        }

        RadialFiberField(Point<DIM> origin = Point<DIM>()
        ): _origin(origin){}
        
    private:
      Point<DIM> _origin;
};

class CircumferentialFiberField2D : public NDProblem<2>::FiberField
{
    public:
        virtual void vector_value(const Point<2> &p, Vector<double> &values) const override
        {
            auto distance_from_origin = p.distance(_origin);
            values[0] = -(p[1] - _origin[0]) / (distance_from_origin + 1e-5);
            values[1] = (p[0] - _origin[1]) / (distance_from_origin + 1e-5);
        }

        virtual double value(const Point<2> &p, const unsigned int component = 0) const override
        {
            auto distance_from_origin = p.distance(_origin);
            if (component == 0) {
                return -(p[1] - _origin[0]) / (distance_from_origin + 1e-5);
            } else {
                return (p[0] - _origin[1]) / (distance_from_origin + 1e-5);
            }
        }

        CircumferentialFiberField2D(Point<2> origin = Point<2>())
        : _origin(origin){}

    private:
        Point<2> _origin;
};

class CircumferentialFiberField3D : public NDProblem<3>::FiberField
{
    public:
        virtual void vector_value(const Point<3> &p, Vector<double> &values) const override
        {

            Point<2> p_projection(p[1], p[2]);

            auto distance_from_projection = p_projection.distance(_origin_projection);
            values[0] = 0;
            values[1] = -(p[2] - _origin_projection[1]) / (distance_from_projection + 1e-5);
            values[2] = (p[1] - _origin_projection[0]) / (distance_from_projection + 1e-5);
        }

        virtual double value(const Point<3> &p, const unsigned int component = 0) const override
        {
            Point<2> p_projection(p[1], p[2]);


            auto distance_from_projection = p_projection.distance(_origin_projection);
            if (component == 0) {
                return 0;
            } else if (component == 1) {
                return -(p[2] - _origin_projection[1]) / (distance_from_projection + 1e-5);
            } else {
                return (p[1] - _origin_projection[0]) / (distance_from_projection + 1e-5);
            }
        }

        CircumferentialFiberField3D(Point<3> origin = Point<3>())
        : _origin_projection(Point<2>(origin[1], origin[2])){}

    private:
        Point<2> _origin_projection;
};


#endif // FIBERFIELDS_HPP