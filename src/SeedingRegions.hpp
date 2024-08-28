#ifndef SEEDING_REGIONS_HPP
#define SEEDING_REGIONS_HPP

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/point.h>
#include <memory>
#include <cmath>
#include "NDProblem.hpp"

using namespace dealii;
using vertices = std::pair<Point<3>, Point<3>>;

class SeedingRegion : public NDProblem<3>::InitialConcentration
{
public:
    bool is_inside(const Point<3> &p) const {
        for (const auto &region : _regions) {
            auto result = GridTools::find_active_cell_around_point(region, p);
            if (result.state() == IteratorState::valid) {
                return true;
            }
        }
        return false;
    }

    double value(const Point<3> &p, const unsigned int /* component */ = 0) const override
    {
        return is_inside(p) ? _C_0 : 0.0;
    }

    void create_regions(const std::vector<vertices> &vertices) {
        for (const auto &vertex : vertices) {
            Triangulation<3> tria;
            GridGenerator::hyper_rectangle(tria, vertex.first, vertex.second);
            _regions.push_back(std::move(tria));
        }
    }

    SeedingRegion(double C_0) : _C_0(C_0) {}
    virtual ~SeedingRegion() = default;
private:
    double _C_0;
    std::vector<Triangulation<3>> _regions;
};

class TauInclusions : public SeedingRegion
{
public:
    TauInclusions(double C_0) : SeedingRegion(C_0) {
        std::vector<vertices> vertices;
        vertices.push_back(std::make_pair(Point<3>(63, 70, 56), Point<3>(81, 90, 69)));
        create_regions(vertices);
    }
};

class AmyloidBetaDeposit : public SeedingRegion
{
public:
    AmyloidBetaDeposit(double C_0) : SeedingRegion(C_0) {
        std::vector<vertices> vertices;
        vertices.push_back(std::make_pair(Point<3>(23, 22, 95), Point<3>(82, 80, 118)));
        vertices.push_back(std::make_pair(Point<3>(23, 100, 95), Point<3>(82, 135, 118)));
        create_regions(vertices);
    }
};

class TDP43Inclusions : public SeedingRegion
{
public:
    TDP43Inclusions(double C_0) : SeedingRegion(C_0) {
        std::vector<vertices> vertices;
        vertices.push_back(std::make_pair(Point<3>(23, 48, 85), Point<3>(82, 75, 117)));
        vertices.push_back(std::make_pair(Point<3>(63, 80, 44), Point<3>(81, 90, 57)));
        create_regions(vertices);
    }
};

class AlphaSynucleinInclusions : public SeedingRegion
{
public:
    AlphaSynucleinInclusions(double C_0) : SeedingRegion(C_0) {
        std::vector<vertices> vertices;
        vertices.push_back(std::make_pair(Point<3>(63, 75, 44), Point<3>(81, 90, 57)));
        create_regions(vertices);
    }
};

#endif // SEEDING_REGIONS_HPP