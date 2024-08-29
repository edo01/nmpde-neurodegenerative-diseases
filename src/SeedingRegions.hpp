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

class SeedingRegion : public NDProblem<3>::InitialConcentration
{
using CornerPair = std::pair<Point<3>, Point<3>>;
public:
    double value(const Point<3> &p, const unsigned int /* component */ = 0) const override
    {
        return is_inside(p) ? _C_0 : 0.0;
    }

protected:
    SeedingRegion(double C_0, const std::vector<CornerPair>& corners)
        : _C_0(C_0), _region() {
        create_region(corners);
    }

private:
    const double _C_0;
    Triangulation<3> _region;

    bool is_inside(const Point<3> &p) const {

        auto result = GridTools::find_active_cell_around_point(_region, p);
        if(result.state() == IteratorState::valid) {
            return true;
        }
        else {
            return false;
        }
    }

    void create_region(const std::vector<CornerPair>& corners) {
        std::vector<Triangulation<3>> regions;

        for (const auto& corner_pair : corners) {
            regions.emplace_back();
            GridGenerator::hyper_rectangle(regions.back(), corner_pair.first, corner_pair.second);
            GridGenerator::merge_triangulations(_region, regions.back(), _region);
        }
    }
};

class TauInclusions : public SeedingRegion
{
public:
    TauInclusions(double C_0)
        : SeedingRegion(C_0, {
            std::make_pair(Point<3>(23, 48, 85), Point<3>(82, 75, 117))
            }) {}
};

class AmyloidBetaDeposit : public SeedingRegion
{
public:
    AmyloidBetaDeposit(double C_0)
        : SeedingRegion(C_0, {
            std::make_pair(Point<3>(23, 22, 95), Point<3>(82, 80, 118)),
            std::make_pair(Point<3>(23, 100, 95), Point<3>(82, 135, 118))
            }) {}
};

class TDP43Inclusions : public SeedingRegion
{
public:
    TDP43Inclusions(double C_0) 
        : SeedingRegion(C_0, {
            std::make_pair(Point<3>(23, 48, 85), Point<3>(82, 75, 117)),
            std::make_pair(Point<3>(63, 80, 44), Point<3>(81, 90, 57))
            }) {}
};

class AlphaSynucleinInclusions : public SeedingRegion
{
public:
    AlphaSynucleinInclusions(double C_0)
        : SeedingRegion(C_0, {
            std::make_pair(Point<3>(63, 75, 44), Point<3>(81, 80, 57))
            }) {}
};

#endif // SEEDING_REGIONS_HPP