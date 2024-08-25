#ifndef SEEDING_REGIONS_HPP
#define SEEDING_REGIONS_HPP

#include <deal.II/base/point.h>
#include <memory>
#include <cmath>

using namespace dealii;

enum class SeedingRegionType
{
    TauInclusions,
    AmyloidBetaDeposit,
    TDP43Inclusions,
    AlphaSynucleinInclusions
};

template <unsigned int DIM>
class SeedingRegion
{
public:
    virtual bool is_inside(const Point<DIM> &p) const = 0;
    virtual ~SeedingRegion() = default;
};

template <unsigned int DIM>
class GeneralSeedingRegion : public SeedingRegion<DIM>
{
public:
    bool is_inside(const Point<DIM> &p) const override
    {
        double distance_squared = 0.0;
        for (unsigned int i = 0; i < DIM; ++i)
        {
            double diff = p[i] - _center[i];
            distance_squared += diff * diff;
        }
        return distance_squared <= _radius * _radius;
    }

    GeneralSeedingRegion(const Point<DIM> &center, double radius)
        : _center(center), _radius(radius) {}

private:
    Point<DIM> _center;
    double _radius;
};

class TauInclusions : public SeedingRegion<3>
{
public:
    bool is_inside(const Point<3> &p) const override
    {
        return p[0] >= 63 && p[0] <= 81 &&
               p[1] >= 70 && p[1] <= 90 &&
               p[2] >= 56 && p[2] <= 69;
    }
};

class AmyloidBetaDeposit : public SeedingRegion<3>
{
public:
    bool is_inside(const Point<3> &p) const override
    {
        return (p[0] >= 23 && p[0] <= 82) &&
               ((p[1] >= 22 && p[1] <= 80) || (p[1] >= 100 && p[1] <= 135)) &&
               (p[2] >= 95 && p[2] <= 118);
    }
};

class TDP43Inclusions : public SeedingRegion<3>
{
public:
    bool is_inside(const Point<3> &p) const override
    {
        return ((p[0] >= 23 && p[0] <= 82 &&
                 p[1] >= 48 && p[1] <= 75 &&
                 p[2] >= 85 && p[2] <= 117) ||
                (p[0] >= 63 && p[0] <= 81 &&
                 p[1] >= 80 && p[1] <= 90 &&
                 p[2] >= 44 && p[2] <= 57));
    }
};

class AlphaSynucleinInclusions : public SeedingRegion<3>
{
public:
    bool is_inside(const Point<3> &p) const override
    {
        return p[0] >= 63 && p[0] <= 81 &&
               p[1] >= 75 && p[1] <= 90 &&
               p[2] >= 44 && p[2] <= 57;
    }
};

class SeedingRegionFactory
{
public:
    static std::unique_ptr<SeedingRegion<3>> create(SeedingRegionType type)
    {
        switch (type)
        {
        case SeedingRegionType::TauInclusions:
            return std::make_unique<TauInclusions>();
        case SeedingRegionType::AmyloidBetaDeposit:
            return std::make_unique<AmyloidBetaDeposit>();
        case SeedingRegionType::TDP43Inclusions:
            return std::make_unique<TDP43Inclusions>();
        case SeedingRegionType::AlphaSynucleinInclusions:
            return std::make_unique<AlphaSynucleinInclusions>();
        default:
            throw std::invalid_argument("Unknown seeding region type");
        }
    }
};

#endif // SEEDING_REGIONS_HPP