#ifndef SEEDING_REGIONS_HPP
#define SEEDING_REGIONS_HPP

#include <deal.II/base/point.h>
#include <array>
#include <vector>
using namespace dealii;

template <unsigned int DIM>
class SeedingRegion
{
public:
    struct Limit
    {
        double min;
        double max;
    };

    bool is_inside(const Point<DIM> &p) const
    {
        for (unsigned int i = 0; i < DIM; ++i)
        {
            bool inside_dimension = false;
            for (const auto &limit : _limits[i])
            {
                if (p[i] >= limit.min && p[i] <= limit.max)
                {
                    inside_dimension = true;
                    break;
                }
            }
            if (!inside_dimension)
            {
                return false;
            }
        }
        return true;
    }

    SeedingRegion(const std::array<std::vector<Limit>, DIM> &limits) : _limits(limits) {}
    SeedingRegion(const Point<DIM> &origin, double ray) : _limits({})
    {
        for (unsigned int i = 0; i < DIM; ++i)
        {
            _limits[i].push_back({origin[i] - ray, origin[i] + ray});
        }
    }
    SeedingRegion() : _limits({}) {}

private:
    std::array<std::vector<Limit>, DIM> _limits;
};


class TauInclusions : public SeedingRegion<3>
{
public:
    TauInclusions() : SeedingRegion<3>({std::vector<Limit>{{63, 81}},
                                        std::vector<Limit>{{70, 90}},
                                        std::vector<Limit>{{56, 69}}}) {}
};

class AmyloidBetaDeposit : public SeedingRegion<3>
{
public:
    AmyloidBetaDeposit() : SeedingRegion<3>({std::vector<Limit>{{23, 82}},
                                             std::vector<Limit>{{22, 80}, {100, 135}},
                                             std::vector<Limit>{{95, 118}}}) {}
};

class TDP43Inclusions : public SeedingRegion<3>
{
public:
    TDP43Inclusions() : SeedingRegion<3>({std::vector<Limit>{{23, 82}, {63, 81}},
                                          std::vector<Limit>{{48, 75}, {80, 90}},
                                          std::vector<Limit>{{85, 117}, {44, 57}}}) {}
};

class AlphaSynucleinInclusions : public SeedingRegion<3>
{
public:
    AlphaSynucleinInclusions() : SeedingRegion<3>({std::vector<Limit>{{63, 81}},
                                                   std::vector<Limit>{{75, 90}},
                                                   std::vector<Limit>{{44, 57}}}) {}
};

#endif // SEEDING_REGIONS_HPP