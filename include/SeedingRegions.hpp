#ifndef SEEDING_REGIONS_HPP
#define SEEDING_REGIONS_HPP

#include <memory>
#include <cmath>

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include "NDProblem.hpp"

using namespace dealii;

enum class SeedingRegionType
{
  AlphaSynuclein = 0,
  AmyloidBeta = 1,
  Tau = 2,
  TDP43 = 3
};

class SeedingRegion : public NDProblem<3>::InitialConcentration
{
public:
  using CornerPair = std::pair<Point<3>, Point<3>>;

  double value(const Point<3> &p, const unsigned int component = 0) const override;

  static SeedingRegion create(SeedingRegionType type, double C_0);

protected:
  SeedingRegion(double C_0, const std::vector<CornerPair> &corners);

private:
  const double _C_0;
  Triangulation<3> _region;

  bool is_inside(const Point<3> &p) const;
};

#endif // SEEDING_REGIONS_HPP
