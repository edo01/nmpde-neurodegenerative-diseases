#ifndef SEEDING_REGIONS_HPP
#define SEEDING_REGIONS_HPP

#include <memory>
#include <vector>
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

template <unsigned int DIM>
class SeedingRegion : public NDProblem<DIM>::InitialConcentration
{
public:
  using CornerPair = std::pair<Point<DIM>, Point<DIM>>;

  double value(const Point<DIM> &p, const unsigned int component = 0) const override;

  static SeedingRegion<DIM> create(SeedingRegionType type, double C_0);

protected:
  SeedingRegion(double C_0, const std::vector<CornerPair> &corners);

private:
  const double _C_0;
  Triangulation<DIM> _region;

  bool is_inside(const Point<DIM> &p) const;
};

#endif // SEEDING_REGIONS_HPP
