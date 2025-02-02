#include "SeedingRegions.hpp"

double SeedingRegion::value(const Point<3> &p, const unsigned int /* component */) const
{
  return is_inside(p) ? _C_0 : 0.0;
}

SeedingRegion::SeedingRegion(double C_0, const std::vector<CornerPair> &corners)
    : _C_0(C_0), _region()
{
  std::vector<Triangulation<3>> regions;
  for (const auto &corner_pair : corners)
  {
    regions.emplace_back();
    GridGenerator::hyper_rectangle(regions.back(), corner_pair.first, corner_pair.second);
    GridGenerator::merge_triangulations(_region, regions.back(), _region);
  }
}

bool SeedingRegion::is_inside(const Point<3> &p) const
{
  auto result = GridTools::find_active_cell_around_point(_region, p);
  return result.state() == IteratorState::valid;
}

SeedingRegion SeedingRegion::create(SeedingRegionType type, double C_0)
{
  using std::make_pair;

  switch (type)
  {
    case SeedingRegionType::AlphaSynuclein:
      return SeedingRegion(C_0, {make_pair(Point<3>(63, 75, 44), Point<3>(81, 80, 57))});
    case SeedingRegionType::AmyloidBeta:
      return SeedingRegion(C_0, {make_pair(Point<3>(23, 22, 95), Point<3>(82, 80, 118)),
                                 make_pair(Point<3>(23, 100, 95), Point<3>(82, 135, 118))});
    case SeedingRegionType::Tau:
      return SeedingRegion(C_0, {make_pair(Point<3>(23, 48, 85), Point<3>(82, 75, 117))});
    case SeedingRegionType::TDP43:
      return SeedingRegion(C_0, {make_pair(Point<3>(23, 48, 85), Point<3>(82, 75, 117)),
                                 make_pair(Point<3>(63, 80, 44), Point<3>(81, 90, 57))});
    default:
      throw std::invalid_argument("Unknown seeding region type");
  }
}
