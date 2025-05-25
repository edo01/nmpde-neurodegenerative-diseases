#include "SeedingRegions.hpp"
#include <deal.II/base/exceptions.h> // For AssertThrow

// Template all method definitions

template <unsigned int DIM>
double SeedingRegion<DIM>::value(const Point<DIM> &p, const unsigned int /* component */) const
{
  return is_inside(p) ? _C_0 : 0.0; // Or EPSILON from InitialConditions.hpp if preferred
}

template <unsigned int DIM>
SeedingRegion<DIM>::SeedingRegion(double C_0, const std::vector<typename SeedingRegion<DIM>::CornerPair> &corners)
    : _C_0(C_0), _region()
{
  // AssertThrow(!corners.empty(), dealii::ExcMessage("SeedingRegion requires at least one corner pair."));
  if (corners.empty()){
    // If no corners are provided, the region remains empty, is_inside will always be false.
    // This means value() will return 0.0 (or EPSILON).
    // This can be valid if a SeedingRegionType might correspond to no actual seeding.
    // Alternatively, throw an exception as originally commented.
    return;
  }

  // Create a temporary Triangulation for merging
  Triangulation<DIM> temp_merge_tria;
  bool first_region = true;

  for (const auto &corner_pair : corners)
  {
    Triangulation<DIM> current_hyper_rectangle;
    GridGenerator::hyper_rectangle(current_hyper_rectangle, corner_pair.first, corner_pair.second);
    
    if (first_region)
    {
      _region.copy_triangulation(current_hyper_rectangle);
      first_region = false;
    }
    else
    {
      // GridGenerator::merge_triangulations requires the first argument to be the one that grows
      // and the third (destination) to be a new one or the same as the first.
      // To avoid issues, we merge into a new temporary tria and then copy.
      Triangulation<DIM> merged_tria;
      GridGenerator::merge_triangulations(_region, current_hyper_rectangle, merged_tria);
      _region.clear();
      _region.copy_triangulation(merged_tria);
    }
  }
}

template <unsigned int DIM>
bool SeedingRegion<DIM>::is_inside(const Point<DIM> &p) const
{
  if (_region.n_active_cells() == 0) // No region defined
    return false;
  // For robustness, check if point is within the bounding box of the triangulation first
  // This can avoid issues with find_active_cell_around_point if the point is far outside
  // BoundingBox<DIM> bbox = GridTools::compute_bounding_box(_region);
  // if (!bbox.contains_point(p))
  //    return false;

  // If the point is very close to a boundary, find_active_cell_around_point might fail
  // or behave unexpectedly. This is a general challenge with point location.
  auto result = GridTools::find_active_cell_around_point(_region, p);
  return result.state() == IteratorState::valid;
}

template <unsigned int DIM>
SeedingRegion<DIM> SeedingRegion<DIM>::create(SeedingRegionType type, double C_0)
{
  using std::make_pair;
  // Typedef for clarity
  using RegionCornerPair = typename SeedingRegion<DIM>::CornerPair;
  std::vector<RegionCornerPair> corners;

  if constexpr (DIM == 3)
  {
    // Original 3D coordinates
    switch (type)
    {
      case SeedingRegionType::AlphaSynuclein:
        corners.push_back(make_pair(Point<3>(63, 75, 44), Point<3>(81, 80, 57)));
        break;
      case SeedingRegionType::AmyloidBeta:
        corners.push_back(make_pair(Point<3>(23, 22, 95), Point<3>(82, 80, 118)));
        corners.push_back(make_pair(Point<3>(23, 100, 95), Point<3>(82, 135, 118)));
        break;
      case SeedingRegionType::Tau:
        corners.push_back(make_pair(Point<3>(70, 77, 50), Point<3>(90, 85, 60)));
        break;
      case SeedingRegionType::TDP43:
        corners.push_back(make_pair(Point<3>(23, 48, 85), Point<3>(82, 75, 117)));
        corners.push_back(make_pair(Point<3>(63, 80, 44), Point<3>(81, 90, 57)));
        break;
      default:
        throw std::invalid_argument("Unknown seeding region type for 3D");
    }
  }
  else if constexpr (DIM == 2)
  {
    // Placeholder 2D coordinates - YOU MUST REPLACE THESE
    // These are just examples and likely not meaningful for your mesh.
    // The sagittal_origin in NDAnalysis2D.cpp is (70, 73)
    // The seeding_center was (87, 58)
    switch (type)
    {
      case SeedingRegionType::AlphaSynuclein: // Example: A small box around old seeding_center
        corners.push_back(make_pair(Point<2>(70, 42), Point<2>(90, 52)));
        break;
      case SeedingRegionType::AmyloidBeta:   // Example: Two boxes
        corners.push_back(make_pair(Point<2>(5, 100), Point<2>(70, 120)));
        corners.push_back(make_pair(Point<2>(100, 70), Point<2>(140, 100)));
        break;
      case SeedingRegionType::Tau:            // Example: A different box
        corners.push_back(make_pair(Point<2>(90, 60), Point<2>(98, 65)));
        break;
      case SeedingRegionType::TDP43:          // Example: Yet another box
        corners.push_back(make_pair(Point<2>(80, 42), Point<2>(90, 47)));
        corners.push_back(make_pair(Point<2>(63, 90), Point<2>(76, 115)));
        break;
      default:
        throw std::invalid_argument("Unknown seeding region type for 2D");
    }
  }
  else
  {
    // Or handle other dimensions if necessary, or throw.
    static_assert(DIM == 2 || DIM == 3, "SeedingRegion::create is only implemented for DIM 2 or 3.");
  }
  return SeedingRegion<DIM>(C_0, corners);
}

// Explicit instantiations for DIM = 2 and DIM = 3
template class SeedingRegion<2>;
template class SeedingRegion<3>;
