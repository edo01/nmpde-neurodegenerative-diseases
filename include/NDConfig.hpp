#ifndef NDCONFIG_HPP
#define NDCONFIG_HPP

#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include "SeedingRegions.hpp"
#include "FiberFields.hpp"

struct NDConfig
{
  unsigned int dim;
  double T;
  double alpha;
  double deltat;
  unsigned int degree;
  double d_ext;
  double d_axn;
  double C_0;
  std::string mesh;
  std::string output_filename = "output";
  std::string output_dir = "./";
  SeedingRegionType seeding_region_type;
  FiberFieldType fiber_field_type;
  double gray_matter_distance_threshold = 0.0; // all white matter by default

  void parse(int argc, char *argv[]);
};

#endif //NDCONFIG_HPP