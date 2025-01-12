cmake_minimum_required(VERSION 3.12.0)
project(NeuroDisease LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Locate Boost.
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add useful compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

# Include directories
include_directories(include)

# Create a static library for the common sources
add_library(common_lib STATIC src/FiberFields.cpp src/NDConfig.cpp src/SeedingRegions.cpp)
deal_ii_setup_target(common_lib)

# Define the neuro_disease_1D executable
add_executable(neuro_disease_1D src/NDAnalysis1D.cpp)
target_link_libraries(neuro_disease_1D common_lib)
deal_ii_setup_target(neuro_disease_1D)

# Define the neuro_disease_2D executable
add_executable(neuro_disease_2D src/NDAnalysis2D.cpp)
target_link_libraries(neuro_disease_2D common_lib)
deal_ii_setup_target(neuro_disease_2D)

# Define the neuro_disease_3D executable
add_executable(neuro_disease_3D src/NDAnalysis3D.cpp)
target_link_libraries(neuro_disease_3D common_lib)
deal_ii_setup_target(neuro_disease_3D)



