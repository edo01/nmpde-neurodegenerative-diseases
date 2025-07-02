#ifndef WHITEGRAYPARTITION_HPP
#define WHITEGRAYPARTITION_HPP
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/point.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/filtered_iterator.h>



namespace WhiteGrayPartition
{

    template <int DIM>
    void
    set_white_gray_material(const Triangulation<DIM> &serial_triangulation,
                         const Triangulation<DIM> &distributed_triangulation,
                         const double distance_threshold)
    {
        std::vector<Point<DIM>> boundary_cell_centers;

        // 1. Collect centers of cells that are on the boundary.
        // Here we use the serial mesh, since we want each process to gather the center coordinates of all
        // boundary cells, regardless of their ownership in the distributed triangulation.
        for (const auto &cell : serial_triangulation.active_cell_iterators() | IteratorFilters::AtBoundary())
        {
            boundary_cell_centers.push_back(cell->center());
        }

        if (boundary_cell_centers.empty())
        {
            std::cout << "Warning: No boundary cells found." << std::endl;
            return;
        }

        // 2. Each process loops through its locally owned active cells
        // and flags them if they are near any of the boundary cell centers.
        for (const auto &cell : distributed_triangulation.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
        {
            const Point<DIM> &current_cell_center = cell->center();
            bool is_gray = false;
            for (const auto &boundary_center : boundary_cell_centers)
            {
                if (current_cell_center.distance(boundary_center) < distance_threshold)
                {
                    cell->set_material_id(1); // Set the material ID to 0 for gray matter
                    is_gray = true;
                    break;
                }
            }
            if (!is_gray) cell->set_material_id(0); 
        }
        return;
    }

    template<int DIM>
    void
    write_partition_to_pvtu(const parallel::fullydistributed::Triangulation<DIM> &triangulation,
                            const std::string &output_directory,
                            const std::string &output_filename)
    {
        // collect the partition data into a vector
        Vector<float> partition_data(triangulation.n_active_cells());
        partition_data = 0.0;

        for (const auto &cell : triangulation.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
        {
            unsigned cell_index = cell->active_cell_index();
            partition_data[cell_index] = cell->material_id();
        }

        // Save the vector to a file
        DataOut<DIM> data_out;
        data_out.attach_triangulation(triangulation);
        data_out.add_data_vector(partition_data, "is_cell_gray");
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(output_directory, output_filename + "_white_gray_partition", 0, MPI_COMM_WORLD, 0);
    }

} // namespace WhiteGrayPartition

#endif // WHITEGRAYPARTITION_HPP