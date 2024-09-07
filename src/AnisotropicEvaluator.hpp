#ifndef ANISOTROPIC_EVALUATOR_HPP
#define ANISOTROPIC_EVALUATOR_HPP

#include <vector>

template <unsigned int DIM>
class AnisotropicEvaluator
{
    public:

        AnisotropicEvaluator(NDProblem<DIM> &problem_, const Triangulation<DIM> &mesh_serial_, const parallel::fullydistributed::Triangulation<DIM> &mesh, const int mpi_rank, const int mpi_size, ConditionalOStream &pcout)
            : problem(problem_), mesh_serial(mesh_serial_), mesh(mesh), mpi_rank(mpi_rank), mpi_size(mpi_size), pcout(pcout)
        {}
            

        void compute_cells_domain();

        // Vector which maps every triangulation cell to a part of the brain denoted by a value:
        // 0 if it is in the white portion or 1 if it is in the gray portion. @TODO: could be boolean but some problems with file storing and loading
        std::vector<int> cells_domain;

    private:

        void printLoadingBar(int current, int total, int barLength = 50);
        void saveVectorToFile(std::vector<int>& vec, const std::string& filename);
        bool loadVectorFromFile(std::vector<int>& vec, const std::string& filename);

        void compute_bounding_box();

        // Problem and mesh references
        const NDProblem<DIM> &problem;
        const Triangulation<DIM> &mesh_serial;
        const parallel::fullydistributed::Triangulation<DIM> &mesh;

        // Center of the mesh box
        Point<DIM> box_center;

        // MPI rank and size
        const unsigned int mpi_rank;
        const unsigned int mpi_size;
        ConditionalOStream &pcout;

};

template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::printLoadingBar(int current, int total, int barLength) {
    float progress = (float)current / total;
    int pos = (int)(barLength * progress);

    std::cout << "[";
    for (int i = 0; i < barLength; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::saveVectorToFile(std::vector<int>& vec, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
    outfile.close();
}

// Fix messages print in parallel
template<unsigned int DIM>
bool AnisotropicEvaluator<DIM>::loadVectorFromFile(std::vector<int>& vec, const std::string& filename) {
    if(mpi_rank == 0)
      std::cout<< "Trying to read quadrature points domain file '" << filename << "'" <<  std::endl;
    std::ifstream infile(filename, std::ios::in | std::ios::binary);
    if (!infile) {
        if(mpi_rank == 0)
          std::cout<< "Failed to open file\n";
        return false;
    }

    infile.seekg(0, std::ios::end);
    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    vec.resize(size / sizeof(int));
    infile.read(reinterpret_cast<char*>(vec.data()), size);

    // Check if the read operation was successful
    if (infile.fail()) {
        std::cout<< "Read operation failed\n";
        return false;
    }

    infile.close();
    return true;
}

template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::compute_bounding_box()
{
    
    pcout << "-----------------------------------------------" << std::endl;

    pcout << "  Mesh file informations:" << std::endl<<std::endl;
    pcout << "  Bounding box sides lenght:" << std::endl;

    auto box = GridTools::compute_bounding_box(mesh_serial);

    box_center = box.center();
      
    static const char labels[3] = {'x', 'y', 'z'}; 
    for(unsigned i=0; i<DIM; i++){
        pcout << "  " << labels[i] << ": " << box.side_length(i) << std::endl;
    }

    pcout << "  Center:  " << box_center << std::endl ;
    pcout << "  Box volume:  " << box.volume()<< std::endl;


    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;

}

/**
 * Since the brain is divided into two parts, white and gray matter,
 * we compute the position of every cell with respect
 * to the white and gray partion of the brain, and save the resulting
 * boolean vector. This will be used to evaluate the diffusion tensor 
 * on the current cell with repesct to the color type.
 *  
*/
template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::compute_cells_domain(){

    // Compute the bounding box of the mesh
    compute_bounding_box();

    // Number of active cells in the triangulation
    unsigned n_cells = mesh_serial.n_global_active_cells();

    // Vector to store the domain of every cell, 0 for white, 1 for gray
    cells_domain = std::vector<int>(n_cells, 0);

    // @TODO: Assign custom name file based on mesh file name 
    const std::string file_name = problem.get_mesh_file_name() + ".cells_domain"; 


    // Tries to load existing file
    if(loadVectorFromFile(cells_domain, file_name)){
        if(mpi_rank == 0)
        std::cout << "Cells color domain file found at " + file_name + "\n";
        return;
    }

    if(mpi_rank == 0) 
        std::cout << "Computing cells color domain, it could take a while.\n";


    // Retrieve all vertices on the boundary 
    std::map<unsigned int, Point<DIM>> boundary_vertices = GridTools::get_all_vertices_at_boundary(mesh_serial);
    
    // Retrieve all vertices from the triangulation 
    std::vector<Point<DIM>> triangulation_vertices=mesh_serial.get_vertices();
    
    // Create a vector to mark every triangulation vertix if they are on boundary.
    // It is needed to calculate the closest point with GridTools::find_closest_vertex, 
    // but limited on the boundary. 
    std::vector<bool> triangulation_boundary_mask = std::vector<bool>(triangulation_vertices.size(), false);
    for(const auto [key, value] : boundary_vertices){
        triangulation_boundary_mask[key] = true;
    }

    // Scale coefficent describing the ratio between white and gray matter.
    double white_coeff = problem.get_white_gray_ratio();

    // Current checked checked idx.
    unsigned checked_cells = 0;

    // MPI distribution parameters
    int process_cells = n_cells / mpi_size;
    int remainder = n_cells % mpi_size;
    int start = process_cells * mpi_rank;
    int end = start + process_cells;
    if(mpi_rank == mpi_size - 1){
        end += remainder;
    }
    //

    // Iteration over the cells is distributed among the processes
    int i = 0;
    for (const auto& cell : mesh_serial){
        // @TODO: Could refactor with the iterator
        if(i >= end)
        break;
        else if(i >= start){
        types::global_cell_index global_cell_index = cell.global_active_cell_index();

        // Choosing randomly one of the vertices of the cell, 
        // but we could be more precise calculating its center. 
        Point<DIM> cell_point = cell.vertex(0);


        // VERY COMPUTING INTENSIVE 
        // Find the closest vertex point, but only on the boundary 
        types::global_vertex_index closest_boundary_id = GridTools::find_closest_vertex(mesh_serial, cell_point, triangulation_boundary_mask);
        Point<DIM> closest_boundary_point = triangulation_vertices[closest_boundary_id];
        // -------------------------

        // If the vertex point is nearest to the center of the mesh box than its closest vertex on the white boundary,
        // it is in the white portion. 
        if(cell_point.distance(box_center) < white_coeff*closest_boundary_point.distance(box_center))
            cells_domain[global_cell_index]=0;
        else
            cells_domain[global_cell_index]=1;

        checked_cells ++;
        }

        if(mpi_rank == 0)
        printLoadingBar(checked_cells, end-start);

        i++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Gather the results from all processes
    {
        std::vector<int> counts(mpi_size-1, process_cells);
        counts.push_back(process_cells + remainder);
        std::vector<int> displacements(mpi_size);
        for(size_t i = 0; i < mpi_size; i++){
        displacements[i] = process_cells * i;
        }
        MPI_Gatherv(cells_domain.data() + start, end-start, MPI_INT, cells_domain.data(), counts.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Just a check
    if(mpi_rank == 0){
        int count = 0;
        for(auto& cell : cells_domain){
        count += cell == 0 ? 0 : 1;
        }
        std::cout << "Found " << count << " gray cells\n";
    }

    // Save the vector to file
    if(mpi_rank == 0)
        saveVectorToFile(cells_domain, file_name);

    MPI_Barrier(MPI_COMM_WORLD);
}



#endif // ANISOTROPIC_EVALUATOR_HPP