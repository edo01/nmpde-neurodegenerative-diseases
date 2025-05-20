#ifndef ANISOTROPIC_EVALUATOR_HPP
#define ANISOTROPIC_EVALUATOR_HPP

#include <vector>

template <unsigned int DIM>
class AnisotropicEvaluator
{
    public:

        AnisotropicEvaluator(const std::string mesh_file_name_, const Triangulation<DIM> &mesh_serial_, const parallel::fullydistributed::Triangulation<DIM> &mesh)
            : mesh_file_name(mesh_file_name_),
              mesh_serial(mesh_serial_),
              mesh(mesh),
              mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
              mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
              pcout(std::cout, mpi_rank == 0)
              {}
      
        void load_cells_domain();

        /**
         * Vector which maps every triangulation cell to a part of the brain denoted by a value:
         * 0 if it is in the white portion or 1 if it is in the gray portion.
         * Calculated previously in Matlab, loaded from a file.
         */
        std::vector<int> cells_colormap;

    private:

        /**
         * Save the centers of the cells to a file, to later compute the colormap/domain of the cells in MataLab
         */
        void save_cells_centers_to_file(const std::string& filename) const;

        /**
         * Save a integer vector to a file
         */
        void saveVectorToFile(std::vector<int>& vec, const std::string& filename);

        /**
         * Load a integer vector from a file
         */
        bool loadVectorFromFile(std::vector<int>& vec, const std::string& filename);

        /**
         * Print the bounding box of the mesh
         */
        void print_bounding_box();

        // mesh references
        const std::string mesh_file_name;
        const Triangulation<DIM> &mesh_serial;
        const parallel::fullydistributed::Triangulation<DIM> &mesh;

        // MPI 
        const unsigned int mpi_rank;
        const unsigned int mpi_size;
        ConditionalOStream pcout;

};


template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::saveVectorToFile(std::vector<int>& vec, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(int));
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
void AnisotropicEvaluator<DIM>::print_bounding_box()
{
    
    pcout << "-----------------------------------------------" << std::endl;

    pcout << "  Mesh file informations:" << std::endl<<std::endl;
    pcout << "  Bounding box sides lenght:" << std::endl;

    auto box = GridTools::compute_bounding_box(mesh_serial);

    Point<DIM> box_center = box.center();
      
    static const char labels[3] = {'x', 'y', 'z'}; 
    for(unsigned i=0; i<DIM; i++){
        pcout << "  " << labels[i] << ": " << box.side_length(i) << std::endl;
    }

    pcout << "  Center:  " << box_center << std::endl ;
    pcout << "  Box volume:  " << box.volume()<< std::endl;


    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;

}


template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::save_cells_centers_to_file(const std::string& filename) const
{
    std::vector<Point<DIM>> vertices;
    for (const auto& cell : mesh_serial){
        // Choose the center 
        vertices.push_back(cell.center());
    }
    std::ofstream outfile(filename);
    for (const auto& vertex : vertices) {
        for (unsigned int d = 0; d < DIM; ++d) {
            outfile << vertex[d] << " ";
        }
        outfile << std::endl;
    }
    outfile.close();
    
}

/**
 * Since the brain is divided into two parts, white and gray matter,
 * we need to tag the position of every cell with respect
 * to the white and gray partion of the brain. A boolean (0-white & 1-gray) 
 * vector will be used to evaluate the diffusion tensor 
 * on the current cell with repesct to the color type.
 *  
*/
template<unsigned int DIM>
void AnisotropicEvaluator<DIM>::load_cells_domain(){

    // Print the bounding box of the mesh
    print_bounding_box();

    // Number of active cells in the triangulation
    unsigned n_cells = mesh_serial.n_global_active_cells();

    // Vector to store the domain of every cell, 0 for white, 1 for gray
    // Not boolean to avoid probleams reading from file
    cells_colormap = std::vector<int>(n_cells, 0);

    std::string file_name_base = mesh_file_name;
    // Remove .msh extension from the file name base if present
    if (file_name_base.size() > 4 && file_name_base.substr(file_name_base.size() - 4) == ".msh") {
        file_name_base = file_name_base.substr(0, file_name_base.size() - 4);
    }
    const std::string file_name_cells_domain = file_name_base + ".cells_colormap"; 


    // Tries to load existing file
    if(loadVectorFromFile(cells_colormap, file_name_cells_domain)){
        if(mpi_rank == 0)
        std::cout << "Cells color domain file found at " + file_name_cells_domain + "\n";
        return;
    }
    
    // File containing the cells domain does not exist, we need to compute it  
    // The center of each cell is saved on the file. Compute the .cells_colormap file using the matlab script
    const std::string file_name_cells_centers = file_name_base + ".txt"; 
    save_cells_centers_to_file(file_name_cells_centers);

    std::cout << "[AnisotropicEvaluator] Cell domain file not found at location: " << file_name_cells_domain << std::endl;
    std::cout << "[AnisotropicEvaluator] Deactivate anysotropic mode or create the file running the Matlab script on the file : " << file_name_cells_centers << std::endl;
    exit(EXIT_FAILURE);
 
}



#endif // ANISOTROPIC_EVALUATOR_HPP