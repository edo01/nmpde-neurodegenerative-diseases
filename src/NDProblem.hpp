#ifndef NEURODEGENERATIVE_PROBLEM_HPP
#define NEURODEGENERATIVE_PROBLEM_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_tools.h>

#include <typeinfo>
#include <iostream>

/*
Problem definition:

    1) Fiber Field -> il Diffusion Tensor direi che è costruito automaticamente.
    2) extracellular diffusion
    3) axonal diffusion
    4) reaction coefficient
    5) origin
    6) delta t
    7) T
    8) Mesh

nota che origin non è un parametro del problema, ma è un parametro della
funzione iniziale e del fiber field.
*/
using namespace dealii;

static const Point<3> brain_origin = Point<3>(48.0, 73.0, 60.0);
static const Point<2> square_origin = Point<2>(0.5, 0.5);
static const Point<3> cube_origin = Point<3>(0.5, 0.5, 0.5);

template<int DIM>
class NDProblem
{

   public:
        //static constexpr unsigned int DIM = 2;

        /**
         * Fiber Field represent the field of fibers in the domain.
         * For each point in the domain, the fiber field is a versor.
         * 
         */
        class FiberField : public Function<DIM>
        {
            public:
                virtual void vector_value(const Point<DIM> &p, Vector<double> &values) const override = 0;
                virtual double value(const Point<DIM> &p, const unsigned int component = 0) const override = 0;

        };




        /**
         * DiffusionTensor represent the diffusion tensor in the domain.
         * It represents the diffusion of the concentration in the domain and is
         * defined as:
         * D = d_ext*I + d_axn*n⊗n
         * where d_ext is the extracellular diffusion coefficient, d_axn is the axial
         * diffusion coefficient, I is the identity tensor and n is the fiber field.
         */
        class DiffusionTensor : public TensorFunction<2,DIM>
        {
            public:
                DiffusionTensor(const FiberField &fiber_field, const double d_ext, const double d_axn) 
                : _fiber_field(fiber_field), 
                _identity(unit_symmetric_tensor<DIM>()),
                _d_ext(d_ext),
                _d_axn(d_axn)
                {}


                Tensor<2,DIM> calculate_fiber_tensor(const Point<DIM> &p) const{

                    // calculate the fiber field at the point p
                    Vector<double> fiberV(DIM);
                    _fiber_field.vector_value(p, fiberV);
                    
                    // calculate the tensor product n⊗n
                    Tensor<1,DIM> fiberT_1D;
                    // copy fiberV into a 1D tensor
                    for (unsigned int i = 0; i < DIM; ++i)
                        fiberT_1D[i] = fiberV[i];
                    
                    return outer_product(fiberT_1D, fiberT_1D);
                }

                protected:
                    const FiberField &_fiber_field;
                    const SymmetricTensor<2,DIM> _identity;
                    
                    const double _d_ext; // cm^2/year
                    const double _d_axn; // cm^2/year 
        };


        class WhiteDiffusionTensor : public DiffusionTensor
        {
            public:
                using DiffusionTensor::DiffusionTensor;
                virtual Tensor<2, DIM, double> value(const Point<DIM> &p) const override
                {
                    return DiffusionTensor::_d_ext*DiffusionTensor::_identity + DiffusionTensor::_d_axn*DiffusionTensor::calculate_fiber_tensor(p);
                }
        
        }; 

        class GrayDiffusionTensor : public DiffusionTensor
        {
            public:
                using DiffusionTensor::DiffusionTensor;
                virtual Tensor<2, DIM, double> value(const Point<DIM> &/*p*/) const override
                {
                    return DiffusionTensor::_d_ext*DiffusionTensor::_identity; 
                }
        
        };
        
        /**
         * Defines the initial condition for the concentration field.
         */
        class InitialConcentration : public Function<DIM>
        {
            public:
                virtual double value(const Point<DIM> & p, const unsigned int component = 0) const override = 0;
        };
        
        // export the parameters of the problem in a human readable format
        void export_problem(std::string filename)
        {
            std::ofstream file(filename);
            file << "Mesh file name: " << _mesh_file_name << std::endl;
            file << "Growth factor: " << _alpha << std::endl;
            file << "Extracellular diffusion coefficient: " << _d_ext << std::endl;
            file << "Axonal diffusion coefficient: " << _d_axn << std::endl;
            file << "Initial concentration: " << typeid(_c_initial).name() << std::endl;
            //file << "Diffusion tensor: " << typeid(_diffusion_tensor).name() << std::endl;
            file.close();
        
        }

        
        /**
         * Getters
         */
        const std::string &get_mesh_file_name() const { return _mesh_file_name; }
        double get_alpha() const { return _alpha; }
        double get_d_ext() const { return _d_ext; }
        double get_d_axn() const { return _d_axn; }
        double get_white_gray_ratio() const { return _white_gray_ratio; }
        const InitialConcentration& get_initial_concentration() const { return _c_initial; }
        WhiteDiffusionTensor get_white_diffusion_tensor() const { return _white_diffusion_tensor; }
        GrayDiffusionTensor get_gray_diffusion_tensor() const  { return _gray_diffusion_tensor; }

        /**
         * Constructor
         */
        NDProblem(
            const std::string &mesh_file_name_,
            const double alpha_,
            const double d_ext,
            const double d_axn,
            const InitialConcentration &c_initial_,
            const FiberField &fiber_field_,
            const double white_gray_ratio= 0.9): 
        _mesh_file_name(mesh_file_name_),
        _alpha(alpha_),
        _d_ext(d_ext),
        _d_axn(d_axn),
        _c_initial(c_initial_), 
        _white_diffusion_tensor(fiber_field_, d_ext, d_axn),
        _gray_diffusion_tensor(fiber_field_, d_ext, d_axn),
        _white_gray_ratio(white_gray_ratio)
        {}

    private:

        // Mesh file name.
        const std::string _mesh_file_name;

        // concentration growth rate
        double _alpha; // year^-1

        // Extracellular diffusion coefficient
        double _d_ext; // cm^2/year 

        // Axonal diffusion coefficient
        double _d_axn; // cm^2/year

        // Initial conditions.
        const InitialConcentration& _c_initial;

        WhiteDiffusionTensor _white_diffusion_tensor;

        GrayDiffusionTensor _gray_diffusion_tensor;

        // White matter portion 0...1 
        double _white_gray_ratio;
};

#endif