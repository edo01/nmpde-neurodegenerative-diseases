#ifndef HEAT_NON_LINEAR_HPP
#define HEAT_NON_LINEAR_HPP

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

#include <fstream>
#include <iostream>

using namespace dealii;

class NeurodegenerativeDisease
{
public:


  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  /**
   * Fiber Field represent the field of fibers in the domain.
   * For each point in the domain, the fiber field is a versor.
   *
   * At the moment the fiber field is defined as:
   * [x y z] = [x y z]/sqrt(x^2 + y^2 + z^2) radial field
   * 
   * NOTE ON THE DESIGN: the user should extend this class and define
   * the fiber field as needed.
   */
  class FiberField : public Function<dim>
  {
    public:
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      double distance_from_origin = p.distance(origin);

      // todo: changeme
        for (unsigned int i = 0; i < dim; ++i)
            values[i] = p[i] / (distance_from_origin + 1e-10);
    }
    
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      // Fiber field 
      double distance_from_origin = p.distance(origin);

      return p[component] / (distance_from_origin + 1e-10);
    }

    protected:  
        //const Point<dim> origin = Point<dim>();
        const Point<dim> origin = Point<dim>(48, 73, 60);
        //const Point<dim> origin = Point<dim>();
  };

  /**
   * DiffusionTensor represent the diffusion tensor in the domain.
   * It represents the diffusion of the concentration in the domain and is
   * defined as:
   * D = d_ext*I + d_axn*n⊗n
   * where d_ext is the extracellular diffusion coefficient, d_axn is the axial
   * diffusion coefficient, I is the identity tensor and n is the fiber field.
   */
  class DiffusionTensor : public TensorFunction<2,dim>
  {
    public:
    DiffusionTensor(const FiberField &fiber_field)
      : _fiber_field(fiber_field), _identity(unit_symmetric_tensor<dim>())
    {}

    virtual Tensor<2, dim, double> value(const Point<dim> &p) const override
    {
      Tensor<2, dim> diffusion_tensor;

      // calculate the fiber field at the point p
      Vector<double> fiberV(dim);
      _fiber_field.vector_value(p, fiberV);
      
      // calculate the tensor product n⊗n
      Tensor<1,dim> fiberT_1D;
      // copy fiberV into a 1D tensor
      for (unsigned int i = 0; i < dim; ++i)
        fiberT_1D[i] = fiberV[i];
      
      Tensor<2,dim> fiber_tensor = outer_product(fiberT_1D, fiberT_1D);

      diffusion_tensor = d_ext*_identity + d_axn*fiber_tensor;

      return diffusion_tensor;
    }

    private:
      const FiberField &_fiber_field;
      const SymmetricTensor<2,dim> _identity;
      
    //   const double d_ext = 1.5; // cm^2/year
    //   const double d_axn = 3.0; // cm^2/year 

    const double d_ext = 2;
    const double d_axn = 5;
  };
  

  /**
   * Defines the initial condition for the concentration field.
   */
  class FunctionIC : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & p,
          const unsigned int /*component*/ = 0) const override
    {
      if (p.distance(Point<dim>(40, 73, 60)) < ray)
        return C_0;
      else
        return 0.0;

    //     auto distance = p.distance(Point<dim>(20, 20));
    //       if (distance < ray)
    //     return C_0 ;
    //   else
    //     return 0.0;
    }
  
  protected:
    double C_0 = 0.4; // initial concentration
    double ray = 30; // radius of the initial condition

  };


  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  NeurodegenerativeDisease(const std::string  &mesh_file_name_,
                const unsigned int &r_,
                const double       &T_,
                const double       &deltat_,
                const FiberField &fiber_field_)
    :
      mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
      //THIS WILL BE CHANGED AND THE FIBER FIELD WILL BE PASSED IN THE SENSIBILITY ANALYSIS
    , diffusion_tensor(fiber_field_) // initialize the diffusion tensor.
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

protected:
  // Assemble the tangent problem.
  void
  assemble_system();

  // Solve the linear system associated to the tangent problem.
  void
  solve_linear_system();

  // Solve the problem for one time step using Newton's method.
  void
  solve_newton();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;


  // SENSIBILITY ANALYSIS /////////////////////////////////////////////////////

  double alpha = 3; // year^-1 // concentration growth rate

  // Initial conditions.
  FunctionIC c_initial;

  // Problem definition. ///////////////////////////////////////////////////////

  // Diffusion coefficient
  DiffusionTensor diffusion_tensor;

  // Initial conditions.
  //FunctionU0 u_0;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector residual_vector;

  // Increment of the solution between Newton iterations.
  TrilinosWrappers::MPI::Vector delta_owned;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // System solution at previous time step.
  TrilinosWrappers::MPI::Vector solution_old;

};

#endif
