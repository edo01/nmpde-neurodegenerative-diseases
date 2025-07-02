### Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
or make sure that the dealii library is installed and the `DEAL_II_DIR` environment variable is set to the correct path.
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ ./executable-name
```

Three executables are created: neuro_disease_1D, neuro_disease_2D, and neuro_disease_3D, one for each dimension.
They come with a set of default parameters and they should run out of the box without any additional arguments, provided that the mesh files have been placed in the correct folder.
The parameters can be changed through command line arguments or by editing the NDConfig object in the source code of each main file. For the full list of parameters, which include mesh file, equation parameters, output options, and more, refer to 'src/NDConfig.cpp'.

The meshes are available at the following link: "https://1drv.ms/f/c/657b08dae1e39d8b/Ei2Ap_XudaFKiIadBoTqekIBgCvOeJMNKVrjh1MtcAq4xA?e=nj7rCu".
They should be placed inside the 'meshes' folder.
The most important meshes are `slice_generated.msh` for the 2D sagittal view simulation and `brain-h3.03D.msh` for the 3D simulation.

We also provide a series of paraview states that can be used to visualize the results of the simulation. For instance, you can use the provided `sagittal.pvsm` file to visualize the results of the 2D sagittal view simulation. 

The executables support MPI parallelization, so you can run them with `mpirun` or `mpiexec` if you have a suitable MPI environment set up.