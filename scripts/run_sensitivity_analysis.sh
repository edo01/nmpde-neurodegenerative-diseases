#!/bin/bash
# Baseline parameters from the paper
BASELINE_DEXT=1.5
BASELINE_DAXN=3.0
BASELINE_ALPHA=0.6

MESH_FILE="meshes/slice_generated.msh"

# Function to run simulation
run_simulation() {
    local dext=$1
    local daxn=$2
    local alpha=$3
    local seeding_type=$4
    local fiber_type=$5
    local output_dir=$6

    n=$(nproc)
    mpirun -n $n ./build/neuro_disease_2D \
        -e $dext \
        -x $daxn \
        -a $alpha \
        -s $seeding_type \
        -f $fiber_type \
        -d $output_dir \
        -m $MESH_FILE 
}

# Run baseline simulation for each analysis
echo "Running baseline simulations..."
run_simulation $BASELINE_DEXT $BASELINE_DAXN $BASELINE_ALPHA 2 2 results/baseline/

# 1. Extracellular diffusion sensitivity analysis
echo "Running diffusion sensitivity analysis..."
run_simulation 6.0 $BASELINE_DAXN $BASELINE_ALPHA 2 2 results/extracellular_diffusion/ 

# 2. Axonal diffusion sensitivity analysis 
echo "Running axonal diffusion sensitivity analysis..."
run_simulation $BASELINE_DEXT 24.0 $BASELINE_ALPHA 2 2 results/axonal_diffusion/ 

# 3. Growth rate sensitivity analysis
echo "Running growth rate sensitivity analysis..."
run_simulation $BASELINE_DEXT $BASELINE_DAXN 1.2 2 2 results/growth_rate/ 

# 4. Fiber orientation and seeding region analysis
echo "Running fiber orientation and seeding region analysis..."

# Test each seeding region with each fiber orientation
for seeding_type in 0 1 2 3; do  # Alpha-synuclein, Amyloid-beta, Tau, TDP-43
    for fiber_type in 0 1 2; do   # Radial, Circumferential, Axon-based
        case $seeding_type in
            0) name="alpha_synuclein" ;;
            1) name="amyloid_beta" ;;
            2) name="tau" ;;
            3) name="tdp43" ;;
        esac
        
        case $fiber_type in
            0) fiber_name="radial" ;;
            1) fiber_name="circumferential" ;;
            2) fiber_name="axon_based" ;;
        esac
        
        # Use baseline parameters for all fiber orientation tests
        echo "Running $name $fiber_name"
        run_simulation $BASELINE_DEXT $BASELINE_DAXN $BASELINE_ALPHA $seeding_type $fiber_type \
            results/fiber_orientation/$name/$fiber_name/
    done
    echo "Running anisotropic diffusion analysis"
    run_simulation $BASELINE_DEXT 0.0 $BASELINE_ALPHA $seeding_type 2 \
        results/fiber_orientation/$name/anisotropic/
done 