#!/bin/bash

# Clean results
./clean_results.sh

# Baseline parameters from the paper
BASELINE_DEXT=1.5
BASELINE_DAXN=3.0
BASELINE_ALPHA=0.6
MESH_FILE="meshes/brain-h3.03D.msh"

# Function to run simulation
run_simulation() {
    local dext=$1
    local daxn=$2
    local alpha=$3
    local seeding_type=$4
    local fiber_type=$5
    local output_dir=$6
    local output_file=$7

    mpirun -n 11 ./build/neuro_disease_3D \
        -e $dext \
        -x $daxn \
        -a $alpha \
        -s $seeding_type \
        -f $fiber_type \
        -m $MESH_FILE \
        -d $output_dir \
        -o $output_file
}

# Run baseline simulation for each analysis
echo "Running baseline simulations..."
run_simulation $BASELINE_DEXT $BASELINE_DAXN $BASELINE_ALPHA 2 2 results/baseline/ baseline

# 1. Diffusion sensitivity analysis (Fig. 8, top right)
echo "Running diffusion sensitivity analysis..."
run_simulation 6.0 $BASELINE_DAXN $BASELINE_ALPHA 2 2 results/diffusion_sensitivity high_diffusion

# 2. Transport sensitivity analysis (Fig. 8, bottom left)
echo "Running transport sensitivity analysis..."
run_simulation $BASELINE_DEXT 24.0 $BASELINE_ALPHA 2 2 results/transport_sensitivity high_transport

# 3. Growth rate sensitivity analysis (Fig. 8, bottom right)
echo "Running growth rate sensitivity analysis..."
run_simulation $BASELINE_DEXT $BASELINE_DAXN 1.2 2 2 results/growth_sensitivity high_growth

# 4. Fiber orientation and seeding region analysis (Fig. 9)
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
        run_simulation $BASELINE_DEXT $BASELINE_DAXN $BASELINE_ALPHA $seeding_type $fiber_type \
            results/fiber_orientation "${name}_${fiber_name}"
    done
done 