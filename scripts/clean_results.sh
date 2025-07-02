#!/bin/bash

# Script to clean up results directories while preserving their structure

# Check if results directory exists
if [ ! -d "./results" ]; then
    echo "No results directory found."
    exit 1
fi

# List of subdirectories to clean
subdirs=("baseline" "diffusion_sensitivity" "transport_sensitivity" "growth_sensitivity" "fiber_orientation")

# Clean each subdirectory
for subdir in "${subdirs[@]}"; do
    if [ -d "./results/$subdir" ]; then
        echo "Cleaning $subdir..."
        # Remove all files and subdirectories in the subdirectory
        rm -rf "./results/$subdir"/*
        echo "✓ $subdir cleaned"
    fi
done

echo -e "\nResults directories have been cleaned while preserving their structure." 