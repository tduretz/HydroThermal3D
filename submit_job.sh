#!/bin/sh
#$ -cwd
#$ -j y 
#$ -N HT3D
#$ -l hostname=physix90.ipr.univ-rennes1.fr
#$ -M thibault.duretz@univ-rennes1.fr
#$ -m ea
#$ -l mem_available=100G
#$ -l cuda_devices=1
module load lib/cuda/latest 
module load programming/julia/latest 
module load lib/cuda/latest
export CUDA_VISIBLE_DEVICES=0 
julia -O3 --check-bounds=no --color=no --banner=no HydroThermal3D_v5.jl
