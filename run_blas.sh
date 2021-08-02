#!/bin/bash
#SBATCH --job-name=blas
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --time=0:10:0
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=z19

module restore /etc/cray-pe.d/PrgEnv-cray

export OMP_NUM_THREADS=1

echo " "
echo "Cray Libsci"
echo " "

srun -n 1 ./blas_cray -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_cray -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_cray -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_cray -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_cray -d 10000 -r 100 -b dgemv

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/z19/z19/adrianj/blas-benchmark

echo " "
echo "AMD BLIS"
echo " "

srun -n 1 ./blas_blis -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_blis -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_blis -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_blis -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_blis -d 10000 -r 100 -b dgemv


module use /work/y07/shared/archer2-modules/modulefiles-cse-dev
module load mkl/19.0-117

echo " "
echo "Intel MKL"
echo " "


srun -n 1 ./blas_mkl -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_mkl -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_mkl -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_mkl -d 10000 -r 100 -b dgemv
srun -n 1 ./blas_mkl -d 10000 -r 100 -b dgemv


