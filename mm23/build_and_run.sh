#!/usr/bin/env sh
set -e

# Ajuste o caminho do gcc-8 caso necessário (algumas instalações exigem --ccbin=gcc-8)
# Se não tiver gcc-8 no PATH, remova o -ccbin e deixe o host default que o nvcc aceitar.
NVCC_CC="-ccbin gcc-8"

echo "Compilando com nvcc (host gcc-8), O3 e OpenMP..."
nvcc mm.cu -O3 -Xcompiler -fopenmp $NVCC_CC -o mm

WIDTH=${1:-2000}
THREADS=${OMP_NUM_THREADS:-0}

echo "Executando seq..."
./mm --variant seq --width $WIDTH

echo "Executando omp..."
if [ "$THREADS" -gt 0 ] 2>/dev/null; then
  OMP_NUM_THREADS=$THREADS ./mm --variant omp --width $WIDTH --threads $THREADS
else
  ./mm --variant omp --width $WIDTH
fi

echo "Executando cuda-naive..."
./mm --variant cuda-naive --width $WIDTH

echo "Executando cuda-tiled..."
./mm --variant cuda-tiled --width $WIDTH

echo "OK. Agora rode ./profile_gpu.sh $WIDTH para coletar métricas do nvprof."
