#!/usr/bin/env sh
set -e

# GTX 1030 (Pascal GP108) = SM 6.1
ARCH=${ARCH:-sm_61}

echo "Compilando com nvcc (host gcc-5), -O3, OpenMP para ${ARCH}..."
nvcc mm.cu -O3 \
  -D_GLIBCXX_USE_FLOAT128=0 -D__STRICT_ANSI__ \
  -Xcompiler "-fopenmp -D_GLIBCXX_USE_FLOAT128=0 -D__STRICT_ANSI__" \
  -gencode arch=compute_61,code=sm_61 \
  -o mm

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
