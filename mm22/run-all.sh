#!/bin/bash
# ============================================================
# Script: run_all.sh
# Objetivo: compilar e executar todas as versões do mm.c
# ============================================================

WIDTH=2000
EXEC=./mm
SRC=mm.c
OUT=results.txt
GPU_COMPILER=clang    # ou nvc, ou gcc (dependendo do seu setup)
GPU_FLAGS="-O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda"
CPU_FLAGS="-O3 -fopenmp"

echo "================== MATRIX MULTIPLICATION BENCHMARK ==================" > $OUT
echo "Matrix size: $WIDTH x $WIDTH" >> $OUT
date >> $OUT
echo "" >> $OUT

# ============================================================
# 1. COMPILAÇÃO PARA CPU E GPU
# ============================================================
echo "[Compilando para CPU...]"
gcc $CPU_FLAGS $SRC -o mm_cpu

echo "[Compilando para GPU...]"
$GPU_COMPILER $GPU_FLAGS $SRC -o mm_gpu

# ============================================================
# 2. EXECUÇÃO SEQUENCIAL (SEM OpenMP)
# ============================================================
echo "Executando versão sequencial..."
./mm_cpu -m seq -w $WIDTH | tee -a $OUT
echo "" >> $OUT

# ============================================================
# 3. EXECUÇÃO MULTICORE (OpenMP CPU)
# ============================================================
echo "Executando versão multicore..."
./mm_cpu -m cpu -w $WIDTH | tee -a $OUT
echo "" >> $OUT

# ============================================================
# 4. EXECUÇÕES GPU (OpenMP target offload)
# ============================================================
for variant in distribute distpar distparsimd; do
  echo "Executando versão GPU ($variant)..."
  echo "--- GPU $variant ---" >> $OUT
  nvprof --events warps_launched --metrics warp_execution_efficiency \
      ./mm_gpu -m gpu -g $variant -w $WIDTH 2>&1 | tee -a $OUT
  echo "" >> $OUT
done

echo "================== FINALIZADO ==================" >> $OUT
date >> $OUT
echo "Resultados salvos em $OUT"
