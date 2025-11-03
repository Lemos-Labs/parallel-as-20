#!/bin/sh
# run_all.sh — compila e executa as variantes (seq, cpu, gpu_*), e coleta métricas GPU com nvprof
# Requisitos: gcc, nvprof, driver NVIDIA, GCC com OpenMP offload NVPTX
# Variáveis:
#   SM    -> arquitetura CUDA (ex.: sm_70, sm_80) [default: sm_70]
#   WIDTH -> tamanho da matriz [default: 2000]
#   CC    -> compilador (default: gcc)

set -eu

CC="${CC:-gcc}"
SM="${SM:-sm_70}"
WIDTH="${WIDTH:-2000}"

echo "==> Config:"
echo "CC=$CC  SM=$SM  WIDTH=$WIDTH"
echo

build_seq() {
  echo "[build] mm_seq"
  "$CC" -O3 -fopenmp -DVARIANT=0 mm.c -o mm_seq
}

build_cpu() {
  echo "[build] mm_cpu (OpenMP multicore)"
  "$CC" -O3 -fopenmp -DVARIANT=1 mm.c -o mm_cpu
}

build_gpu() {
  name="$1" variant="$2"
  echo "[build] $name (OpenMP target offload)"
  "$CC" -O3 -fopenmp -foffload=nvptx-none -misa="$SM" -DVARIANT="$variant" mm.c -o "$name"
}

run_bin() {
  bin="$1"
  echo
  echo "---- Running: ./$bin $WIDTH ----"
  "./$bin" "$WIDTH"
}

nvprof_maybe() {
  bin="$1"
  if command -v nvprof >/dev/null 2>&1; then
    echo
    echo ">> nvprof metrics for $bin"
    nvprof --events warps_launched --metrics warp_execution_efficiency "./$bin" "$WIDTH" 2>&1 | \
      awk '/warps_launched/ || /warp_execution_efficiency/'
  else
    echo "[warn] nvprof não encontrado — pulando métricas."
  fi
}

# 1) Build
build_seq
build_cpu
build_gpu mm_gpu_dist  2
build_gpu mm_gpu_par   3
build_gpu mm_gpu_simd  4

# 2) Run
run_bin mm_seq
run_bin mm_cpu
run_bin mm_gpu_dist
nvprof_maybe mm_gpu_dist
run_bin mm_gpu_par
nvprof_maybe mm_gpu_par
run_bin mm_gpu_simd
nvprof_maybe mm_gpu_simd

echo
echo "==> Concluído. Anote os tempos e as métricas (warps_launched, warp_execution_efficiency)."
