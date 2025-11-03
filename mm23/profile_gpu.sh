#!/usr/bin/env sh
set -e

WIDTH=${1:-2000}

if ! command -v nvprof >/dev/null 2>&1; then
  echo "nvprof não encontrado no PATH. Instale/adicione CUDA toolkit (<=11) com nvprof."
  exit 1
fi

echo "== nvprof: cuda-naive =="
nvprof --events warps_launched --metrics warp_execution_efficiency ./mm --variant cuda-naive --width $WIDTH 2>&1 | tee nvprof_cuda_naive.txt

echo "== nvprof: cuda-tiled =="
nvprof --events warps_launched --metrics warp_execution_efficiency ./mm --variant cuda-tiled --width $WIDTH 2>&1 | tee nvprof_cuda_tiled.txt

echo "Logs salvos: nvprof_cuda_naive.txt, nvprof_cuda_tiled.txt"
echo "Copie os valores de warps_launched e warp_execution_efficiency para o cabeçalho do mm.cu."
