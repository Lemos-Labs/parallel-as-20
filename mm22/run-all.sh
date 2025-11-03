#!/bin/bash
# =====================================================================
# Script: run_all_gcc.sh
# Requisitos: gcc com suporte a OpenMP offload NVPTX (libgomp-plugin-nvptx)
#             NVIDIA driver + CUDA toolkit (para nvprof)
# Uso: chmod +x run_all_gcc.sh && ./run_all_gcc.sh
# Dica: export SM=sm_80  (ou sm_70, sm_75, sm_86...) antes de rodar
# =====================================================================

set -euo pipefail

SRC="mm.c"
WIDTH="${WIDTH:-2000}"
OUT="results_gcc.txt"

# Ajuste a arquitetura da GPU aqui (ou via env SM=sm_80)
SM="${SM:-sm_70}"

CPU_FLAGS="-O3 -fopenmp"
GPU_FLAGS="-O3 -fopenmp -foffload=nvptx-none -foffload-options=nvptx-none=-misa=${SM}"

echo "================== MATRIX MULTIPLICATION (gcc) ==================" > "$OUT"
echo "Matrix size: ${WIDTH} x ${WIDTH}" | tee -a "$OUT"
echo "Compiler : gcc" | tee -a "$OUT"
echo "CPU flags: ${CPU_FLAGS}" | tee -a "$OUT"
echo "GPU flags: ${GPU_FLAGS}" | tee -a "$OUT"
date | tee -a "$OUT"
echo "" | tee -a "$OUT"

# ---------------------------------------------------------------------
# 1) Compilar
# ---------------------------------------------------------------------
echo "[Compilando (CPU)]"
gcc ${CPU_FLAGS} "${SRC}" -o mm_cpu

echo "[Compilando (GPU offload NVPTX)]"
gcc ${GPU_FLAGS} "${SRC}" -o mm_gpu

# ---------------------------------------------------------------------
# 2) Rodar SEQ
# ---------------------------------------------------------------------
echo -e "\n[Executando SEQ]" | tee -a "$OUT"
./mm_cpu -m seq -w "${WIDTH}" | tee -a "$OUT"

# ---------------------------------------------------------------------
# 3) Rodar CPU OpenMP (necessita descomentar pragma em mm_omp_cpu para ganho real)
# ---------------------------------------------------------------------
echo -e "\n[Executando CPU OpenMP]" | tee -a "$OUT"
./mm_cpu -m cpu -w "${WIDTH}" | tee -a "$OUT"

# ---------------------------------------------------------------------
# 4) Rodar GPU (3 variações) — é necessário descomentar um bloco de pragma por vez em mm_omp_gpu
#    Para cada variação abaixo, deixe APENAS aquela variação descomentada e recompile mm_gpu.
#    Se você já descomentou todas manualmente e criou 3 arquivos distintos, ajuste o script conforme.
# ---------------------------------------------------------------------

run_gpu_variant () {
  local variant="$1"
  echo -e "\n[Executando GPU OpenMP target] variant=${variant}" | tee -a "$OUT"
  echo "--- GPU ${variant} ---" | tee -a "$OUT"
  # Garante que falhe se não houver offload (em vez de cair na CPU)
  OMP_TARGET_OFFLOAD=MANDATORY \
  nvprof --events warps_launched --metrics warp_execution_efficiency \
    ./mm_gpu -m gpu -g "${variant}" -w "${WIDTH}" 2>&1 | tee -a "$OUT"
}

# AVISO IMPORTANTE:
#   Antes de rodar cada uma das linhas abaixo, descomente APENAS o bloco correspondente no mm.c
#   (dentro da função mm_omp_gpu) e recompile (linha "Compilando GPU" acima).
#   Exemplo de ciclo:
#     1) Descomente "distribute" -> ./run_all_gcc.sh (ou rode só o bloco abaixo)
#     2) Comente "distribute", descomente "distpar" -> recompila e roda
#     3) Comente "distpar", descomente "distparsimd" -> recompila e roda
#
# Se preferir, crie 3 cópias do arquivo com cada variação descomentada e troque o binário chamado aqui.

# Descomente UMA chamada por vez quando o respectivo bloco estiver ativo no mm.c:
# run_gpu_variant distribute
# run_gpu_variant distpar
# run_gpu_variant distparsimd

echo -e "\n================== FINALIZADO ==================" | tee -a "$OUT"
date | tee -a "$OUT"
echo "Resultados em: ${OUT}"
