#!/bin/sh
# setup_modules_and_run.sh — prepara "environment modules", tenta carregar GCC+NVPTX+CUDA
# e então executa o run_all.sh (que compila/roda CPU e GPU com métricas nvprof).
#
# Uso:
#   chmod +x setup_modules_and_run.sh run_all.sh
#   ./setup_modules_and_run.sh
#
# Variáveis opcionais de ambiente:
#   WIDTH  (default: 2000)
#   SM     (default: sm_70)     # ex.: sm_80 para A100/H100, sm_75 para T4, sm_70 para V100
#   GCCMOD (força um módulo gcc específico, ex.: gcc/12.3.0)
#   CUDAMOD (força um módulo cuda específico, ex.: cuda/11.8)
#   EXTRA_MODS (outros módulos a carregar, ex.: "nvptx-tools libgomp-nvptx")
#
set -eu

WIDTH="${WIDTH:-2000}"
SM="${SM:-sm_70}"

# 1) Tentar inicializar o "module" (Environment Modules / Lmod) em shell POSIX
if ! command -v module >/dev/null 2>&1; then
  # Tenta localizar os scripts de inicialização comuns
  for f in /etc/profile.d/modules.sh /usr/share/Modules/init/sh /etc/profile.d/lmod.sh; do
    if [ -r "$f" ]; then
      # shellcheck disable=SC1090
      . "$f"
      break
    fi
  done
fi

if ! command -v module >/dev/null 2>&1; then
  echo "[erro] O comando 'module' não está disponível no seu shell."
  echo "       Entre em um ambiente interativo do cluster (ex.: 'bash --login' ou 'source /etc/profile'),"
  echo "       ou peça ao administrador a ativação do Environment Modules/Lmod."
  exit 1
fi

echo "==> Módulos: inicializado. Usando 'module' de $(command -v module)"

# 2) Helper para tentar carregar módulos candidatos
try_load() {
  # $1 = nome do módulo (ex.: gcc/12)
  mod="$1"
  if module avail "$mod" 2>&1 | grep -q "$mod"; then
    echo "[mod] carregando: $mod"
    module load "$mod" || return 1
    return 0
  fi
  return 1
}

# 3) Carregar GCC com suporte a OpenMP (preferindo versões novas)
if [ -n "${GCCMOD:-}" ]; then
  try_load "$GCCMOD" || { echo "[warn] módulo GCC '$GCCMOD' não encontrado."; }
else
  for gcccand in gcc/14 gcc/13 gcc/12 gcc/11 gcc/10; do
    if try_load "$gcccand"; then break; fi
  done
fi

# 4) Carregar CUDA (nvprof vem junto em versões 10.x/11.x; 12.x usa nv-nsight-cu-cli)
if [ -n "${CUDAMOD:-}" ]; then
  try_load "$CUDAMOD" || { echo "[warn] módulo CUDA '$CUDAMOD' não encontrado."; }
else
  for cudacand in cuda/12.4 cuda/12.3 cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8 cuda/11.7 cuda/11.6 cuda/11.4 cuda/11.2 cuda/11.0 cuda/10.2; do
    if try_load "$cudacand"; then break; fi
  done
fi

# 5) Tentar carregar utilitários NVPTX do GCC (nomes variam por cluster)
EXTRA_MODS="${EXTRA_MODS:-}"
if [ -n "$EXTRA_MODS" ]; then
  for m in $EXTRA_MODS; do
    try_load "$m" || echo "[warn] módulo extra '$m' indisponível (ok se sua toolchain já traz NVPTX embutido)."
  done
else
  # Alguns nomes comuns:
  for m in libgomp-nvptx libgomp-plugin-nvptx nvptx offload-nvptx nvptx-tools; do
    try_load "$m" && echo "[info] módulo NVPTX extra '$m' carregado." && break || true
  done
fi

echo
echo "==> Módulos ativos:"
module list 2>&1 || true
echo

# 6) Verificação rápida: gcc com offload NVPTX funcional?
cat > .omp_offload_test.c <<'EOF'
#include <omp.h>
int main() {
  int x=0;
  #pragma omp target map(tofrom:x)
  { x = 42; }
  return (x==42)?0:1;
}
EOF

if gcc -O2 -fopenmp -foffload=nvptx-none -misa="$SM" .omp_offload_test.c -o .omp_offload_test 2>/dev/null; then
  echo "[ok] GCC parece suportar OpenMP offload para NVPTX (sm=$SM)."
else
  echo "[warn] GCC ainda não compila offload NVPTX com as combinações atuais de módulos."
  echo "       Tentarei compilar via Clang (se existir) *apenas para GPU* dentro do run_all.sh."
  echo "       Caso falhe, carregue manualmente módulos como:"
  echo "         module load gcc/12"
  echo "         module load cuda/11.8"
  echo "         module load nvptx-tools   (ou libgomp-nvptx / libgomp-plugin-nvptx)"
  echo
fi

# 7) Detectar nvprof (se não houver, o run_all.sh avisa e segue sem métricas)
if ! command -v nvprof >/dev/null 2>&1; then
  echo "[warn] 'nvprof' não encontrado no PATH. Se seu cluster usa CUDA 12+, prefira:"
  echo "       ncu --metrics sm__warps_launched.sum,smsp__thread_inst_executed_per_inst_executed.pct"
  echo "       (o run_all.sh usa nvprof por exigência do enunciado; se não houver nvprof, ele só avisa.)"
fi

# 8) Chamar o pipeline principal
echo
echo "==> Rodando pipeline com SM=$SM WIDTH=$WIDTH"
SM="$SM" WIDTH="$WIDTH" sh ./run_all.sh
