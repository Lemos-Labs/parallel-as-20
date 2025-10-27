#!/bin/bash
set -euo pipefail

# Detecta compilador disponível
if command -v gcc-8 &> /dev/null; then
    CC=gcc-8
else
    echo "⚠️  gcc-8 não encontrado, usando gcc padrão ($(gcc --version | head -n1))"
    CC=gcc
fi

CFLAGS="-O3 -fopenmp -lm"
N=${1:-100000000}

echo "Compilador: $CC"
echo "N = $N"

# Compila versão CPU (seq/cpu)
echo "-> Compilando crivo (CPU)..."
$CC $CFLAGS crivo.c -o crivo

# Compila versão com tentativa de offload GPU (se suportado)
echo "-> Tentando compilar crivo com offload GPU..."
GPU_OK=1
if $CC $CFLAGS -foffload=nvptx-none crivo.c -o crivo_gpu 2> build_gpu.err; then
  echo "   Compilação GPU OK."
else
  echo "   Falhou a compilação com offload (veja build_gpu.err). Usarei o binário CPU para 'gpu' como fallback."
  GPU_OK=0
  cp crivo crivo_gpu
fi

measure() {
  local exe="$1"; shift
  local label="$1"; shift
  local nval="$1"; shift
  local t
  t=$(/usr/bin/time -f "%e" ./$exe $label "$nval" >/tmp/out.$$ 2>&1) || true
  local primes
  primes=$(tail -n1 /tmp/out.$$)
  rm -f /tmp/out.$$
  echo "$label: tempo(s)=$t  primos=$primes"
}

echo
echo "==== Execuções ===="
echo "Sequencial:"
SEQ_LINE=$(measure crivo seq "$N")

echo "OpenMP multicore:"
CPU_LINE=$(measure crivo cpu "$N")

echo "OpenMP GPU:"
if [[ "$GPU_OK" -eq 1 ]]; then
  GPU_LINE=$(measure crivo_gpu gpu "$N")
else
  GPU_LINE=$(measure crivo_gpu gpu "$N")
  echo "   *Aviso*: esta execução 'gpu' é fallback CPU (sem offload)."
fi

echo
echo "==== Resumo (copie para o cabeçalho do crivo.c) ===="
echo "$SEQ_LINE"
echo "$CPU_LINE"
echo "$GPU_LINE"
