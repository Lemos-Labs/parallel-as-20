/*
  mm.c — Matrix-Matrix multiply (C = A * B) com variantes:
    VARIANT=seq        -> sequencial (CPU)
    VARIANT=cpu        -> OpenMP multicore (CPU)
    VARIANT=gpu_dist   -> OpenMP target teams distribute
    VARIANT=gpu_par    -> OpenMP target teams distribute parallel for
    VARIANT=gpu_simd   -> OpenMP target teams distribute parallel for simd

  Como compilar (exemplos; o script run_all.sh faz isso pra você):
    gcc -O3 -fopenmp -DVARIANT=seq mm.c -o mm_seq
    gcc -O3 -fopenmp -DVARIANT=cpu mm.c -o mm_cpu
    gcc -O3 -fopenmp -foffload=nvptx-none -misa=sm_70 -DVARIANT=gpu_dist mm.c -o mm_gpu_dist
    gcc -O3 -fopenmp -foffload=nvptx-none -misa=sm_70 -DVARIANT=gpu_par  mm.c -o mm_gpu_par
    gcc -O3 -fopenmp -foffload=nvptx-none -misa=sm_70 -DVARIANT=gpu_simd mm.c -o mm_gpu_simd

  Execução:
    ./mm_seq [WIDTH]
    ./mm_cpu [WIDTH]
    ./mm_gpu_dist [WIDTH]
    ./mm_gpu_par [WIDTH]
    ./mm_gpu_simd [WIDTH]

  Coleta de métricas (GPU):
    nvprof --events warps_launched --metrics warp_execution_efficiency ./mm_gpu_dist
    (idem para as outras variantes GPU)

  Observação do enunciado:
    Se a sua submissão final exigir PRAGMAS comentados, basta submeter este mesmo arquivo
    SEM o script, pois as diretivas estão “fixadas” por VARIANT em tempo de compilação.
*/

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#ifndef WIDTH_DEFAULT
#define WIDTH_DEFAULT 2000
#endif

static void init(double *a, double *b, double *c, int n) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[(size_t)i*n + j] = (double)i;
      b[(size_t)i*n + j] = (double)j;
      c[(size_t)i*n + j] = 0.0;
    }
  }
}

static double checksum(const double *c, int n) {
  double s = 0.0;
  #pragma omp parallel for reduction(+:s) schedule(static)
  for (size_t i = 0; i < (size_t)n*n; i++) s += c[i];
  return s;
}

static void mm_seq_impl(const double *a, const double *b, double *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
      }
      c[(size_t)i*n + j] = sum;
    }
  }
}

static void mm_cpu_omp_impl(const double *a, const double *b, double *c, int n) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
      }
      c[(size_t)i*n + j] = sum;
    }
  }
}

static void mm_gpu_distribute_impl(const double *a, const double *b, double *c, int n) {
  size_t sz = (size_t)n * (size_t)n;
  #pragma omp target data map(to: a[0:sz], b[0:sz]) map(from: c[0:sz])
  {
    #pragma omp target teams distribute collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
          sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        }
        c[(size_t)i*n + j] = sum;
      }
    }
  }
}

static void mm_gpu_distribute_parallel_for_impl(const double *a, const double *b, double *c, int n) {
  size_t sz = (size_t)n * (size_t)n;
  #pragma omp target data map(to: a[0:sz], b[0:sz]) map(from: c[0:sz])
  {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
          sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        }
        c[(size_t)i*n + j] = sum;
      }
    }
  }
}

static void mm_gpu_distribute_parallel_for_simd_impl(const double *a, const double *b, double *c, int n) {
  size_t sz = (size_t)n * (size_t)n;
  #pragma omp target data map(to: a[0:sz], b[0:sz]) map(from: c[0:sz])
  {
    #pragma omp target teams distribute parallel for simd collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
          sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        }
        c[(size_t)i*n + j] = sum;
      }
    }
  }
}

static double run_and_time(void (*fn)(const double*, const double*, double*, int),
                           const char *label,
                           const double *a, const double *b, double *c, int n)
{
  // zera C
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < (size_t)n*n; i++) c[i] = 0.0;

  double t0 = omp_get_wtime();
  fn(a,b,c,n);
  double t1 = omp_get_wtime();
  double sec = t1 - t0;
  double s = checksum(c, n);
  printf("%-35s | time = %.6f s | checksum = %.4f\n", label, sec, s);
  return sec;
}

int main(int argc, char **argv) {
  int n = WIDTH_DEFAULT;
  if (argc > 1) {
    n = atoi(argv[1]);
    if (n <= 0) { fprintf(stderr, "WIDTH inválido.\n"); return 1; }
  }

  size_t elems = (size_t)n * (size_t)n;
  size_t bytes = elems * sizeof(double);

  double *a = (double*) malloc(bytes);
  double *b = (double*) malloc(bytes);
  double *c = (double*) malloc(bytes);
  if (!a || !b || !c) {
    fprintf(stderr, "Falha ao alocar memória (%.2f MB por matriz)\n", bytes/1048576.0);
    return 1;
  }

  init(a,b,c,n);

  printf("n=%d  (%.2f MB por matriz)  VARIANT=%s\n",
         n, bytes/1048576.0,
#ifdef VARIANT
#  define STR_(x) #x
#  define STR(x) STR_(x)
         STR(VARIANT)
#else
         "seq(default)"
#endif
  );

#ifndef VARIANT
#  define VARIANT seq
#endif

#if defined(VARIANT) && ( \
     (0 * (VARIANT==zzz)) || 1 )
  // No-op: only to avoid warnings in some compilers about empty translation
#endif

#if defined(VARIANT) && !defined(__NVPTX__)
  // nothing here; just a placeholder for preprocessors that expand macros oddly
#endif

#if   defined(VARIANT) && (0)
  // unreachable
#elif defined(VARIANT) && !defined(__NVPTX__)
  // host build conditions are fine; selection happens below
#endif

#if   defined(VARIANT) && (strcmp(STR(VARIANT), "seq") == 0)
  run_and_time(mm_seq_impl, "SEQ (CPU 1T)", a,b,c,n);
#elif defined(VARIANT) && (strcmp(STR(VARIANT), "cpu") == 0)
  run_and_time(mm_cpu_omp_impl, "OpenMP CPU", a,b,c,n);
#elif defined(VARIANT) && (strcmp(STR(VARIANT), "gpu_dist") == 0)
  run_and_time(mm_gpu_distribute_impl, "GPU distribute", a,b,c,n);
#elif defined(VARIANT) && (strcmp(STR(VARIANT), "gpu_par") == 0)
  run_and_time(mm_gpu_distribute_parallel_for_impl, "GPU dist+parallel for", a,b,c,n);
#elif defined(VARIANT) && (strcmp(STR(VARIANT), "gpu_simd") == 0)
  run_and_time(mm_gpu_distribute_parallel_for_simd_impl, "GPU dist+parallel for simd", a,b,c,n);
#else
  run_and_time(mm_seq_impl, "SEQ (CPU 1T) [default]", a,b,c,n);
#endif

  free(a); free(b); free(c);
  return 0;
}
