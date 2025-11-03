/*
  mm.c — Matrix-Matrix multiply (C = A * B) com variantes:

    VARIANT=0  -> sequencial (CPU)
    VARIANT=1  -> OpenMP multicore (CPU)
    VARIANT=2  -> OpenMP target teams distribute            (GPU)
    VARIANT=3  -> OpenMP target teams distribute parallel for (GPU)
    VARIANT=4  -> OpenMP target teams distribute parallel for simd (GPU)

  Exemplos (o run_all.sh já faz isso):
    gcc -O3 -fopenmp -DVARIANT=0 mm.c -o mm_seq
    gcc -O3 -fopenmp -DVARIANT=1 mm.c -o mm_cpu
    gcc -O3 -fopenmp -foffload=nvptx-none -misa=sm_70 -DVARIANT=2 mm.c -o mm_gpu_dist
    gcc -O3 -fopenmp -foffload=nvptx-none -misa=sm_70 -DVARIANT=3 mm.c -o mm_gpu_par
    gcc -O3 -fopenmp -foffload=nvptx-none -misa=sm_70 -DVARIANT=4 mm.c -o mm_gpu_simd
*/

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef VARIANT
#define VARIANT 0   /* default: sequencial */
#endif

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

/* SEQ */
static void mm_seq_impl(const double *a, const double *b, double *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
        sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
      c[(size_t)i*n + j] = sum;
    }
  }
}

/* OpenMP CPU */
static void mm_cpu_omp_impl(const double *a, const double *b, double *c, int n) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
        sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
      c[(size_t)i*n + j] = sum;
    }
  }
}

/* GPU: target teams distribute */
static void mm_gpu_distribute_impl(const double *a, const double *b, double *c, int n) {
  size_t sz = (size_t)n * (size_t)n;
  #pragma omp target data map(to: a[0:sz], b[0:sz]) map(from: c[0:sz])
  {
    #pragma omp target teams distribute collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
          sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        c[(size_t)i*n + j] = sum;
      }
    }
  }
}

/* GPU: target teams distribute parallel for */
static void mm_gpu_distribute_parallel_for_impl(const double *a, const double *b, double *c, int n) {
  size_t sz = (size_t)n * (size_t)n;
  #pragma omp target data map(to: a[0:sz], b[0:sz]) map(from: c[0:sz])
  {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
          sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        c[(size_t)i*n + j] = sum;
      }
    }
  }
}

/* GPU: target teams distribute parallel for simd */
static void mm_gpu_distribute_parallel_for_simd_impl(const double *a, const double *b, double *c, int n) {
  size_t sz = (size_t)n * (size_t)n;
  #pragma omp target data map(to: a[0:sz], b[0:sz]) map(from: c[0:sz])
  {
    #pragma omp target teams distribute parallel for simd collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++)
          sum += a[(size_t)i*n + k] * b[(size_t)k*n + j];
        c[(size_t)i*n + j] = sum;
      }
    }
  }
}

static double run_and_time(void (*fn)(const double*, const double*, double*, int),
                           const char *label,
                           const double *a, const double *b, double *c, int n)
{
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < (size_t)n*n; i++) c[i] = 0.0;

  double t0 = omp_get_wtime();
  fn(a,b,c,n);
  double t1 = omp_get_wtime();
  double sec = t1 - t0;
  double s = checksum(c, n);
  printf("%-38s | time = %.6f s | checksum = %.4f\n", label, sec, s);
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

  printf("n=%d  (%.2f MB por matriz)  VARIANT=%d\n", n, bytes/1048576.0, VARIANT);

#if   VARIANT == 0
  run_and_time(mm_seq_impl, "SEQ (CPU 1T)", a,b,c,n);
#elif VARIANT == 1
  run_and_time(mm_cpu_omp_impl, "OpenMP CPU", a,b,c,n);
#elif VARIANT == 2
  run_and_time(mm_gpu_distribute_impl, "GPU distribute", a,b,c,n);
#elif VARIANT == 3
  run_and_time(mm_gpu_distribute_parallel_for_impl, "GPU dist+parallel for", a,b,c,n);
#elif VARIANT == 4
  run_and_time(mm_gpu_distribute_parallel_for_simd_impl, "GPU dist+parallel for simd", a,b,c,n);
#else
# error "VARIANT inválido. Use 0..4."
#endif

  free(a); free(b); free(c);
  return 0;
}
