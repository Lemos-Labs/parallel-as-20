/*******************************************************
RESULTS (preencha após executar os scripts)

MACHINE / GPU: <GPU/CPU info aqui>  (ex.: GTX 1030 - SM 6.1)
COMPILERS:
  - gcc-5 (host)   flags: -O3 -fopenmp
  - nvcc           flags: -O3 -Xcompiler -fopenmp -gencode arch=compute_61,code=sm_61

MATRIX SIZE (width): 2000  (ou o que você usou)

TIMINGS (segundos):
  seq (CPU -O3):                   <preencher>
  omp (CPU OpenMP -O3):            <preencher>   (num_threads=<N>)
  cuda-naive (end-to-end):         <preencher>   (H2D + kernel + D2H)
    - kernel-only:                 <preencher>
  cuda-tiled (end-to-end):         <preencher>   (H2D + kernel + D2H)
    - kernel-only:                 <preencher>

GPU PROFILING (nvprof):
  cuda-naive:
    warps_launched:                <preencher>
    warp_execution_efficiency:     <preencher> %
  cuda-tiled:
    warps_launched:                <preencher>
    warp_execution_efficiency:     <preencher> %

Como coletar (exemplo):
  nvprof --events warps_launched --metrics warp_execution_efficiency ./mm --variant cuda-tiled --width 2000
*******************************************************/

/* ---- Hacks p/ NVCC + GCC5: desabilita __float128 nos headers da libstdc++ ---- */
#ifndef _GLIBCXX_USE_FLOAT128
#define _GLIBCXX_USE_FLOAT128 0
#endif
#define __STRICT_ANSI__ 1
/* ------------------------------------------------------------------------------ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>   // gettimeofday
#ifdef _OPENMP
  #include <omp.h>
#endif

#include <cuda_runtime.h>

/* -------------------------- Utils -------------------------- */

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

static inline double walltime_s() {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timeval tv;
  gettimeofday(&tv, 0);
  return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
#endif
}

static void init_mats(double* a, double* b, double* c, int width) {
  /* mesmo padrão do código original: a[i,j]=i; b[i,j]=j; c=0 */
  int i, j;
#ifdef _OPENMP
  #pragma omp parallel for private(j) schedule(static)
#endif
  for (i = 0; i < width; ++i) {
    for (j = 0; j < width; ++j) {
      a[i*width + j] = (double)i;
      b[i*width + j] = (double)j;
      c[i*width + j] = 0.0;
    }
  }
}

static void mm_seq(const double* a, const double* b, double* c, int width) {
  int i, j, k;
  for (i = 0; i < width; ++i) {
    for (j = 0; j < width; ++j) {
      double sum = 0.0;
      for (k = 0; k < width; ++k) {
        sum += a[i*width + k] * b[k*width + j];
      }
      c[i*width + j] = sum;
    }
  }
}

static void mm_omp(const double* a, const double* b, double* c, int width) {
  int i, j, k;
#ifdef _OPENMP
  #pragma omp parallel for private(j,k) schedule(static) collapse(2)
#endif
  for (i = 0; i < width; ++i) {
    for (j = 0; j < width; ++j) {
      double sum = 0.0;
      for (k = 0; k < width; ++k) {
        sum += a[i*width + k] * b[k*width + j];
      }
      c[i*width + j] = sum;
    }
  }
}

/* -------------------------- CUDA kernels -------------------------- */

/* Versão ingênua: um thread por elemento (i,j), loop em k dentro do kernel */
__global__ void mm_kernel_naive(const double* __restrict__ A,
                                const double* __restrict__ B,
                                double* __restrict__ C,
                                int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= width || col >= width) return;

  double sum = 0.0;
  for (int k = 0; k < width; ++k) {
    sum += A[row * width + k] * B[k * width + col];
  }
  C[row * width + col] = sum;
}

/* Versão otimizada com tiling e shared memory */
template<int TILE>
__global__ void mm_kernel_tiled(const double* __restrict__ A,
                                const double* __restrict__ B,
                                double* __restrict__ C,
                                int width)
{
  __shared__ double As[TILE][TILE];
  __shared__ double Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  double sum = 0.0;
  int tiles = (width + TILE - 1) / TILE;

  for (int t = 0; t < tiles; ++t) {
    int tiledCol = t * TILE + threadIdx.x; // coluna para A
    int tiledRow = t * TILE + threadIdx.y; // linha para B

    As[threadIdx.y][threadIdx.x] = (row < width && tiledCol < width) ? A[row * width + tiledCol] : 0.0;
    Bs[threadIdx.y][threadIdx.x] = (tiledRow < width && col < width) ? B[tiledRow * width + col] : 0.0;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < width && col < width) {
    C[row * width + col] = sum;
  }
}

/* -------------------------- CUDA runners -------------------------- */

typedef struct {
  float total_ms;   /* H2D + kernel + D2H */
  float kernel_ms;  /* somente kernel     */
} GpuTimings;

static GpuTimings run_cuda_naive(const double* hA, const double* hB, double* hC, int width) {
  const size_t bytes = (size_t)width * (size_t)width * sizeof(double);
  double *dA = 0, *dB = 0, *dC = 0;

  cudaEvent_t t0, t1, k0, k1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventCreate(&k0));
  CUDA_CHECK(cudaEventCreate(&k1));

  GpuTimings out; out.total_ms = 0.0f; out.kernel_ms = 0.0f;

  CUDA_CHECK(cudaEventRecord(t0));
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 block(32, 32);
  dim3 grid((width + block.x - 1) / block.x,
            (width + block.y - 1) / block.y);

  CUDA_CHECK(cudaEventRecord(k0));
  mm_kernel_naive<<<grid, block>>>(dA, dB, dC, width);
  CUDA_CHECK(cudaEventRecord(k1));
  CUDA_CHECK(cudaEventSynchronize(k1));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));

  CUDA_CHECK(cudaEventElapsedTime(&out.total_ms, t0, t1));
  CUDA_CHECK(cudaEventElapsedTime(&out.kernel_ms, k0, k1));

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaEventDestroy(k0));
  CUDA_CHECK(cudaEventDestroy(k1));

  return out;
}

template<int TILE>
static GpuTimings run_cuda_tiled(const double* hA, const double* hB, double* hC, int width) {
  const size_t bytes = (size_t)width * (size_t)width * sizeof(double);
  double *dA = 0, *dB = 0, *dC = 0;

  cudaEvent_t t0, t1, k0, k1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventCreate(&k0));
  CUDA_CHECK(cudaEventCreate(&k1));

  GpuTimings out; out.total_ms = 0.0f; out.kernel_ms = 0.0f;

  CUDA_CHECK(cudaEventRecord(t0));
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((width + TILE - 1) / TILE,
            (width + TILE - 1) / TILE);

  CUDA_CHECK(cudaEventRecord(k0));
  mm_kernel_tiled<TILE><<<grid, block>>>(dA, dB, dC, width);
  CUDA_CHECK(cudaEventRecord(k1));
  CUDA_CHECK(cudaEventSynchronize(k1));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));

  CUDA_CHECK(cudaEventElapsedTime(&out.total_ms, t0, t1));
  CUDA_CHECK(cudaEventElapsedTime(&out.kernel_ms, k0, k1));

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaEventDestroy(k0));
  CUDA_CHECK(cudaEventDestroy(k1));

  return out;
}

/* -------------------------- Driver / CLI -------------------------- */

static void usage(const char* prog) {
  printf("Usage: %s [--width N] [--variant seq|omp|cuda-naive|cuda-tiled] [--threads T]\n", prog);
}

int main(int argc, char** argv) {
  int width = 2000;
  const char* variant = "seq";
  int threads = 0; /* 0 = usar padrão do OMP */

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--width") && i+1 < argc) {
      width = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--variant") && i+1 < argc) {
      variant = argv[++i];
    } else if (!strcmp(argv[i], "--threads") && i+1 < argc) {
      threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--help")) {
      usage(argv[0]);
      return 0;
    }
  }

  printf("width=%d variant=%s\n", width, variant);

  size_t bytes = (size_t)width * (size_t)width * sizeof(double);
  double* A = (double*) std::malloc(bytes);
  double* B = (double*) std::malloc(bytes);
  double* C = (double*) std::malloc(bytes);

  if (!A || !B || !C) {
    fprintf(stderr, "malloc failed\n");
    return EXIT_FAILURE;
  }

  init_mats(A, B, C, width);

#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif

  if (!strcmp(variant, "seq")) {
    double t0 = walltime_s();
    mm_seq(A, B, C, width);
    double t1 = walltime_s();
    printf("[seq] time(s): %.6f\n", (t1 - t0));
  }
  else if (!strcmp(variant, "omp")) {
    double t0 = walltime_s();
    mm_omp(A, B, C, width);
    double t1 = walltime_s();
#ifdef _OPENMP
    int nt = 1;
    #pragma omp parallel
    {
      #pragma omp master
      nt = omp_get_num_threads();
    }
    printf("[omp] threads=%d time(s): %.6f\n", nt, (t1 - t0));
#else
    printf("[omp] (compiled w/o OpenMP) time(s): %.6f\n", (t1 - t0));
#endif
  }
  else if (!strcmp(variant, "cuda-naive")) {
    GpuTimings gt = run_cuda_naive(A, B, C, width);
    printf("[cuda-naive] total_ms(H2D+kernel+D2H)=%.3f  kernel_only_ms=%.3f\n", gt.total_ms, gt.kernel_ms);
  }
  else if (!strcmp(variant, "cuda-tiled")) {
    GpuTimings gt = run_cuda_tiled<32>(A, B, C, width); /* TILE=32 */
    printf("[cuda-tiled] total_ms(H2D+kernel+D2H)=%.3f  kernel_only_ms=%.3f\n", gt.total_ms, gt.kernel_ms);
  }
  else {
    usage(argv[0]);
  }

  std::free(A);
  std::free(B);
  std::free(C);
  return 0;
}
