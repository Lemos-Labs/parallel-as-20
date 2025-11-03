/*
========================================================
Matrix-Matrix Multiplication (mm.c) — OpenMP & GPU Offload
========================================================

[RESULTADOS (preencher após as execuções) — todos com -O3]

Machine / Compilers:
- CPU:
- GPU:
- Compiler (CPU):
- Compiler (GPU OpenMP target):

Width = 2000  (padrão; altere via linha de comando -w N)

Tempos de execução (segundos):
- Sequencial:                  TODO
- Multicore (OpenMP CPU):      TODO
- GPU - distribute:            TODO
- GPU - distribute parallel for:        TODO
- GPU - distribute parallel for simd:   TODO

Métricas GPU (via nvprof):
- GPU - distribute:
    warps_launched = TODO
    warp_execution_efficiency = TODO %
- GPU - distribute parallel for:
    warps_launched = TODO
    warp_execution_efficiency = TODO %
- GPU - distribute parallel for simd:
    warps_launched = TODO
    warp_execution_efficiency = TODO %

Como medir (exemplos):

# Multicore (CPU):
gcc -O3 -fopenmp mm.c -o mm
./mm -m cpu -w 2000

# GPU (OpenMP target) — exemplos (ajuste ao seu toolchain/GPU):
# Clang/LLVM:
clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda mm.c -o mm
# GCC (versões recentes com NVPTX):
gcc -O3 -fopenmp -foffload=nvptx-none mm.c -o mm
# NVIDIA HPC SDK (nvc):
nvc -O3 -mp=gpu mm.c -o mm

# Rodar e perfilar (substitua a variação por uma das abaixo):
#   -g distribute
#   -g distpar
#   -g distparsimd
nvprof --events warps_launched --metrics warp_execution_efficiency ./mm -m gpu -g distpar -w 2000

Notas:
- Por exigência da entrega, os pragmas para multicore e GPU estão PRESENTES porém COMENTADOS.
  Para rodar as versões paralelas, descomente os pragmas correspondentes.
- Todas as versões devem ser compiladas com -O3.
========================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Alocação alinhada opcional para melhor throughput em algumas plataformas */
static double* alloc_matrix(size_t n)
{
    size_t bytes = n * sizeof(double);
    void* p = NULL;
#if defined(_MSC_VER)
    p = _aligned_malloc(bytes, 64);
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
#else
    if (posix_memalign(&p, 64, bytes) != 0) { fprintf(stderr, "alloc failed\n"); exit(1); }
#endif
    return (double*)p;
}

static void free_matrix(double* p)
{
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Versão sequencial (baseline) */
void mm_seq(const double* a, const double* b, double* c, int width)
{
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0.0;
            for (int k = 0; k < width; k++) {
                double x = a[i * width + k];
                double y = b[k * width + j];
                sum += x * y;
            }
            c[i * width + j] = sum;
        }
    }
}

/* Versão multicore (OpenMP CPU) — pragmas COMENTADOS por exigência da entrega */
void mm_omp_cpu(const double* a, const double* b, double* c, int width)
{
    /* Descomente as linhas abaixo para ativar paralelismo em CPU

    #pragma omp parallel for collapse(2) schedule(static)
    */
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0.0;
            /* Opcional: simd para o loop interno
            #pragma omp simd reduction(+:sum)
            */
            for (int k = 0; k < width; k++) {
                double x = a[i * width + k];
                double y = b[k * width + j];
                sum += x * y;
            }
            c[i * width + j] = sum;
        }
    }
}

/* ===== GPU OpenMP Target Offload =====
   Três variações pedidas, todas COMENTADAS:
   1) target teams distribute
   2) target teams distribute parallel for
   3) target teams distribute parallel for simd
   Use a mesma função e descomente apenas UM bloco de pragma por vez.
*/

void mm_omp_gpu(const double* a, const double* b, double* c, int width, const char* variant)
{
    size_t N = (size_t)width * (size_t)width;

    if (strcmp(variant, "distribute") == 0) {

        /* -------------------- VARIAÇÃO 1: distribute --------------------
        #pragma omp target teams distribute collapse(2) \
            map(to: a[0:N], b[0:N]) map(from: c[0:N])
        */
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                double sum = 0.0;
                for (int k = 0; k < width; k++) {
                    double x = a[i * width + k];
                    double y = b[k * width + j];
                    sum += x * y;
                }
                c[i * width + j] = sum;
            }
        }

    } else if (strcmp(variant, "distpar") == 0) {

        /* -------------- VARIAÇÃO 2: distribute parallel for --------------
        #pragma omp target teams distribute parallel for collapse(2) \
            map(to: a[0:N], b[0:N]) map(from: c[0:N]) schedule(static)
        */
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                double sum = 0.0;
                for (int k = 0; k < width; k++) {
                    double x = a[i * width + k];
                    double y = b[k * width + j];
                    sum += x * y;
                }
                c[i * width + j] = sum;
            }
        }

    } else if (strcmp(variant, "distparsimd") == 0) {

        /* -------- VARIAÇÃO 3: distribute parallel for simd --------
        #pragma omp target teams distribute parallel for simd collapse(2) \
            map(to: a[0:N], b[0:N]) map(from: c[0:N]) schedule(static)
        */
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < width; j++) {
                double sum = 0.0;
                /* Dentro do k, poderíamos usar simd adicional, mas aqui a
                   combinação já aplica vetorização no loop colapsado. */
                for (int k = 0; k < width; k++) {
                    double x = a[i * width + k];
                    double y = b[k * width + j];
                    sum += x * y;
                }
                c[i * width + j] = sum;
            }
        }

    } else {
        fprintf(stderr, "Variante GPU desconhecida: %s\n", variant);
        exit(1);
    }
}

/* Inicialização simples e checksum para validar */
static void init_mats(double* a, double* b, double* c, int width)
{
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            a[i * width + j] = (double)i;
            b[i * width + j] = (double)j;
            c[i * width + j] = 0.0;
        }
    }
}

static double checksum(const double* c, int width)
{
    double s = 0.0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            s += c[i * width + j];
        }
    }
    return s;
}

static void usage(const char* prog)
{
    fprintf(stderr,
        "Uso: %s [-w width] [-m mode] [-g variant]\n"
        "  -w width   : tamanho da matriz (default 2000)\n"
        "  -m mode    : seq | cpu | gpu   (default seq)\n"
        "  -g variant : distribute | distpar | distparsimd (para -m gpu; default distpar)\n"
        "Exemplos:\n"
        "  %s -m seq -w 2000\n"
        "  %s -m cpu -w 2000      (descomente pragmas CPU em mm_omp_cpu)\n"
        "  %s -m gpu -g distpar   (descomente UMA variação em mm_omp_gpu)\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv)
{
    int width = 2000;
    const char* mode = "seq";       /* seq | cpu | gpu */
    const char* gvar = "distpar";   /* distribute | distpar | distparsimd */

    /* Parse simples */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-w") && i+1 < argc) {
            width = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-m") && i+1 < argc) {
            mode = argv[++i];
        } else if (!strcmp(argv[i], "-g") && i+1 < argc) {
            gvar = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    size_t N = (size_t)width * (size_t)width;

    double* a = alloc_matrix(N);
    double* b = alloc_matrix(N);
    double* c = alloc_matrix(N);

    init_mats(a, b, c, width);

    double t0 = omp_get_wtime();

    if (!strcmp(mode, "seq")) {
        mm_seq(a, b, c, width);
    } else if (!strcmp(mode, "cpu")) {
        /* Lembre-se: para ter ganho, DESCOMENTE os pragmas em mm_omp_cpu */
        mm_omp_cpu(a, b, c, width);
    } else if (!strcmp(mode, "gpu")) {
        /* Lembre-se: DESCOMENTE apenas UMA variação de pragma em mm_omp_gpu */
        mm_omp_gpu(a, b, c, width, gvar);
    } else {
        fprintf(stderr, "Modo desconhecido: %s\n", mode);
        free_matrix(a); free_matrix(b); free_matrix(c);
        return 1;
    }

    double t1 = omp_get_wtime();
    double secs = t1 - t0;

    /* Checksum para validar e evitar eliminação */
    double s = checksum(c, width);

    printf("Mode=%s", mode);
    if (!strcmp(mode, "gpu")) printf(" Variant=%s", gvar);
    printf(" | Width=%d | Time(s)=%.6f | Checksum=%.6f\n", width, secs, s);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    return 0;
}
