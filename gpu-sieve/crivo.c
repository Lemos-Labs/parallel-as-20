/*
===========================================================
Crivo de Eratóstenes — 3 versões (Seq, OpenMP CPU, OpenMP GPU)
Aluno: <SEU NOME AQUI>

Tempos (compilar com -O3):
- Sequencial:        <preencha após rodar o run_all.sh>
- OpenMP multicore:  <preencha após rodar o run_all.sh>
- OpenMP GPU:        <preencha após rodar o run_all.sh>

Como compilar manualmente:
  gcc-8 -O3 -fopenmp crivo.c -o crivo     # CPU (seq/omp)
  gcc-8 -O3 -fopenmp -foffload=nvptx-none crivo.c -o crivo_gpu  # GPU (se suportado)

Como executar:
  ./crivo seq   [N]   # sequencial
  ./crivo cpu   [N]   # OpenMP multicore
  ./crivo gpu   [N]   # OpenMP target offload (GPU)
===========================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

// Use 1 byte por posição para simplificar mapeamento a dispositivos
typedef unsigned char ubyte;

// Inicializa o vetor de "primo?" com 1 (verdadeiro)
static inline ubyte* alloc_prime_array(long long n) {
    ubyte *a = (ubyte*) malloc((size_t)(n + 1));
    if (!a) { fprintf(stderr, "Erro: malloc(%lld)\n", n+1); exit(1); }
    memset(a, 1, (size_t)(n + 1));
    if (n >= 0) a[0] = 0;
    if (n >= 1) a[1] = 0;
    return a;
}

static inline int count_primes_host(const ubyte *a, long long n) {
    long long c = 0;
    for (long long i = 2; i <= n; ++i) c += a[i] ? 1 : 0;
    return (int)c;
}

/* ----------------- Versão Sequencial ----------------- */
int sieve_seq(long long n) {
    ubyte *prime = alloc_prime_array(n);
    long long sqrt_n = (long long) floor(sqrt((double)n));
    for (long long p = 2; p <= sqrt_n; ++p) {
        if (prime[p]) {
            long long start = p * p;                 // p^2
            for (long long i = start; i <= n; i += p)
                prime[i] = 0;
        }
    }
    int total = count_primes_host(prime, n);
    free(prime);
    return total;
}

/* -------------- Versão OpenMP Multicore -------------- */
int sieve_omp_cpu(long long n) {
    ubyte *prime = alloc_prime_array(n);
    long long sqrt_n = (long long) floor(sqrt((double)n));
    for (long long p = 2; p <= sqrt_n; ++p) {
        if (prime[p]) {
            long long start = p * p;
            #pragma omp parallel for schedule(static)
            for (long long i = start; i <= n; i += p) {
                prime[i] = 0; // Cada thread escreve índices distintos
            }
        }
    }
    int total = count_primes_host(prime, n);
    free(prime);
    return total;
}

/* -------- Versão OpenMP Target Offload (GPU) --------- */
/* Observações:
   - Requer toolchain com suporte a OpenMP 4.5+ e offload nvptx no gcc-8.
   - Usamos uma região target data para manter 'prime' no device e evitar
     remapeamento a cada p. Para cada p, lançamos um kernel simples que
     risca múltiplos de p em paralelo.
*/
int sieve_omp_gpu(long long n) {
    ubyte *prime = alloc_prime_array(n);
    long long sqrt_n = (long long) floor(sqrt((double)n));

    // Copia prime para o device e mantém lá durante o laço
    #pragma omp target data map(tofrom: prime[0:n+1])
    {
        for (long long p = 2; p <= sqrt_n; ++p) {
            // Precisamos checar prime[p]. Esse acesso é no host.
            if (prime[p]) {
                long long start = p * p;
                // Marca múltiplos em paralelo no device
                #pragma omp target teams distribute parallel for firstprivate(p, n, start)
                for (long long i = start; i <= n; i += p) {
                    prime[i] = 0;
                }
            }
        }
        // Ao sair da região target data, 'prime' volta para o host.
    }

    int total = count_primes_host(prime, n);
    free(prime);
    return total;
}

/* ------------------------- main ---------------------- */
static void usage(const char *prog) {
    fprintf(stderr, "Uso: %s {seq|cpu|gpu} [N]\n", prog);
    fprintf(stderr, "Ex.: %s cpu 100000000\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    const char *mode = argv[1];
    long long N = 100000000LL; // padrão
    if (argc >= 3) {
        char *endp = NULL;
        long long v = strtoll(argv[2], &endp, 10);
        if (!endp || *endp != '\0' || v < 2) {
            fprintf(stderr, "N inválido: %s\n", argv[2]);
            return 1;
        }
        N = v;
    }

    int result = -1;
    if (strcmp(mode, "seq") == 0) {
        result = sieve_seq(N);
    } else if (strcmp(mode, "cpu") == 0) {
        #ifndef _OPENMP
        fprintf(stderr, "Aviso: compilado sem OpenMP; rodando versão sequencial.\n");
        result = sieve_seq(N);
        #else
        result = sieve_omp_cpu(N);
        #endif
    } else if (strcmp(mode, "gpu") == 0) {
        #ifndef _OPENMP
        fprintf(stderr, "Erro: compilado sem OpenMP; GPU indisponível.\n");
        return 2;
        #else
        // Tentativa de offload; se a toolchain não suportar, pode falhar em runtime.
        result = sieve_omp_gpu(N);
        #endif
    } else {
        usage(argv[0]);
        return 1;
    }

    printf("%d\n", result); // imprime quantidade de primos até N
    return 0;
}
