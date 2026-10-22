/* Row-major SGEMM  C[M][N] = A[M][K] * B[K][N]  (float32). Rows are split across NT POSIX threads;
   cache-blocked, inner loop auto-vectorised (NEON). Build: gcc -O3 -shared -fPIC -pthread */
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#define KB 256
#define NB 512
typedef struct { int i0, i1, N, K; const float *A, *B; float *C; } job_t;
static void *work(void *p) {
    job_t *j = (job_t *)p; int N = j->N, K = j->K;
    for (int k0 = 0; k0 < K; k0 += KB) { int k1 = k0 + KB < K ? k0 + KB : K;
        for (int j0 = 0; j0 < N; j0 += NB) { int j1 = j0 + NB < N ? j0 + NB : N;
            for (int i = j->i0; i < j->i1; i++) { float *c = j->C + (size_t)i * N; const float *a = j->A + (size_t)i * K;
                for (int k = k0; k < k1; k++) { const float av = a[k]; const float *b = j->B + (size_t)k * N;
                    for (int x = j0; x < j1; x++) c[x] += av * b[x]; } } } }
    return 0;
}
static int NT = 4;
void set_threads(int n) { NT = n < 1 ? 1 : n; }
void sgemm(int M, int N, int K, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * (size_t)M * N);
    int nt = NT < M ? NT : M; pthread_t th[16]; job_t jb[16];
    for (int t = 0; t < nt; t++) { jb[t] = (job_t){ M * t / nt, M * (t + 1) / nt, N, K, A, B, C };
        if (t) pthread_create(&th[t], 0, work, &jb[t]); }
    work(&jb[0]);
    for (int t = 1; t < nt; t++) pthread_join(th[t], 0);
}
