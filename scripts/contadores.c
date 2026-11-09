/* Conta eventos de hardware de um programa, como um perf stat reduzido.
 * O Franky nao tem o perf instalado, mas libera perf_event_open para o proprio usuario
 * (perf_event_paranoid = 2), contando apenas o modo usuario.
 *
 * Compilar: gcc -O2 -o contadores scripts/contadores.c
 * Usar:     taskset -c 8 ./contadores ./barenn
 *
 * O i9-14900 tem PMUs separadas para P-cores e E-cores. Os eventos abaixo sao da PMU dos
 * P-cores, entao o programa medido precisa estar fixado em P-cores.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

struct evento {
    const char* nome;
    uint32_t tipo;
    uint64_t config;
};

struct leitura {
    uint64_t valor;
    uint64_t habilitado;
    uint64_t rodando;
};

static uint64_t tipo_pmu_p_core(void) {
    int tipo = 0;
    FILE* f = fopen("/sys/bus/event_source/devices/cpu_core/type", "r");
    if (f) {
        if (fscanf(f, "%d", &tipo) != 1) {
            tipo = 0;
        }
        fclose(f);
    }
    return (uint64_t)tipo << 32;
}

static int abre(const struct evento* e, pid_t pid) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = e->tipo;
    a.config = e->config;
    a.disabled = 1;
    a.enable_on_exec = 1;
    a.inherit = 1;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return (int)syscall(SYS_perf_event_open, &a, pid, -1, -1, 0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "uso: %s programa [argumentos]\n", argv[0]);
        return 2;
    }

    const uint64_t pmu = tipo_pmu_p_core();
    const uint64_t l1d_leitura_falha = PERF_COUNT_HW_CACHE_L1D |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    const struct evento eventos[] = {
        {"instrucoes", PERF_TYPE_HARDWARE, pmu | PERF_COUNT_HW_INSTRUCTIONS},
        {"ciclos", PERF_TYPE_HARDWARE, pmu | PERF_COUNT_HW_CPU_CYCLES},
        {"ciclos_ref", PERF_TYPE_HARDWARE, pmu | PERF_COUNT_HW_REF_CPU_CYCLES},
        {"falhas_L1D", PERF_TYPE_HW_CACHE, pmu | l1d_leitura_falha},
        {"acessos_LLC", PERF_TYPE_HARDWARE, pmu | PERF_COUNT_HW_CACHE_REFERENCES},
        {"falhas_LLC", PERF_TYPE_HARDWARE, pmu | PERF_COUNT_HW_CACHE_MISSES},
        {"falhas_desvio", PERF_TYPE_HARDWARE, pmu | PERF_COUNT_HW_BRANCH_MISSES},
    };
    enum { N = sizeof eventos / sizeof eventos[0] };

    /* O filho espera no pipe ate os contadores estarem abertos. */
    int canal[2];
    if (pipe(canal) != 0) {
        perror("pipe");
        return 2;
    }
    pid_t filho = fork();
    if (filho == 0) {
        char c;
        close(canal[1]);
        if (read(canal[0], &c, 1) != 1) {
            _exit(127);
        }
        close(canal[0]);
        execvp(argv[1], argv + 1);
        perror("execvp");
        _exit(127);
    }
    close(canal[0]);

    int fd[N];
    for (int i = 0; i < N; ++i) {
        fd[i] = abre(&eventos[i], filho);
        if (fd[i] < 0) {
            fprintf(stderr, "%s: %s\n", eventos[i].nome, strerror(errno));
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (write(canal[1], "x", 1) != 1) {
        perror("write");
        return 2;
    }
    close(canal[1]);
    int status = 0;
    waitpid(filho, &status, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double contado[N] = {0};
    double habilitado[N] = {0};
    double rodando[N] = {0};
    for (int i = 0; i < N; ++i) {
        struct leitura l = {0, 0, 0};
        if (fd[i] >= 0 && read(fd[i], &l, sizeof l) == (ssize_t)sizeof l) {
            contado[i] = (double)l.valor;
            habilitado[i] = (double)l.habilitado;
            rodando[i] = (double)l.rodando;
        }
    }

    const double instr = contado[0];
    const double seg = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    fprintf(stderr, "contadores (modo usuario, PMU dos P-cores):\n");
    for (int i = 0; i < N; ++i) {
        fprintf(stderr, "  %-14s %16.0f   por 1000 instr: %10.3f   tempo contado: %5.1f%%\n",
                eventos[i].nome, contado[i], instr > 0 ? 1000.0 * contado[i] / instr : 0.0,
                habilitado[i] > 0 ? 100.0 * rodando[i] / habilitado[i] : 0.0);
    }
    fprintf(stderr, "  IPC %.3f | tempo total %.6f s\n",
            contado[1] > 0 ? instr / contado[1] : 0.0, seg);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
