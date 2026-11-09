#!/bin/bash
# Mede o treino na GPU variando o tamanho do lote e as amostras por bloco.
# Rode em um no de GPU, da raiz do repositorio:
#   LOTES="32 128" BLOCOS="4 8 16" scripts/escala_gpu.sh ./barenn_cuda 10
BIN=${1:-./barenn_cuda}
REPS=${2:-10}
LOTES=${LOTES:-"32 64 128 256 512 1024"}
BLOCOS=${BLOCOS:-8}

echo "no: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,pstate --format=csv,noheader
echo "binario: $BIN | repeticoes: $REPS"
printf "%6s %8s %10s %10s\n" lote blocoamo media_s desvio_s
for B in $LOTES; do
    for C in $BLOCOS; do
        for r in $(seq 1 "$REPS"); do
            BARENN_LOTE=$B BARENN_AMOSTRAS_BLOCO=$C taskset -c 8 "$BIN" |
                awk '/Tempo de treino/ {print $4}'
        done | awk -v b="$B" -v c="$C" '{n++; s+=$1; q+=$1*$1}
            END {printf "%6d %8d %10.6f %10.6f\n", b, c, s/n, sqrt(q/n-(s/n)^2)}'
    done
done
