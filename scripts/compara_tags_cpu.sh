#!/bin/bash
# Compara as tags de CPU em um job so, intercalando as execucoes.
# O clock efetivo do P-core muda de um job para outro (4.76 a 5.5 GHz no compute20), entao
# comparar tempos de jobs diferentes engana. Intercalar expoe todas as tags ao mesmo clock,
# e os ciclos nao dependem do clock.
# Uso, em um no de GPU: scripts/compara_tags_cpu.sh <pasta_com_as_tags> <contadores> <rodadas>
PASTA=${1:?pasta das tags}
CONT=${2:?binario contadores}
RODADAS=${3:-20}
TAGS="v0.1-baseline v0.2-sem-copias v0.3-sem-gradiente-morto v0.4-layout-simd v0.5-paralelismo-minilote"
export OMP_PROC_BIND=true OMP_WAIT_POLICY=active BARENN_LOTE=128 OMP_NUM_THREADS=1

SAIDA=$(mktemp -d)
for r in $(seq 1 "$RODADAS"); do
    for t in $TAGS; do
        s=$(cd "$PASTA/$t" && taskset -c 8 "$CONT" ./barenn 2>&1)
        treino=$(echo "$s" | awk '/Tempo de treino/ {print $4}')
        ciclos=$(echo "$s" | awk '/  ciclos / {print $2}')
        ref=$(echo "$s" | awk '/  ciclos_ref / {print $2}')
        instr=$(echo "$s" | awk '/instrucoes/ {print $2}')
        l1=$(echo "$s" | awk '/falhas_L1D/ {print $2}')
        echo "$treino $ciclos $ref $instr $l1" >> "$SAIDA/$t"
    done
done

echo "no: $(hostname) | rodadas: $RODADAS | lote 128 nas tags que ja tem lote, um P-core"
printf "%-28s %10s %10s %10s %14s %14s %7s %8s %11s\n" \
    tag mediana_s min_s max_s ciclos instrucoes IPC GHz_efet falhas_L1D
for t in $TAGS; do
    sort -n "$SAIDA/$t" | awk -v t="$t" '{a[NR]=$1; c+=$2; r+=$3; i+=$4; l+=$5}
        END {m=a[int((NR+1)/2)]; printf "%-28s %10.4f %10.4f %10.4f %14.0f %14.0f %7.3f %8.2f %11.0f\n",
             t, m, a[1], a[NR], c/NR, i/NR, i/c, 2.0*c/r, l/NR}'
done
rm -rf "$SAIDA"
