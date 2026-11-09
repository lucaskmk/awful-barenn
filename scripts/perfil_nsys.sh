#!/bin/bash
# Perfila um binario de GPU com o Nsight Systems e resume o rastro em texto.
# Rode no no de GPU, da raiz do repositorio:
#   scripts/perfil_nsys.sh ./barenn_cuda resultados/06-cuda-inicial/nsys 2 1
# Os dois ultimos numeros sao opcionais: atraso e duracao da janela de captura, em segundos.
set -eu
BIN=$1
PREFIXO=$2
ATRASO=${3:-}
DURACAO=${4:-}

source scripts/ambiente_cuda.sh
JANELA=()
[ -n "$ATRASO" ] && JANELA+=(--delay="$ATRASO")
[ -n "$DURACAO" ] && JANELA+=(--duration="$DURACAO")

# kill=none: ao fim da janela o programa continua, em vez de receber SIGTERM do nsys.
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true --kill=none \
    ${JANELA[@]+"${JANELA[@]}"} -o "$PREFIXO" "$BIN" > "$PREFIXO.saida.txt" 2>&1
nsys stats --force-export=true --format table \
    --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
    --report cuda_gpu_mem_size_sum --report cuda_api_sum \
    "$PREFIXO.nsys-rep" > "$PREFIXO.stats.txt" 2>&1
echo "gerado: $PREFIXO.stats.txt e $PREFIXO.sqlite"
