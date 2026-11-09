#!/bin/bash
# Compara a saida de um binario com uma referencia, ignorando o tempo.
# Sem argumentos treina com lote 1 e compara com o legado, bit a bit.
# Uso: scripts/valida.sh ./barenn_hash
#      BARENN_LOTE=128 scripts/valida.sh ./barenn_hash resultados/referencia_lote128.txt
set -u
BIN=${1:-./barenn_hash}
REF=${2:-resultados/referencia_legado.txt}
export BARENN_LOTE=${BARENN_LOTE:-1}

saida=$(mktemp)
"$BIN" | grep -v "Tempo de treino" > "$saida"

if diff -u "$REF" "$saida" > /dev/null; then
    echo "OK: saida identica a referencia ($(grep 'Hash dos pesos' "$saida"))"
    rm -f "$saida"
    exit 0
fi
echo "DIFERENTE da referencia:"
diff -u "$REF" "$saida" | head -40
rm -f "$saida"
exit 1
