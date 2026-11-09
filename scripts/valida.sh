#!/bin/bash
# Compara a saida de um binario com a referencia do legado, ignorando o tempo.
# Uso: scripts/valida.sh ./barenn_hash
set -u
BIN=${1:-./barenn_hash}
REF=${2:-resultados/referencia_legado.txt}

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
