#!/bin/bash
# Mede o speedup do treino em CPU para cada combinacao de lote e numero de threads.
# Compile antes, no no de login, e submeta da raiz do repositorio:
#   sbatch scripts/escala_cpu.slurm ./barenn 10
#SBATCH --job-name=barenn-escala
#SBATCH --partition=pluton_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=escala_%j.out

BIN=${1:-./barenn}
REPS=${2:-10}
LOTES=${LOTES:-"32 64 128 256"}
export OMP_PROC_BIND=true
# Sem isso o libgomp faz as threads dormirem na barreira e o lote espera elas acordarem.
export OMP_WAIT_POLICY=active

echo "no: $(hostname)"
lscpu | grep "Model name"
echo "binario: $BIN | repeticoes: $REPS | OMP_SCHEDULE: ${OMP_SCHEDULE:-static}"

# Um thread por nucleo: os P-cores sao as CPUs logicas pares de 0 a 14 e os E-cores, 16 a 31.
cpus() {
    case "$1" in
        16) echo "0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23" ;;
        24) echo "0,2,4,6,8,10,12,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" ;;
        *) seq -s, 0 2 $((2 * $1 - 2)) ;;
    esac
}

printf "%6s %4s %10s %10s %8s %10s\n" lote T media_s desvio_s speedup eficiencia
for B in $LOTES; do
    base=""
    for T in ${THREADS:-1 2 4 8 16 24}; do
        CPU=$(cpus "$T")
        stats=$(for r in $(seq 1 "$REPS"); do
                    BARENN_LOTE=$B OMP_NUM_THREADS=$T taskset -c "$CPU" "$BIN" |
                        awk '/Tempo de treino/ {print $4}'
                done | awk '{n++; s+=$1; q+=$1*$1} END {printf "%.6f %.6f", s/n, sqrt(q/n-(s/n)^2)}')
        media=${stats% *}
        desvio=${stats#* }
        [ -z "$base" ] && base=$media
        awk -v b="$B" -v t="$T" -v m="$media" -v d="$desvio" -v r="$base" \
            'BEGIN {printf "%6d %4d %10.6f %10.6f %8.2f %9.1f%%\n", b, t, m, d, r/m, 100*r/m/t}'
    done
done
