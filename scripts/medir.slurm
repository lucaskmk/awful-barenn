#!/bin/bash
# Mede o tempo de treino e o tempo total de um binario, com repeticoes.
# Compile antes, no no de login, e submeta da raiz do repositorio:
#   sbatch scripts/medir.slurm ./barenn 10
#SBATCH --job-name=barenn-medir
#SBATCH --partition=pluton_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=medir_%j.out

BIN=${1:-./barenn}
REPS=${2:-10}
# Fixa o processo em um P-core: sem isso o escalonador alterna P- e E-cores.
CPU=${CPU:-8}

echo "no: $(hostname)"
lscpu | grep "Model name"
echo "binario: $BIN | repeticoes: $REPS | cpu: $CPU | OMP_NUM_THREADS: ${OMP_NUM_THREADS:-nao definido}"

: > tempos_$SLURM_JOB_ID.txt
for i in $(seq 1 "$REPS"); do
    inicio=$(date +%s%N)
    saida=$(taskset -c "$CPU" "$BIN")
    fim=$(date +%s%N)
    treino=$(echo "$saida" | awk '/Tempo de treino/ {print $4}')
    total=$(awk -v a="$inicio" -v b="$fim" 'BEGIN {printf "%.6f", (b-a)/1e9}')
    echo "$i $treino $total" | tee -a tempos_$SLURM_JOB_ID.txt
done

awk '{ n++; t+=$2; t2+=$2*$2; p+=$3; p2+=$3*$3;
       if (min==""||$2<min) min=$2; if ($2>max) max=$2 }
     END { printf "treino: media %.6f s | desvio %.6f s | min %.6f s | max %.6f s\n",
                  t/n, sqrt(t2/n-(t/n)^2), min, max;
           printf "total:  media %.6f s | desvio %.6f s\n", p/n, sqrt(p2/n-(p/n)^2) }' tempos_$SLURM_JOB_ID.txt
rm -f tempos_$SLURM_JOB_ID.txt
