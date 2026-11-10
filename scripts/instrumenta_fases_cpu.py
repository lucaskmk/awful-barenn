#!/usr/bin/env python3
"""Instrumenta o barenn.cpp da tag v0.5 para medir o tempo por fase de cada lote.

Serve so para medir. Use em uma copia:
  git archive v0.5-paralelismo-minilote | tar -x -C /tmp/fases
  python3 scripts/instrumenta_fases_cpu.py /tmp/fases/barenn.cpp
  make barenn e rode ./barenn com OMP_WAIT_POLICY=active. O resumo sai no stderr.
"""
import sys

caminho = sys.argv[1]
s = open(caminho, encoding="utf8", newline="").read().replace("\r\n", "\n")


def troca(velho, novo):
    global s
    if velho not in s:
        raise SystemExit("nao achei: " + velho[:60])
    s = s.replace(velho, novo, 1)


troca("#pragma omp for schedule(runtime)\n", "double f0 = omp_get_wtime();\n#pragma omp for schedule(runtime)\n")
troca("                // Com os gradientes do lote prontos", "                double f1 = omp_get_wtime();\n                // Com os gradientes do lote prontos")
troca("#pragma omp barrier\n#pragma omp single\n", "double f2 = omp_get_wtime();\n#pragma omp barrier\ndouble f3 = omp_get_wtime();\n#pragma omp single\n")
troca("                    l3.SincronizaEspelho();\n                }\n", "                    l3.SincronizaEspelho();\n                }\n                double f4 = omp_get_wtime();\n                if (omp_get_thread_num() == 0) { FASES[0] += f1 - f0; FASES[1] += f2 - f1; FASES[2] += f3 - f2; FASES[3] += f4 - f3; FASES[4] += 1; }\n")
troca("int main(int argc", "static double FASES[5];\nint main(int argc")
troca('        cout << "Tempo de treino "', '        cerr << "por lote (us): amostras+barreira " << 1e6 * FASES[0] / FASES[4] << " | atualiza " << 1e6 * FASES[1] / FASES[4] << " | barreira " << 1e6 * FASES[2] / FASES[4] << " | espelho+barreira " << 1e6 * FASES[3] / FASES[4] << " | lotes " << FASES[4] << "\\n";\n        cout << "Tempo de treino "')
open(caminho, "w", encoding="utf8", newline="").write(s)
print("ok")
