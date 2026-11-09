#!/usr/bin/env python3
"""Resume o SQLite do Nsight Systems: tempo por kernel, copias e tempo ocioso da GPU.

Uso: python3 scripts/analisa_nsys.py rastro.sqlite
"""
import sqlite3
import sys
from collections import defaultdict


def main(caminho):
    db = sqlite3.connect(caminho)
    kernels = db.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id ORDER BY k.start"
    ).fetchall()
    tabelas = {t[0] for t in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    copias = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabelas:
        copias = db.execute(
            "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
        ).fetchall()

    por_nome = defaultdict(lambda: [0, 0])
    for inicio, fim, nome in kernels:
        por_nome[nome][0] += 1
        por_nome[nome][1] += fim - inicio
    tempo_kernels = sum(v[1] for v in por_nome.values())
    tempo_copias = sum(fim - inicio for inicio, fim, _ in copias)
    bytes_copias = sum(b for _, _, b in copias)

    # Ocupacao da GPU: uniao dos intervalos de kernels e copias dentro da janela.
    intervalos = sorted([(k[0], k[1]) for k in kernels] + [(c[0], c[1]) for c in copias])
    if not intervalos:
        print("sem atividade de GPU no rastro")
        return
    inicio_janela = intervalos[0][0]
    fim_janela = max(fim for _, fim in intervalos)
    janela = fim_janela - inicio_janela
    ocupado = 0
    atual_inicio, atual_fim = intervalos[0]
    for inicio, fim in intervalos[1:]:
        if inicio > atual_fim:
            ocupado += atual_fim - atual_inicio
            atual_inicio, atual_fim = inicio, fim
        else:
            atual_fim = max(atual_fim, fim)
    ocupado += atual_fim - atual_inicio

    print(f"janela capturada: {janela / 1e9:.6f} s")
    print(f"kernels: {len(kernels)} em {tempo_kernels / 1e9:.6f} s "
          f"({100 * tempo_kernels / janela:.1f}% da janela)")
    print(f"copias: {len(copias)} em {tempo_copias / 1e9:.6f} s "
          f"({100 * tempo_copias / janela:.1f}% da janela), {bytes_copias / 1e6:.3f} MB")
    print(f"GPU sem trabalho: {(janela - ocupado) / 1e9:.6f} s "
          f"({100 * (janela - ocupado) / janela:.1f}% da janela)")
    if kernels:
        print(f"tempo medio de kernel: {tempo_kernels / len(kernels):.0f} ns")
    if copias:
        print(f"tempo medio de copia: {tempo_copias / len(copias):.0f} ns")
    print(f"operacoes de GPU por ms: {(len(kernels) + len(copias)) / (janela / 1e6):.1f}")
    print("por kernel (nome, instancias, total s, medio ns):")
    for nome, (n, total) in sorted(por_nome.items(), key=lambda x: -x[1][1]):
        print(f"  {nome}: {n}, {total / 1e9:.6f}, {total / n:.0f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
