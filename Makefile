CXX      ?= g++
# fp-contract=off: sem isso o compilador funde mul+add em FMA e o resultado muda.
CXXFLAGS ?= -O3 -std=c++17 -march=x86-64-v3 -ffp-contract=off -fopenmp

.PHONY: all clean

all: barenn barenn_hash contadores

barenn: barenn.cpp comum.h
	$(CXX) $(CXXFLAGS) $< -o $@

# Mesma compilacao, imprimindo tambem o hash dos pesos treinados.
barenn_hash: barenn.cpp comum.h
	$(CXX) $(CXXFLAGS) -DBARENN_HASH $< -o $@

# Conta eventos de hardware do programa medido, ja que o Franky nao tem perf.
contadores: scripts/contadores.c
	$(CC) -O2 $< -o $@

clean:
	rm -f barenn barenn_hash contadores
