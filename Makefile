CXX      ?= g++
# fp-contract=off: sem isso o compilador funde mul+add em FMA e o resultado muda.
CXXFLAGS ?= -O3 -std=c++17 -march=x86-64-v3 -ffp-contract=off -fopenmp

.PHONY: all clean

all: barenn barenn_hash

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Mesma compilacao, imprimindo tambem o hash dos pesos treinados.
barenn_hash: barenn.cpp
	$(CXX) $(CXXFLAGS) -DBARENN_HASH $< -o $@

clean:
	rm -f barenn barenn_hash
