CXX      ?= g++
# fp-contract=off: sem isso o compilador funde mul+add em FMA e o resultado muda.
CXXFLAGS ?= -O3 -std=c++17 -march=x86-64-v3 -ffp-contract=off -fopenmp

NVCC ?= nvcc
# fmad=false: o mesmo papel do fp-contract=off, sem FMA o resultado bate com a CPU.
# sm_86: a RTX A1000 do Franky. O login nao tem GPU, entao -arch=native nao serve.
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_86 -fmad=false

.PHONY: all cuda clean

all: barenn barenn_hash contadores

# Antes: source scripts/ambiente_cuda.sh
cuda: barenn_cuda barenn_cuda_hash

barenn_cuda: barenn_cuda.cu comum.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

barenn_cuda_hash: barenn_cuda.cu comum.h
	$(NVCC) $(NVCCFLAGS) -DBARENN_HASH $< -o $@

# Instrumentada: conta quantos blocos cada SM executou e por quantos ciclos.
barenn_cuda_ocupacao: barenn_cuda.cu comum.h
	$(NVCC) $(NVCCFLAGS) -DBARENN_OCUPACAO $< -o $@

barenn: barenn.cpp comum.h
	$(CXX) $(CXXFLAGS) $< -o $@

# Mesma compilacao, imprimindo tambem o hash dos pesos treinados.
barenn_hash: barenn.cpp comum.h
	$(CXX) $(CXXFLAGS) -DBARENN_HASH $< -o $@

# Conta eventos de hardware do programa medido, ja que o Franky nao tem perf.
contadores: scripts/contadores.c
	$(CC) -O2 $< -o $@

clean:
	rm -f barenn barenn_hash contadores barenn_cuda barenn_cuda_hash barenn_cuda_ocupacao
