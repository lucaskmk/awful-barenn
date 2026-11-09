CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17

.PHONY: all clean

all: barenn barenn_hash

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Mesma compilacao, imprimindo tambem o hash dos pesos treinados.
barenn_hash: barenn.cpp
	$(CXX) $(CXXFLAGS) -DBARENN_HASH $< -o $@

clean:
	rm -f barenn barenn_hash
