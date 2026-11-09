// Partes comuns às versões em CPU e em GPU: dados, perda, relatório e hash.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;
constexpr int epocas = 51;

// Lê o tamanho do lote. Com BARENN_LOTE=1, o treino é o SGD original bit a bit.
inline int le_lote(size_t amostras) {
    int lote = 128;
    if (const char* valor = getenv("BARENN_LOTE")) {
        lote = atoi(valor);
    }
    if (lote < 1 || static_cast<size_t>(lote) > amostras) {
        throw runtime_error("BARENN_LOTE deve estar entre 1 e o tamanho do treino.");
    }
    return lote;
}

// Até 32 amostras a taxa original é estável. Acima disso ela cai com o lote,
// para manter taxa x lote em 0.64, o limite medido antes de a perda divergir.
inline float taxa_do_lote(int lote) {
    return lote <= 32 ? 0.02f : 0.02f * 32 / lote;
}

// As imagens ficam em um único bloco, uma linha de 64 atributos por amostra.
struct Dados {
    vector<float> imagens;
    vector<int> rotulos;

    size_t tamanho() const { return rotulos.size(); }
    const float* imagem(size_t i) const { return imagens.data() + i * entradas; }
};

inline Dados ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    Dados dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        int lidos = 0;
        int rotulo = -1;
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            dados.imagens.push_back(valor / 16.0f);
            ++lidos;
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (lidos != entradas || !(campos >> rotulo) ||
            rotulo < 0 || rotulo >= classes || !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.rotulos.push_back(rotulo);
    }
    if (arquivo.bad() || dados.rotulos.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Sorteia os pesos de uma camada na ordem [i][j], a mesma do código original,
// e entrega cada valor a quem chamou. A semente fixa torna a inicialização reproduzível.
template <typename Guardar>
void inicializa_pesos(int entradas_camada, int saidas_camada, Guardar guardar) {
    mt19937 gen(1337);
    uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    for (int i = 0; i < saidas_camada; ++i) {
        for (int j = 0; j < entradas_camada; ++j) {
            guardar(i, j, distribution(gen));
        }
    }
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
inline float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

// prever(i) devolve um ponteiro para os dez escores da imagem i do teste.
template <typename Prever>
void avaliar(Prever prever, const Dados& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.tamanho(); ++i) {
        const int rotulo = teste.rotulos[i];
        const float* escores = prever(i);
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores, escores + classes) - escores);
        bool acertou = previsto == rotulo;
        acertos += acertou;
        ++confusoes[rotulo][previsto];
        perda_total += perda(escores, rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            exemplos.push_back({i + 1, rotulo, previsto});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.tamanho()
         << "% | " << acertos << " acertos em " << teste.tamanho() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.tamanho() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

// Imprime a perda das épocas e o tempo. A medição termina antes de imprimir.
inline void imprime_treino(size_t amostras, const array<double, epocas>& perdas, double tempo) {
    cout << fixed << setprecision(4) << "TREINO | " << amostras << " imagens\n";
    for (int epoca = 0; epoca < epocas; epoca += 5) {
        cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
             << " | perda media " << perdas[epoca] << "\n";
    }
    cout << "Tempo de treino " << setprecision(6) << tempo << " s\n" << setprecision(4);
}

// FNV-1a sobre os bits de pesos e vieses, em ordem canônica [i][j].
constexpr uint64_t hash_inicial = 1469598103934665603ULL;

inline uint64_t mistura_hash(uint64_t h, float valor) {
    uint32_t bits;
    memcpy(&bits, &valor, sizeof bits);
    return (h ^ bits) * 1099511628211ULL;
}

inline void imprime_hash(uint64_t h) {
    cout << "Hash dos pesos " << hex << h << dec << "\n";
}

} // namespace Barenn
