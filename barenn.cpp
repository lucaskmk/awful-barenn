#include <chrono>
#include <filesystem>

#include "comum.h"

#include <omp.h>

namespace Barenn {

// A maior camada da rede tem 32 neurônios.
constexpr int maximo_neuronios = 32;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    bool precisa_grad_entrada;
    // Pesos por coluna: o Forward percorre colunas e a soma de cada neurônio
    // cai em uma pista de SIMD, sem mudar a ordem das parcelas.
    vector<float> pesos_coluna;
    // Espelho por linha, usado só no gradiente de entrada, que percorre linhas.
    // A primeira camada não precisa desse gradiente e não mantém o espelho.
    vector<float> pesos_linha;
    vector<float> bias;
    // Estado de cada amostra do lote, em linhas contíguas, sem alocação no treino.
    vector<float> soma;
    vector<float> saida;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, int lote, bool is_final = false,
               bool grad_entrada = true)
        : in_features(input_size), out_features(output_size), final(is_final),
          precisa_grad_entrada(grad_entrada),
          pesos_coluna(output_size * input_size),
          pesos_linha(grad_entrada ? output_size * input_size : 0),
          bias(output_size, 0), soma(lote * output_size), saida(lote * output_size),
          delta(lote * output_size), grad_in(lote * input_size) {
        inicializa_pesos(in_features, out_features, [this](int i, int j, float valor) {
            pesos_coluna[j * out_features + i] = valor;
            if (precisa_grad_entrada) {
                pesos_linha[i * in_features + j] = valor;
            }
        });
    }

    const float* Saidas() const { return saida.data(); }

    // Calcula a amostra s do lote. Os pesos não mudam durante o lote.
    const float* Forward(const float* input, int s) {
        // Entrada, pesos e saída nunca se sobrepõem: restrict informa isso ao compilador.
        const float* __restrict x = input;
        float* __restrict z = soma.data() + s * out_features;
        float* __restrict y = saida.data() + s * out_features;
        copy(bias.begin(), bias.end(), z);
        // Percorre as colunas por fora: cada z[i] recebe as parcelas na mesma ordem.
        for (int j = 0; j < in_features; ++j) {
            const float* __restrict w = pesos_coluna.data() + j * out_features;
            const float xj = x[j];
            for (int i = 0; i < out_features; ++i) {
                z[i] += xj * w[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < out_features; ++i) {
            y[i] = final ? z[i] : (0.0f < z[i] ? z[i] : 0.0f);
        }
        return y;
    }

    const float* Backward(const float* grad_out, int s) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        const float* __restrict grad = grad_out;
        const float* __restrict z = soma.data() + s * out_features;
        float* __restrict d = delta.data() + s * out_features;
        for (int i = 0; i < out_features; ++i) {
            d[i] = final ? grad[i] : grad[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        // Percorre as linhas por fora: cada g[c] recebe as parcelas na mesma ordem.
        float* __restrict g = grad_in.data() + s * in_features;
        if (precisa_grad_entrada) {
            fill(g, g + in_features, 0.0f);
            for (int r = 0; r < out_features; ++r) {
                const float* __restrict w = pesos_linha.data() + r * in_features;
                const float dr = d[r];
                for (int c = 0; c < in_features; ++c) {
                    g[c] += dr * w[c];
                }
            }
        }
        return g;
    }

    // Soma os gradientes das tam amostras e só então move os pesos. Cada peso
    // recebe as parcelas na ordem das amostras, qualquer que seja a divisão do trabalho.
    void Atualiza(const float* entradas_lote, int tam, float learning_rate) {
#pragma omp for schedule(static) nowait
        for (int j = 0; j < in_features; ++j) {
            // Acumulador na pilha da thread: colunas vizinhas são de threads diferentes
            // e, em um vetor comum, dividiriam a mesma linha de cache.
            alignas(64) float g[maximo_neuronios] = {};
            // O gradiente do peso é delta vezes a entrada.
            for (int s = 0; s < tam; ++s) {
                const float* __restrict d = delta.data() + s * out_features;
                const float xj = entradas_lote[s * in_features + j];
                for (int i = 0; i < out_features; ++i) {
                    g[i] += d[i] * xj;
                }
            }
            float* __restrict w = pesos_coluna.data() + j * out_features;
            for (int i = 0; i < out_features; ++i) {
                w[i] -= learning_rate * g[i];
            }
        }
        // O gradiente do viés é o próprio delta.
#pragma omp for schedule(static) nowait
        for (int i = 0; i < out_features; ++i) {
            float total = 0.0f;
            for (int s = 0; s < tam; ++s) {
                total += delta[s * out_features + i];
            }
            bias[i] -= learning_rate * total;
        }
    }

    // Copia os pesos novos para o espelho por linha usado no gradiente de entrada.
    // Só pode rodar depois que todas as threads terminarem o Atualiza.
    void SincronizaEspelho() {
        if (!precisa_grad_entrada) {
            return;
        }
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                pesos_linha[i * in_features + j] = pesos_coluna[j * out_features + i];
            }
        }
    }

#ifdef BARENN_HASH
    uint64_t Hash(uint64_t h) const {
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                h = mistura_hash(h, pesos_coluna[j * out_features + i]);
            }
        }
        for (float b : bias) {
            h = mistura_hash(h, b);
        }
        return h;
    }
#endif
};

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        // Amostras por atualização. Com BARENN_LOTE=1, o treino é o SGD original bit a bit.
        int lote = 128;
        if (const char* valor = getenv("BARENN_LOTE")) {
            lote = atoi(valor);
        }
        if (lote < 1 || static_cast<size_t>(lote) > treino.tamanho()) {
            throw runtime_error("BARENN_LOTE deve estar entre 1 e o tamanho do treino.");
        }
        // Até 32 amostras a taxa original é estável. Acima disso ela cai com o lote,
        // para manter taxa x lote em 0.64, o limite medido antes de a perda divergir.
        const float taxa = lote <= 32 ? 0.02f : 0.02f * 32 / lote;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32, lote, false, false);
        Barenn::DenseLayer l2(32, 16, lote);
        Barenn::DenseLayer l3(16, Barenn::classes, lote, true);
        vector<float> gradiente(lote * Barenn::classes);
        vector<float> perda_amostra(treino.tamanho());
        array<double, Barenn::epocas> perdas{};

        // Sem OMP_SCHEDULE, cada thread fica com um pedaço fixo do lote.
        if (!getenv("OMP_SCHEDULE")) {
            omp_set_schedule(omp_sched_static, 0);
        }

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        // Uma única região paralela no treino inteiro: abrir uma por lote custaria caro.
#pragma omp parallel
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t ini = 0; ini < treino.tamanho(); ini += lote) {
                const int tam = static_cast<int>(min<size_t>(lote, treino.tamanho() - ini));
                // Dentro do lote os pesos não mudam: cada amostra é independente.
#pragma omp for schedule(runtime)
                for (int s = 0; s < tam; ++s) {
                    const float* imagem = treino.imagem(ini + s);
                    const float* escores = l3.Forward(l2.Forward(l1.Forward(imagem, s), s), s);
                    float* grad = gradiente.data() + s * Barenn::classes;
                    perda_amostra[ini + s] = Barenn::perda(escores, treino.rotulos[ini + s], grad);

                    // Propaga o gradiente da última camada até a primeira.
                    const float* grad3 = l3.Backward(grad, s);
                    const float* grad2 = l2.Backward(grad3, s);
                    l1.Backward(grad2, s);
                }
                // Com os gradientes do lote prontos, cada camada atualiza os próprios pesos.
                // Os laços dos três Atualiza não têm barreira: as camadas não se tocam.
                l1.Atualiza(treino.imagem(ini), tam, taxa);
                l2.Atualiza(l1.Saidas(), tam, taxa);
                l3.Atualiza(l2.Saidas(), tam, taxa);
#pragma omp barrier
#pragma omp single
                {
                    l2.SincronizaEspelho();
                    l3.SincronizaEspelho();
                }
            }
            // A perda é somada em ordem, então não depende do número de threads.
#pragma omp single
            {
                double perda_total = 0;
                for (float valor : perda_amostra) {
                    perda_total += valor;
                }
                perdas[epoca] = perda_total / treino.tamanho();
            }
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        Barenn::imprime_treino(treino.tamanho(), perdas, tempo.count());
#ifdef BARENN_HASH
        uint64_t hash_pesos = Barenn::hash_inicial;
        hash_pesos = l1.Hash(hash_pesos);
        hash_pesos = l2.Hash(hash_pesos);
        hash_pesos = l3.Hash(hash_pesos);
        Barenn::imprime_hash(hash_pesos);
#endif
        Barenn::avaliar([&](size_t i) {
            return l3.Forward(l2.Forward(l1.Forward(teste.imagem(i), 0), 0), 0);
        }, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
