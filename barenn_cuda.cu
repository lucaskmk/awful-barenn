// Treino da rede na GPU, com a mesma conta da versão em CPU.
// Os dados ficam na GPU e cada época é um único kernel: o host só dispara e espera.
#include <chrono>
#include <filesystem>

#include <cuda_runtime.h>

#include "comum.h"

namespace {

constexpr int neuronios1 = 32;
constexpr int neuronios2 = 16;
constexpr int neuronios3 = Barenn::classes;
constexpr int entradas1 = Barenn::entradas;
constexpr int entradas2 = neuronios1;
constexpr int entradas3 = neuronios2;

void checa(cudaError_t erro, const char* onde) {
    if (erro != cudaSuccess) {
        throw runtime_error(string(onde) + ": " + cudaGetErrorString(erro));
    }
}

template <typename T>
T* aloca_gpu(size_t elementos) {
    T* ponteiro = nullptr;
    checa(cudaMalloc(&ponteiro, elementos * sizeof(T)), "cudaMalloc");
    return ponteiro;
}

template <typename T>
void envia(T* destino, const T* origem, size_t elementos) {
    checa(cudaMemcpy(destino, origem, elementos * sizeof(T), cudaMemcpyHostToDevice),
          "cudaMemcpy para a GPU");
}

template <typename T>
void recebe(T* destino, const T* origem, size_t elementos) {
    checa(cudaMemcpy(destino, origem, elementos * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy para o host");
}

// Pesos e vieses da rede na memória da GPU, por linha: w[i * entradas + j].
struct Rede {
    float* w1;
    float* b1;
    float* w2;
    float* b2;
    float* w3;
    float* b3;
};

// Valores intermediários de uma amostra, compartilhados entre as threads do bloco.
struct Rascunho {
    float* y1;
    float* y2;
    float* erro;
    float* g3;
    float* d2;
    float* g2;
};

// Treina uma época inteira. Um bloco de 32 threads percorre as amostras em ordem e cada
// __syncthreads separa uma etapa da seguinte, na mesma sequência do treino em CPU.
__global__ void kernel_epoca(const float* __restrict__ imagens, const int* __restrict__ rotulos,
                             int amostras, Rede rede, Rascunho r,
                             float* __restrict__ perdas, float taxa) {
    const int t = threadIdx.x;
    for (int s = 0; s < amostras; ++s) {
        const float* x = imagens + s * entradas1;

        // Forward da camada 1: uma thread por neurônio.
        {
            const float* w = rede.w1 + t * entradas1;
            float z = rede.b1[t];
            for (int j = 0; j < entradas1; ++j) {
                z += x[j] * w[j];
            }
            r.y1[t] = 0.0f < z ? z : 0.0f;
        }
        __syncthreads();

        // Forward da camada 2.
        if (t < neuronios2) {
            const float* w = rede.w2 + t * entradas2;
            float z = rede.b2[t];
            for (int j = 0; j < entradas2; ++j) {
                z += r.y1[j] * w[j];
            }
            r.y2[t] = 0.0f < z ? z : 0.0f;
        }
        __syncthreads();

        // Forward da camada 3 e erro de cada escore em relação ao alvo.
        if (t < neuronios3) {
            const float* w = rede.w3 + t * entradas3;
            float z = rede.b3[t];
            for (int j = 0; j < entradas3; ++j) {
                z += r.y2[j] * w[j];
            }
            r.erro[t] = z - (t == rotulos[s] ? 1.0f : 0.0f);
        }
        __syncthreads();

        // A perda soma as classes em ordem, como na CPU.
        if (t == 0) {
            float total = 0.0f;
            for (int i = 0; i < neuronios3; ++i) {
                total += 0.5f * r.erro[i] * r.erro[i];
            }
            perdas[s] = total;
        }

        // Backward da camada 3: gradiente de entrada, com os pesos ainda antigos.
        if (t < entradas3) {
            float g = 0.0f;
            for (int i = 0; i < neuronios3; ++i) {
                g += r.erro[i] * rede.w3[i * entradas3 + t];
            }
            r.g3[t] = g;
        }
        __syncthreads();

        // Atualização da camada 3.
        if (t < neuronios3) {
            float* w = rede.w3 + t * entradas3;
            const float d = r.erro[t];
            for (int j = 0; j < entradas3; ++j) {
                w[j] -= taxa * (d * r.y2[j]);
            }
            rede.b3[t] -= taxa * d;
        }
        __syncthreads();

        // Backward da camada 2. Na ReLU, o gradiente passa só se a soma foi positiva.
        if (t < neuronios2) {
            r.d2[t] = r.g3[t] * (r.y2[t] > 0.0f ? 1.0f : 0.0f);
        }
        __syncthreads();

        {
            float g = 0.0f;
            for (int i = 0; i < neuronios2; ++i) {
                g += r.d2[i] * rede.w2[i * entradas2 + t];
            }
            r.g2[t] = g;
        }
        __syncthreads();

        // Atualização da camada 2.
        if (t < neuronios2) {
            float* w = rede.w2 + t * entradas2;
            const float d = r.d2[t];
            for (int j = 0; j < entradas2; ++j) {
                w[j] -= taxa * (d * r.y1[j]);
            }
            rede.b2[t] -= taxa * d;
        }

        // Backward e atualização da camada 1, que não precisa do gradiente de entrada.
        {
            float* w = rede.w1 + t * entradas1;
            const float d = r.g2[t] * (r.y1[t] > 0.0f ? 1.0f : 0.0f);
            for (int j = 0; j < entradas1; ++j) {
                w[j] -= taxa * (d * x[j]);
            }
            rede.b1[t] -= taxa * d;
        }
        __syncthreads();
    }
}

// Forward de todas as imagens do teste, para a avaliação no final.
__global__ void kernel_prever(const float* __restrict__ imagens, int amostras, Rede rede,
                              Rascunho r, float* __restrict__ escores) {
    const int t = threadIdx.x;
    for (int s = 0; s < amostras; ++s) {
        const float* x = imagens + s * entradas1;
        {
            const float* w = rede.w1 + t * entradas1;
            float z = rede.b1[t];
            for (int j = 0; j < entradas1; ++j) {
                z += x[j] * w[j];
            }
            r.y1[t] = 0.0f < z ? z : 0.0f;
        }
        __syncthreads();
        if (t < neuronios2) {
            const float* w = rede.w2 + t * entradas2;
            float z = rede.b2[t];
            for (int j = 0; j < entradas2; ++j) {
                z += r.y1[j] * w[j];
            }
            r.y2[t] = 0.0f < z ? z : 0.0f;
        }
        __syncthreads();
        if (t < neuronios3) {
            const float* w = rede.w3 + t * entradas3;
            float z = rede.b3[t];
            for (int j = 0; j < entradas3; ++j) {
                z += r.y2[j] * w[j];
            }
            escores[s * neuronios3 + t] = z;
        }
        __syncthreads();
    }
}

void inicializa_camada(float* w, float* b, int entradas_camada, int saidas_camada) {
    vector<float> iniciais(saidas_camada * entradas_camada);
    Barenn::inicializa_pesos(entradas_camada, saidas_camada, [&](int i, int j, float valor) {
        iniciais[i * entradas_camada + j] = valor;
    });
    envia(w, iniciais.data(), iniciais.size());
    const vector<float> zeros(saidas_camada, 0.0f);
    envia(b, zeros.data(), zeros.size());
}

#ifdef BARENN_HASH
uint64_t hash_camada(uint64_t h, const float* w, const float* b, int entradas_camada,
                     int saidas_camada) {
    vector<float> pesos(saidas_camada * entradas_camada);
    vector<float> vies(saidas_camada);
    recebe(pesos.data(), w, pesos.size());
    recebe(vies.data(), b, vies.size());
    for (float valor : pesos) {
        h = Barenn::mistura_hash(h, valor);
    }
    for (float valor : vies) {
        h = Barenn::mistura_hash(h, valor);
    }
    return h;
}
#endif

} // namespace

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
        const int amostras = static_cast<int>(treino.tamanho());
        const int amostras_teste = static_cast<int>(teste.tamanho());

        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Rede rede{aloca_gpu<float>(neuronios1 * entradas1), aloca_gpu<float>(neuronios1),
                  aloca_gpu<float>(neuronios2 * entradas2), aloca_gpu<float>(neuronios2),
                  aloca_gpu<float>(neuronios3 * entradas3), aloca_gpu<float>(neuronios3)};
        Rascunho rascunho{aloca_gpu<float>(neuronios1), aloca_gpu<float>(neuronios2),
                          aloca_gpu<float>(neuronios3), aloca_gpu<float>(entradas3),
                          aloca_gpu<float>(neuronios2), aloca_gpu<float>(entradas2)};
        inicializa_camada(rede.w1, rede.b1, entradas1, neuronios1);
        inicializa_camada(rede.w2, rede.b2, entradas2, neuronios2);
        inicializa_camada(rede.w3, rede.b3, entradas3, neuronios3);

        // O conjunto inteiro sobe uma vez, antes do treino.
        float* imagens_gpu = aloca_gpu<float>(treino.imagens.size());
        int* rotulos_gpu = aloca_gpu<int>(treino.rotulos.size());
        float* teste_gpu = aloca_gpu<float>(teste.imagens.size());
        envia(imagens_gpu, treino.imagens.data(), treino.imagens.size());
        envia(rotulos_gpu, treino.rotulos.data(), treino.rotulos.size());
        envia(teste_gpu, teste.imagens.data(), teste.imagens.size());

        // Perda de cada amostra em cada época, baixada só no final.
        float* perdas_gpu = aloca_gpu<float>(static_cast<size_t>(amostras) * Barenn::epocas);

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            kernel_epoca<<<1, neuronios1>>>(imagens_gpu, rotulos_gpu, amostras, rede, rascunho,
                                            perdas_gpu + static_cast<size_t>(epoca) * amostras,
                                            taxa);
        }
        checa(cudaDeviceSynchronize(), "treino");
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        vector<float> perda_amostra(static_cast<size_t>(amostras) * Barenn::epocas);
        recebe(perda_amostra.data(), perdas_gpu, perda_amostra.size());
        array<double, Barenn::epocas> perdas{};
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            for (int s = 0; s < amostras; ++s) {
                perda_total += perda_amostra[static_cast<size_t>(epoca) * amostras + s];
            }
            perdas[epoca] = perda_total / amostras;
        }

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        Barenn::imprime_treino(treino.tamanho(), perdas, tempo.count());
#ifdef BARENN_HASH
        uint64_t hash_pesos = Barenn::hash_inicial;
        hash_pesos = hash_camada(hash_pesos, rede.w1, rede.b1, entradas1, neuronios1);
        hash_pesos = hash_camada(hash_pesos, rede.w2, rede.b2, entradas2, neuronios2);
        hash_pesos = hash_camada(hash_pesos, rede.w3, rede.b3, entradas3, neuronios3);
        Barenn::imprime_hash(hash_pesos);
#endif
        float* escores_gpu = aloca_gpu<float>(static_cast<size_t>(amostras_teste) * neuronios3);
        kernel_prever<<<1, neuronios1>>>(teste_gpu, amostras_teste, rede, rascunho, escores_gpu);
        vector<float> escores(static_cast<size_t>(amostras_teste) * neuronios3);
        recebe(escores.data(), escores_gpu, escores.size());
        Barenn::avaliar([&](size_t i) { return escores.data() + i * neuronios3; }, teste);

        for (float* p : {rede.w1, rede.b1, rede.w2, rede.b2, rede.w3, rede.b3, rascunho.y1,
                         rascunho.y2, rascunho.erro, rascunho.g3, rascunho.d2, rascunho.g2,
                         imagens_gpu, teste_gpu, perdas_gpu, escores_gpu}) {
            checa(cudaFree(p), "cudaFree");
        }
        checa(cudaFree(rotulos_gpu), "cudaFree");
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
