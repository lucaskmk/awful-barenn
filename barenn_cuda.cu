// Treino em mini-lotes na GPU, com a mesma conta da versão em CPU.
// As amostras de um lote são independentes, então os blocos ocupam vários SMs ao mesmo tempo.
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

// Estado de cada amostra do lote, em linhas contíguas.
struct EstadoLote {
    float* y1;
    float* y2;
    float* erro;
    float* g3;
    float* d2;
    float* g2;
    float* d1;
#ifdef BARENN_OCUPACAO
    unsigned long long* ciclos_sm;
    unsigned int* blocos_sm;
#endif
};

// Forward e backward de um pedaço do lote: uma thread por par amostra e neurônio.
// Só escreve no estado do lote. Os pesos só mudam depois, em kernel_atualiza.
__global__ void kernel_lote(const float* __restrict__ imagens, const int* __restrict__ rotulos,
                            int tam, int amostras_bloco, Rede rede, EstadoLote l,
                            float* __restrict__ perdas) {
    const int tid = threadIdx.x;
    const int base = blockIdx.x * amostras_bloco;
#ifdef BARENN_OCUPACAO
    const long long ciclos_inicio = tid == 0 ? clock64() : 0;
#endif

    // Forward da camada 1.
    {
        const int s = base + tid / neuronios1;
        const int i = tid % neuronios1;
        if (s < tam) {
            const float* x = imagens + s * entradas1;
            const float* w = rede.w1 + i * entradas1;
            float z = rede.b1[i];
            for (int j = 0; j < entradas1; ++j) {
                z += x[j] * w[j];
            }
            l.y1[s * neuronios1 + i] = 0.0f < z ? z : 0.0f;
        }
    }
    __syncthreads();

    // Forward da camada 2.
    if (tid < amostras_bloco * neuronios2) {
        const int s = base + tid / neuronios2;
        const int i = tid % neuronios2;
        if (s < tam) {
            const float* x = l.y1 + s * entradas2;
            const float* w = rede.w2 + i * entradas2;
            float z = rede.b2[i];
            for (int j = 0; j < entradas2; ++j) {
                z += x[j] * w[j];
            }
            l.y2[s * neuronios2 + i] = 0.0f < z ? z : 0.0f;
        }
    }
    __syncthreads();

    // Forward da camada 3 e erro de cada escore em relação ao alvo.
    if (tid < amostras_bloco * neuronios3) {
        const int s = base + tid / neuronios3;
        const int i = tid % neuronios3;
        if (s < tam) {
            const float* x = l.y2 + s * entradas3;
            const float* w = rede.w3 + i * entradas3;
            float z = rede.b3[i];
            for (int j = 0; j < entradas3; ++j) {
                z += x[j] * w[j];
            }
            l.erro[s * neuronios3 + i] = z - (i == rotulos[s] ? 1.0f : 0.0f);
        }
    }
    __syncthreads();

    // A perda soma as classes em ordem, uma thread por amostra.
    if (tid < amostras_bloco) {
        const int s = base + tid;
        if (s < tam) {
            const float* e = l.erro + s * neuronios3;
            float total = 0.0f;
            for (int i = 0; i < neuronios3; ++i) {
                total += 0.5f * e[i] * e[i];
            }
            perdas[s] = total;
        }
    }

    // Backward da camada 3: gradiente de cada entrada.
    if (tid < amostras_bloco * entradas3) {
        const int s = base + tid / entradas3;
        const int c = tid % entradas3;
        if (s < tam) {
            const float* e = l.erro + s * neuronios3;
            float g = 0.0f;
            for (int r = 0; r < neuronios3; ++r) {
                g += e[r] * rede.w3[r * entradas3 + c];
            }
            l.g3[s * entradas3 + c] = g;
        }
    }
    __syncthreads();

    // Delta da camada 2. Na ReLU, o gradiente passa só se a soma foi positiva.
    if (tid < amostras_bloco * neuronios2) {
        const int s = base + tid / neuronios2;
        const int i = tid % neuronios2;
        if (s < tam) {
            const int p = s * neuronios2 + i;
            l.d2[p] = l.g3[p] * (l.y2[p] > 0.0f ? 1.0f : 0.0f);
        }
    }
    __syncthreads();

    // Gradiente de entrada da camada 2.
    {
        const int s = base + tid / entradas2;
        const int c = tid % entradas2;
        if (s < tam) {
            const float* d = l.d2 + s * neuronios2;
            float g = 0.0f;
            for (int r = 0; r < neuronios2; ++r) {
                g += d[r] * rede.w2[r * entradas2 + c];
            }
            l.g2[s * entradas2 + c] = g;
        }
    }
    __syncthreads();

    // Delta da camada 1.
    {
        const int s = base + tid / neuronios1;
        const int i = tid % neuronios1;
        if (s < tam) {
            const int p = s * neuronios1 + i;
            l.d1[p] = l.g2[p] * (l.y1[p] > 0.0f ? 1.0f : 0.0f);
        }
    }

#ifdef BARENN_OCUPACAO
    if (tid == 0) {
        unsigned int smid;
        asm("mov.u32 %0, %smid;" : "=r"(smid));
        atomicAdd(&l.ciclos_sm[smid], static_cast<unsigned long long>(clock64() - ciclos_inicio));
        atomicAdd(&l.blocos_sm[smid], 1u);
    }
#endif
}

// Soma o gradiente do lote e move os pesos. Cada bloco cuida de um neurônio e cada thread
// de um peso, somando as amostras em ordem: a conta de cada peso é a mesma da CPU.
__global__ void kernel_atualiza(Rede rede, EstadoLote l, const float* __restrict__ imagens,
                                int tam, float taxa) {
    const int bloco = blockIdx.x;
    const int t = threadIdx.x;

    const float* deltas;
    const float* x;
    float* pesos;
    float* vies;
    int i;
    int saidas;
    int entradas_camada;
    if (bloco < neuronios1) {
        i = bloco;
        deltas = l.d1;
        x = imagens;
        pesos = rede.w1;
        vies = rede.b1;
        saidas = neuronios1;
        entradas_camada = entradas1;
    } else if (bloco < neuronios1 + neuronios2) {
        i = bloco - neuronios1;
        deltas = l.d2;
        x = l.y1;
        pesos = rede.w2;
        vies = rede.b2;
        saidas = neuronios2;
        entradas_camada = entradas2;
    } else {
        i = bloco - neuronios1 - neuronios2;
        deltas = l.erro;
        x = l.y2;
        pesos = rede.w3;
        vies = rede.b3;
        saidas = neuronios3;
        entradas_camada = entradas3;
    }

    // O gradiente do peso é delta vezes a entrada.
    if (t < entradas_camada) {
        float g = 0.0f;
        for (int s = 0; s < tam; ++s) {
            g += deltas[s * saidas + i] * x[s * entradas_camada + t];
        }
        pesos[i * entradas_camada + t] -= taxa * g;
    }
    // O gradiente do viés é o próprio delta.
    if (t == 0) {
        float total = 0.0f;
        for (int s = 0; s < tam; ++s) {
            total += deltas[s * saidas + i];
        }
        vies[i] -= taxa * total;
    }
}

// Forward de todas as imagens do teste, para a avaliação no final.
__global__ void kernel_prever(const float* __restrict__ imagens, int amostras, Rede rede,
                              float* __restrict__ y1, float* __restrict__ y2,
                              float* __restrict__ escores) {
    const int t = threadIdx.x;
    for (int s = 0; s < amostras; ++s) {
        const float* x = imagens + s * entradas1;
        {
            const float* w = rede.w1 + t * entradas1;
            float z = rede.b1[t];
            for (int j = 0; j < entradas1; ++j) {
                z += x[j] * w[j];
            }
            y1[t] = 0.0f < z ? z : 0.0f;
        }
        __syncthreads();
        if (t < neuronios2) {
            const float* w = rede.w2 + t * entradas2;
            float z = rede.b2[t];
            for (int j = 0; j < entradas2; ++j) {
                z += y1[j] * w[j];
            }
            y2[t] = 0.0f < z ? z : 0.0f;
        }
        __syncthreads();
        if (t < neuronios3) {
            const float* w = rede.w3 + t * entradas3;
            float z = rede.b3[t];
            for (int j = 0; j < entradas3; ++j) {
                z += y2[j] * w[j];
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

        const int lote = Barenn::le_lote(treino.tamanho());
        const float taxa = Barenn::taxa_do_lote(lote);
        // Cada bloco cuida deste número de amostras, com 32 threads por amostra.
        int amostras_bloco = 8;
        if (const char* valor = getenv("BARENN_AMOSTRAS_BLOCO")) {
            amostras_bloco = atoi(valor);
        }
        if (amostras_bloco < 1 || amostras_bloco * neuronios1 > 1024) {
            throw runtime_error("BARENN_AMOSTRAS_BLOCO deve estar entre 1 e 32.");
        }
        const int threads_bloco = amostras_bloco * neuronios1;

        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Rede rede{aloca_gpu<float>(neuronios1 * entradas1), aloca_gpu<float>(neuronios1),
                  aloca_gpu<float>(neuronios2 * entradas2), aloca_gpu<float>(neuronios2),
                  aloca_gpu<float>(neuronios3 * entradas3), aloca_gpu<float>(neuronios3)};
        inicializa_camada(rede.w1, rede.b1, entradas1, neuronios1);
        inicializa_camada(rede.w2, rede.b2, entradas2, neuronios2);
        inicializa_camada(rede.w3, rede.b3, entradas3, neuronios3);

        EstadoLote estado{};
        estado.y1 = aloca_gpu<float>(lote * neuronios1);
        estado.y2 = aloca_gpu<float>(lote * neuronios2);
        estado.erro = aloca_gpu<float>(lote * neuronios3);
        estado.g3 = aloca_gpu<float>(lote * entradas3);
        estado.d2 = aloca_gpu<float>(lote * neuronios2);
        estado.g2 = aloca_gpu<float>(lote * entradas2);
        estado.d1 = aloca_gpu<float>(lote * neuronios1);
#ifdef BARENN_OCUPACAO
        constexpr int maximo_sms = 256;
        estado.ciclos_sm = aloca_gpu<unsigned long long>(maximo_sms);
        estado.blocos_sm = aloca_gpu<unsigned int>(maximo_sms);
        checa(cudaMemset(estado.ciclos_sm, 0, maximo_sms * sizeof(unsigned long long)), "cudaMemset");
        checa(cudaMemset(estado.blocos_sm, 0, maximo_sms * sizeof(unsigned int)), "cudaMemset");
#endif

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
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (int ini = 0; ini < amostras; ini += lote) {
                const int tam = min(lote, amostras - ini);
                const int blocos = (tam + amostras_bloco - 1) / amostras_bloco;
                const float* imagens_lote = imagens_gpu + ini * entradas1;
                kernel_lote<<<blocos, threads_bloco>>>(
                    imagens_lote, rotulos_gpu + ini, tam, amostras_bloco, rede, estado,
                    perdas_gpu + static_cast<size_t>(epoca) * amostras + ini);
                kernel_atualiza<<<neuronios1 + neuronios2 + neuronios3, entradas1>>>(
                    rede, estado, imagens_lote, tam, taxa);
            }
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
#ifdef BARENN_OCUPACAO
        {
            vector<unsigned long long> ciclos(maximo_sms);
            vector<unsigned int> blocos(maximo_sms);
            recebe(ciclos.data(), estado.ciclos_sm, ciclos.size());
            recebe(blocos.data(), estado.blocos_sm, blocos.size());
            int sms_usados = 0;
            unsigned long long soma_ciclos = 0;
            unsigned long long soma_blocos = 0;
            for (int sm = 0; sm < maximo_sms; ++sm) {
                sms_usados += blocos[sm] > 0;
                soma_ciclos += ciclos[sm];
                soma_blocos += blocos[sm];
            }
            cerr << "ocupacao: SMs usados " << sms_usados << " | blocos " << soma_blocos
                 << " | ciclos de bloco somados " << soma_ciclos << "\n";
            for (int sm = 0; sm < maximo_sms; ++sm) {
                if (blocos[sm] > 0) {
                    cerr << "  sm " << sm << ": blocos " << blocos[sm]
                         << " ciclos " << ciclos[sm] << "\n";
                }
            }
        }
#endif
        float* escores_gpu = aloca_gpu<float>(static_cast<size_t>(amostras_teste) * neuronios3);
        kernel_prever<<<1, neuronios1>>>(teste_gpu, amostras_teste, rede, estado.y1, estado.y2,
                                         escores_gpu);
        vector<float> escores(static_cast<size_t>(amostras_teste) * neuronios3);
        recebe(escores.data(), escores_gpu, escores.size());
        Barenn::avaliar([&](size_t i) { return escores.data() + i * neuronios3; }, teste);

        for (float* p : {rede.w1, rede.b1, rede.w2, rede.b2, rede.w3, rede.b3, estado.y1,
                         estado.y2, estado.erro, estado.g3, estado.d2, estado.g2, estado.d1,
                         imagens_gpu, teste_gpu, perdas_gpu, escores_gpu}) {
            checa(cudaFree(p), "cudaFree");
        }
        checa(cudaFree(rotulos_gpu), "cudaFree");
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
