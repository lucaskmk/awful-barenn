// Treino da rede na GPU, com a mesma conta da versão em CPU.
// Primeira migração: cada operação de cada camada é um kernel, e a perda fica no host.
#include <chrono>
#include <filesystem>

#include <cuda_runtime.h>

#include "comum.h"

namespace {

void checa(cudaError_t erro, const char* onde) {
    if (erro != cudaSuccess) {
        throw runtime_error(string(onde) + ": " + cudaGetErrorString(erro));
    }
}

float* aloca_gpu(int elementos) {
    float* ponteiro = nullptr;
    checa(cudaMalloc(&ponteiro, elementos * sizeof(float)), "cudaMalloc");
    return ponteiro;
}

void envia(float* destino, const float* origem, int elementos) {
    checa(cudaMemcpy(destino, origem, elementos * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy para a GPU");
}

void recebe(float* destino, const float* origem, int elementos) {
    checa(cudaMemcpy(destino, origem, elementos * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy para o host");
}

// Uma thread por neurônio. A soma percorre as entradas em ordem, como na CPU.
__global__ void kernel_forward(const float* __restrict__ x, const float* __restrict__ pesos,
                               const float* __restrict__ vies, float* __restrict__ soma,
                               float* __restrict__ saida, int entradas_camada, bool final) {
    const int i = threadIdx.x;
    const float* w = pesos + i * entradas_camada;
    float z = vies[i];
    for (int j = 0; j < entradas_camada; ++j) {
        z += x[j] * w[j];
    }
    soma[i] = z;
    // As camadas ocultas usam ReLU. A última devolve escores lineares.
    saida[i] = final ? z : (0.0f < z ? z : 0.0f);
}

// Na ReLU, o gradiente passa apenas quando a soma antes da ativação é positiva.
__global__ void kernel_delta(const float* __restrict__ grad, const float* __restrict__ soma,
                             float* __restrict__ delta, bool final) {
    const int i = threadIdx.x;
    delta[i] = final ? grad[i] : grad[i] * (soma[i] > 0.0f ? 1.0f : 0.0f);
}

// Uma thread por entrada. Usa os pesos anteriores à atualização.
__global__ void kernel_gradiente_entrada(const float* __restrict__ delta,
                                         const float* __restrict__ pesos,
                                         float* __restrict__ grad_in,
                                         int saidas_camada, int entradas_camada) {
    const int c = threadIdx.x;
    float g = 0.0f;
    for (int r = 0; r < saidas_camada; ++r) {
        g += delta[r] * pesos[r * entradas_camada + c];
    }
    grad_in[c] = g;
}

// SGD por amostra. O gradiente do peso é delta vezes a entrada.
__global__ void kernel_atualiza(const float* __restrict__ delta, const float* __restrict__ x,
                                float* __restrict__ pesos, float* __restrict__ vies,
                                int entradas_camada, float taxa) {
    const int i = threadIdx.x;
    float* w = pesos + i * entradas_camada;
    const float di = delta[i];
    for (int j = 0; j < entradas_camada; ++j) {
        w[j] -= taxa * (di * x[j]);
    }
    vies[i] -= taxa * di;
}

class CamadaGPU {
    int entradas_camada;
    int saidas_camada;
    bool final;
    bool precisa_grad_entrada;
    float* pesos;
    float* vies;
    float* soma;
    float* saida;
    float* delta;
    float* grad_in;

public:
    CamadaGPU(int n, int m, bool camada_final, bool grad_entrada)
        : entradas_camada(n), saidas_camada(m), final(camada_final),
          precisa_grad_entrada(grad_entrada), pesos(aloca_gpu(m * n)), vies(aloca_gpu(m)),
          soma(aloca_gpu(m)), saida(aloca_gpu(m)), delta(aloca_gpu(m)), grad_in(aloca_gpu(n)) {
        vector<float> iniciais(m * n);
        Barenn::inicializa_pesos(n, m, [&](int i, int j, float valor) {
            iniciais[i * n + j] = valor;
        });
        envia(pesos, iniciais.data(), m * n);
        const vector<float> zeros(m, 0.0f);
        envia(vies, zeros.data(), m);
    }

    ~CamadaGPU() {
        cudaFree(pesos);
        cudaFree(vies);
        cudaFree(soma);
        cudaFree(saida);
        cudaFree(delta);
        cudaFree(grad_in);
    }

    CamadaGPU(const CamadaGPU&) = delete;
    CamadaGPU& operator=(const CamadaGPU&) = delete;

    const float* Saida() const { return saida; }
    const float* GradEntrada() const { return grad_in; }

    void Forward(const float* entrada) {
        kernel_forward<<<1, saidas_camada>>>(entrada, pesos, vies, soma, saida,
                                             entradas_camada, final);
    }

    void Backward(const float* grad_saida, const float* entrada, float taxa) {
        kernel_delta<<<1, saidas_camada>>>(grad_saida, soma, delta, final);
        if (precisa_grad_entrada) {
            kernel_gradiente_entrada<<<1, entradas_camada>>>(delta, pesos, grad_in,
                                                             saidas_camada, entradas_camada);
        }
        kernel_atualiza<<<1, saidas_camada>>>(delta, entrada, pesos, vies, entradas_camada, taxa);
    }

    uint64_t Hash(uint64_t h) const {
        vector<float> p(saidas_camada * entradas_camada);
        vector<float> b(saidas_camada);
        recebe(p.data(), pesos, saidas_camada * entradas_camada);
        recebe(b.data(), vies, saidas_camada);
        for (float valor : p) {
            h = Barenn::mistura_hash(h, valor);
        }
        for (float valor : b) {
            h = Barenn::mistura_hash(h, valor);
        }
        return h;
    }
};

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

        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        CamadaGPU l1(Barenn::entradas, 32, false, false);
        CamadaGPU l2(32, 16, false, true);
        CamadaGPU l3(16, Barenn::classes, true, true);
        float* imagem_gpu = aloca_gpu(Barenn::entradas);
        float* gradiente_gpu = aloca_gpu(Barenn::classes);
        array<float, Barenn::classes> escores;
        array<float, Barenn::classes> gradiente;
        array<double, Barenn::epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < Barenn::epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (size_t s = 0; s < treino.tamanho(); ++s) {
                envia(imagem_gpu, treino.imagem(s), Barenn::entradas);
                l1.Forward(imagem_gpu);
                l2.Forward(l1.Saida());
                l3.Forward(l2.Saida());

                // A perda continua no host: os escores vão e o gradiente volta.
                recebe(escores.data(), l3.Saida(), Barenn::classes);
                perda_total += Barenn::perda(escores.data(), treino.rotulos[s], gradiente.data());
                envia(gradiente_gpu, gradiente.data(), Barenn::classes);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                l3.Backward(gradiente_gpu, l2.Saida(), taxa);
                l2.Backward(l3.GradEntrada(), l1.Saida(), taxa);
                l1.Backward(l2.GradEntrada(), imagem_gpu, taxa);
            }
            perdas[epoca] = perda_total / treino.tamanho();
            checa(cudaGetLastError(), "kernel de treino");
        }
        checa(cudaDeviceSynchronize(), "fim do treino");
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
            envia(imagem_gpu, teste.imagem(i), Barenn::entradas);
            l1.Forward(imagem_gpu);
            l2.Forward(l1.Saida());
            l3.Forward(l2.Saida());
            recebe(escores.data(), l3.Saida(), Barenn::classes);
            return escores.data();
        }, teste);
        checa(cudaFree(imagem_gpu), "cudaFree");
        checa(cudaFree(gradiente_gpu), "cudaFree");
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
