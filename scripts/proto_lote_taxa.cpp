// Prototipo: qualidade do treino em mini-lotes em funcao de B e da taxa.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>
using namespace std;
struct A { vector<float> x; int y; };
vector<A> ler(const char* c) { ifstream f(c); vector<A> d; string l; while (getline(f, l)) { for (char& ch : l) if (ch == ',') ch = ' '; istringstream s(l); A a; float v; for (int i = 0; i < 64; i++) { s >> v; a.x.push_back(v / 16.f); } s >> a.y; d.push_back(a); } return d; }
struct L { int n, m; bool fin; vector<float> w, b; L(int n_, int m_, bool f) : n(n_), m(m_), fin(f), w(n_ * m_), b(m_, 0) { mt19937 g(1337); uniform_real_distribution<float> u(-.05f, .05f); for (auto& v : w) v = u(g); } };
int main(int argc, char** argv) {
    auto tr = ler("optdigits.tra"), te = ler("optdigits.tes");
    int Bs[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    double fat[] = {1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125};
    for (int B : Bs) for (double f : fat) {
        float lr = 0.02f * f;
        L c[3] = {L(64, 32, false), L(32, 16, false), L(16, 10, true)};
        double perda_final = 0;
        bool nan = false;
        for (int ep = 0; ep < 51 && !nan; ep++) {
            perda_final = 0;
            for (size_t s = 0; s < tr.size(); s += B) {
                size_t fim = min(tr.size(), s + B);
                vector<float> gw[3], gb[3];
                for (int l = 0; l < 3; l++) { gw[l].assign(c[l].w.size(), 0); gb[l].assign(c[l].m, 0); }
                for (size_t k = s; k < fim; k++) {
                    vector<float> xs[4], zs[3];
                    xs[0] = tr[k].x;
                    for (int l = 0; l < 3; l++) {
                        auto& C = c[l]; vector<float> z(C.m), y(C.m);
                        for (int i = 0; i < C.m; i++) { float a = C.b[i]; for (int j = 0; j < C.n; j++) a += xs[l][j] * C.w[i * C.n + j]; z[i] = a; y[i] = C.fin ? a : max(0.f, a); }
                        zs[l] = z; xs[l + 1] = y;
                    }
                    vector<float> g(10);
                    for (int i = 0; i < 10; i++) { float e = xs[3][i] - (i == tr[k].y ? 1.f : 0.f); perda_final += .5 * e * e; g[i] = e; }
                    for (int l = 2; l >= 0; l--) {
                        auto& C = c[l]; vector<float> d(C.m), gi(C.n, 0);
                        for (int i = 0; i < C.m; i++) d[i] = C.fin ? g[i] : g[i] * (zs[l][i] > 0 ? 1.f : 0.f);
                        for (int r = 0; r < C.m; r++) for (int q = 0; q < C.n; q++) gi[q] += d[r] * C.w[r * C.n + q];
                        for (int i = 0; i < C.m; i++) { for (int j = 0; j < C.n; j++) gw[l][i * C.n + j] += d[i] * xs[l][j]; gb[l][i] += d[i]; }
                        g = gi;
                    }
                }
                for (int l = 0; l < 3; l++) { auto& C = c[l]; for (size_t q = 0; q < C.w.size(); q++) C.w[q] -= lr * gw[l][q]; for (int i = 0; i < C.m; i++) C.b[i] -= lr * gb[l][i]; }
            }
            perda_final /= tr.size();
            if (!isfinite(perda_final)) nan = true;
        }
        int ok = 0;
        for (auto& a : te) { vector<float> x = a.x; vector<float> y; for (int l = 0; l < 3; l++) { auto& C = c[l]; y.assign(C.m, 0); for (int i = 0; i < C.m; i++) { float s = C.b[i]; for (int j = 0; j < C.n; j++) s += x[j] * C.w[i * C.n + j]; y[i] = C.fin ? s : max(0.f, s); } x = y; } ok += (max_element(y.begin(), y.end()) - y.begin()) == a.y; }
        printf("B=%4d lr=%.6f perda=%8.4f acc=%6.2f%%\n", B, lr, perda_final, 100.0 * ok / te.size());
        fflush(stdout);
    }
}
