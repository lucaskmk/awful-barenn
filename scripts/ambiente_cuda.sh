# Coloca o toolkit CUDA 12.8.1 do Franky no PATH. Use com: source scripts/ambiente_cuda.sh
# O module load cuda/12.8.1 funciona no login, mas falha nos nos de GPU.
export CUDA_HOME=/opt/ohpc/pub/spack/opt/spack/linux-skylake/cuda-12.8.1-hk7c4l6fpeucmtbivmr7kn73dnkarzyt
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
