#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "roll_op.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void RollCudaKernel(int N, int D, int* dim_size, const T* input, T* output,\
                const int* shifts, const int* strides) {
  CUDA_1D_KERNEL_LOOP(in_i, out_size) {
    int out_i = in_i;
    // loop through dimensions
    for (int d = 0; d < D; d++) {
      // find indices input/output for current dimension
      const int ds = dim_size[d];
      const int in_dim_i = (in_i / strides[d]) % ds;
      const int out_dim_i = ((in_dim_i + shifts[d]) % ds + ds) % ds; // modulo that works with negatives
      // convert back to flat index
      out_i += (out_dim_i - in_dim_i) * strides[d];
    }

    output[out_i] = input[in_i];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct RollFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int N, int D, int* dim_size, const T* input, T* output,\
                  const int* shifts, const int* strides){
    const int64 out_size = out.size();
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    RollCudaKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            N, D, dim_size, input, output, shifts, strides);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in pad_op.cc.
#define DEFINE_GPU_SPECS(T)                      \
  template struct RollFunctor<GPUDevice, T>; \
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
