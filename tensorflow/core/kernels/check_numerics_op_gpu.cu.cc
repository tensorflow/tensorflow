#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <assert.h>
#include <stdio.h>

#include <math.h>
#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

// A Cuda kernel to check if each element is Inf or Nan. If any exists, the
// relevant elements in abnormal_detected will be set
template <typename T>
__global__ void CheckNumericsKernel(const T *data, int size,
                                    int abnormal_detected[2]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;

  while (offset < size) {
    if (isnan(data[offset])) {
      abnormal_detected[0] = 1;
    }
    if (isinf(data[offset])) {
      abnormal_detected[1] = 1;
    }
    offset += total_thread_count;
  }
}

}  // namespace

// A simple launch pad to launch the Cuda kernels that checks the numerical
// abnormality in the given array
template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice &d, const T *data, int size,
           int abnormal_detected[2]) {
    const int32 block_size = d.maxCudaThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumCudaMultiProcessors() * d.maxCudaThreadsPerMultiProcessor()) /
        block_size;

    CheckNumericsKernel<T><<<num_blocks, block_size, 0, d.stream()>>>(
        data, size, abnormal_detected);
  }
};

template struct CheckNumericsLaunch<float>;
template struct CheckNumericsLaunch<double>;

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
