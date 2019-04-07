/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef TENSORFLOW_USE_MPI

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_launch_config.h"

#include "tensorflow/contrib/mpi_collectives/kernels/ring.h"

namespace tensorflow {
namespace contrib {
namespace mpi_collectives {

using CPUDevice = Eigen::ThreadPoolDevice;

template <>
MPI_Datatype MPIType<float>() {
  return MPI_FLOAT;
};
template <>
MPI_Datatype MPIType<int>() {
  return MPI_INT;
};
template <>
MPI_Datatype MPIType<long long>() {
  return MPI_LONG_LONG;
};

template <>
DataType TensorFlowDataType<float>() {
  return DT_FLOAT;
};
template <>
DataType TensorFlowDataType<int>() {
  return DT_INT32;
};
template <>
DataType TensorFlowDataType<long long>() {
  return DT_INT64;
};

// Generate all necessary specializations for RingAllreduce.
template Status RingAllreduce<GPUDevice, int>(OpKernelContext*, const Tensor*,
                                              Tensor*, Tensor*);
template Status RingAllreduce<GPUDevice, long long>(OpKernelContext*,
                                                    const Tensor*, Tensor*,
                                                    Tensor*);
template Status RingAllreduce<GPUDevice, float>(OpKernelContext*, const Tensor*,
                                                Tensor*, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<GPUDevice, int>(OpKernelContext*, const Tensor*,
                                              const std::vector<size_t>&,
                                              Tensor*);
template Status RingAllgather<GPUDevice, long long>(OpKernelContext*,
                                                    const Tensor*,
                                                    const std::vector<size_t>&,
                                                    Tensor*);
template Status RingAllgather<GPUDevice, float>(OpKernelContext*, const Tensor*,
                                                const std::vector<size_t>&,
                                                Tensor*);

// Synchronously copy data on the GPU, using a different stream than the default
// and than TensorFlow to avoid synchronizing on operations unrelated to the
// allreduce.
template <>
void CopyTensorData<GPUDevice>(void* dst, void* src, size_t size) {
  auto stream = CudaStreamForMPI();
  cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
  cudaStreamSynchronize(stream);
};

// Elementwise accumulation kernel for GPU.
template <typename T>
__global__ void elemwise_accum(T* out, const T* in, const size_t N) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    out[i] += in[i];
  }
}

// Synchronously accumulate tensors on the GPU, using a different stream than
// the default and than TensorFlow to avoid synchronizing on operations
// unrelated to the allreduce.
#define GENERATE_ACCUMULATE(type)                                          \
  template <>                                                              \
  void AccumulateTensorData<GPUDevice, type>(type * dst, type * src,       \
                                             size_t size) {                \
    auto stream = CudaStreamForMPI();                                      \
    TF_CHECK_OK(CudaLaunchKernel(elemwise_accum<type>, 32, 256, 0, stream, \
                                 dst, src, size));                         \
    cudaStreamSynchronize(stream);                                         \
  };
GENERATE_ACCUMULATE(int);
GENERATE_ACCUMULATE(long long);
GENERATE_ACCUMULATE(float);
#undef GENERATE_ACCUMULATE

}  // namespace mpi_collectives
}  // namespace contrib
}  // namespace tensorflow
#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_USE_MPI
