/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/cwise_op_clip.h"
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

template <typename T>
__global__ void UnaryClipCustomKernel(const int32 size_in,
                                      const T *__restrict__ in0,
                                      const T *__restrict__ in1,
                                      const T *__restrict__ in2,
                                      T *__restrict__ out) {
  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[0] < in0[i] ? in2[0] : in0[i];
    out[i] = value < in1[0] ? in1[0] : value;
  }
}

template <typename T>
__global__ void BinaryRightClipCustomKernel(const int32 size_in,
                                            const T *__restrict__ in0,
                                            const T *__restrict__ in1,
                                            const T *__restrict__ in2,
                                            T *__restrict__ out) {
  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[i] < in0[i] ? in2[i] : in0[i];
    out[i] = value < in1[0] ? in1[0] : value;
  }
}

template <typename T>
__global__ void BinaryLeftClipCustomKernel(const int32 size_in,
                                           const T *__restrict__ in0,
                                           const T *__restrict__ in1,
                                           const T *__restrict__ in2,
                                           T *__restrict__ out) {
  GPU_1D_KERNEL_LOOP(i, size_in) {
    T value = in2[0] < in0[i] ? in2[0] : in0[i];
    out[i] = value < in1[i] ? in1[i] : value;
  }
}

namespace functor {

// Unary functor for clip [Tensor, Scalar, Scalar]
template <typename T>
struct UnaryClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    GpuLaunchConfig config = GetGpuLaunchConfig(in0_flat.size(), d);

    TF_CHECK_OK(GpuLaunchKernel(
        UnaryClipCustomKernel<T>, config.block_count, config.thread_per_block,
        0, d.stream(), in0_flat.size(), in0_flat.data(), in1_flat.data(),
        in2_flat.data(), out_flat.data()));
  }
};

// Binary functor for clip [Tensor, Scalar, Tensor]
template <typename T>
struct BinaryRightClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    GpuLaunchConfig config = GetGpuLaunchConfig(in0_flat.size(), d);

    TF_CHECK_OK(GpuLaunchKernel(
        BinaryRightClipCustomKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), in0_flat.size(),
        in0_flat.data(), in1_flat.data(), in2_flat.data(), out_flat.data()));
  }
};

// Binary functor for clip [Tensor, Tensor, Scalar]
template <typename T>
struct BinaryLeftClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    GpuLaunchConfig config = GetGpuLaunchConfig(in0_flat.size(), d);

    TF_CHECK_OK(GpuLaunchKernel(
        BinaryLeftClipCustomKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), in0_flat.size(),
        in0_flat.data(), in1_flat.data(), in2_flat.data(), out_flat.data()));
  }
};

// Ternary functor for clip [Tensor, Tensor, Tensor]
template <typename T>
struct TernaryClipOp<GPUDevice, T> {
  void operator()(const GPUDevice &d, typename TTypes<T>::ConstFlat &in0_flat,
                  typename TTypes<T>::ConstFlat &in1_flat,
                  typename TTypes<T>::ConstFlat &in2_flat,
                  typename TTypes<T>::Flat &out_flat) const {
    out_flat.device(d) = in0_flat.cwiseMin(in2_flat).cwiseMax(in1_flat);
  }
};

#define INSTANTIATE_GPU(T)                         \
  template struct UnaryClipOp<GPUDevice, T>;       \
  template struct BinaryRightClipOp<GPUDevice, T>; \
  template struct BinaryLeftClipOp<GPUDevice, T>;  \
  template struct TernaryClipOp<GPUDevice, T>;
INSTANTIATE_GPU(Eigen::half);
INSTANTIATE_GPU(float);
INSTANTIATE_GPU(double);
INSTANTIATE_GPU(int8);
INSTANTIATE_GPU(int16);
INSTANTIATE_GPU(int32);
INSTANTIATE_GPU(int64);
INSTANTIATE_GPU(uint8);
INSTANTIATE_GPU(uint16);
#undef INSTANTIATE_GPU

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
