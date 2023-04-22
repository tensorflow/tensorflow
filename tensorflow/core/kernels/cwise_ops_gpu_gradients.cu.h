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

#if !GOOGLE_CUDA && !TENSORFLOW_USE_ROCM
#error This file must only be included when building with Cuda or ROCm support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_GRADIENTS_CU_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_GRADIENTS_CU_H_

#define EIGEN_USE_GPU

#include <complex>

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_gradients.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/platform/logging.h"
namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

// Partial specialization of SimpleBinaryFunctor<Device=GPUDevice, Functor>.
template <typename Functor>
struct SimpleBinaryFunctor<GPUDevice, Functor> {
  void operator()(const GPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in1,
                  typename Functor::tin_type in2) {
    To32Bit(out).device(d) =
        To32Bit(in1).binaryExpr(in2, typename Functor::func());
  }
};

// Macros to explicitly instantiate kernels on GPU for multiple types
// (T0, T1, etc.) for SimpleBinaryFunctor (e.g., functor::tanh_grad).
#define DEFINE_SIMPLE_BINARY1(F, T) \
  template struct SimpleBinaryFunctor<GPUDevice, F<T> >
#define DEFINE_SIMPLE_BINARY2(F, T0, T1) \
  DEFINE_SIMPLE_BINARY1(F, T0);          \
  DEFINE_SIMPLE_BINARY1(F, T1)
#define DEFINE_SIMPLE_BINARY3(F, T0, T1, T2) \
  DEFINE_SIMPLE_BINARY2(F, T0, T1);          \
  DEFINE_SIMPLE_BINARY1(F, T2)
#define DEFINE_SIMPLE_BINARY4(F, T0, T1, T2, T3) \
  DEFINE_SIMPLE_BINARY2(F, T0, T1);              \
  DEFINE_SIMPLE_BINARY2(F, T2, T3)
#define DEFINE_SIMPLE_BINARY5(F, T0, T1, T2, T3, T4) \
  DEFINE_SIMPLE_BINARY2(F, T0, T1);                  \
  DEFINE_SIMPLE_BINARY3(F, T2, T3, T4)

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_GPU_GRADIENTS_CU_H_
