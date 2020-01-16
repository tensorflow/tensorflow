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

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/cwise_ops_gradients.h"

namespace tensorflow {
REGISTER5(UnaryOp, CPU, "Sigmoid", functor::sigmoid, float, Eigen::half, double,
          complex64, complex128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER3(UnaryOp, GPU, "Sigmoid", functor::sigmoid, float, Eigen::half,
          double);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER(UnaryOp, SYCL, "Sigmoid", functor::sigmoid, float);
#endif  // TENSORFLOW_USE_SYCL

REGISTER5(SimpleBinaryOp, CPU, "SigmoidGrad", functor::sigmoid_grad, float,
          Eigen::half, double, complex64, complex128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER3(SimpleBinaryOp, GPU, "SigmoidGrad", functor::sigmoid_grad, float,
          Eigen::half, double);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER(SimpleBinaryOp, SYCL, "SigmoidGrad", functor::sigmoid_grad, float);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
