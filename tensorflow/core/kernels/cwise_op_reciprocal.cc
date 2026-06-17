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

namespace tensorflow {
REGISTER6(UnaryOp, CPU, "Inv", functor::inverse, float, Eigen::half, double,
          bfloat16, complex64, complex128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER6(UnaryOp, GPU, "Inv", functor::inverse, float, Eigen::half, double,
          int64, complex64, complex128);
#endif
REGISTER(UnaryOp, GPU, "Inv", functor::inverse, bfloat16);
#endif

REGISTER6(SimpleBinaryOp, CPU, "InvGrad", functor::inverse_grad, float,
          Eigen::half, double, bfloat16, complex64, complex128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER4(SimpleBinaryOp, GPU, "InvGrad", functor::inverse_grad, float,
          Eigen::half, bfloat16, double);
#endif

REGISTER6(UnaryOp, CPU, "Reciprocal", functor::inverse, float, Eigen::half,
          bfloat16, double, complex64, complex128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER6(UnaryOp, GPU, "Reciprocal", functor::inverse, float, Eigen::half,
          double, int64, complex64, complex128);
#endif
REGISTER(UnaryOp, GPU, "Reciprocal", functor::inverse, bfloat16);
#endif

REGISTER6(SimpleBinaryOp, CPU, "ReciprocalGrad", functor::inverse_grad, float,
          Eigen::half, bfloat16, double, complex64, complex128);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER4(SimpleBinaryOp, GPU, "ReciprocalGrad", functor::inverse_grad, float,
          Eigen::half, bfloat16, double);
#endif
}  // namespace tensorflow
