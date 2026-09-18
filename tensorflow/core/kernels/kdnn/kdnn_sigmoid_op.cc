/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// KDNN-backed Sigmoid kernel. Compiled only when the build flag
// --define=enable_kdnn=true is passed AND the target is aarch64 (the
// `if_enable_kdnn` macro enforces both). The kernel is a drop-in
// replacement for the public `Sigmoid` op; the Grappler remapper
// substitutes it in when conditions are met.

#ifdef KERNEL_KDNN

#include "tensorflow/core/kernels/kdnn/kdnn_unary_op.h"

namespace tensorflow {

#define REGISTER_KDNN_SIGMOID_KERNEL(TYPE)                          \
  REGISTER_KERNEL_BUILDER(Name("_KdnnSigmoid")                     \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<TYPE>("T"),          \
                          KdnnUnaryOp<KDNN_ACT_SIGMOID, TYPE>);

REGISTER_KDNN_SIGMOID_KERNEL(float);
REGISTER_KDNN_SIGMOID_KERNEL(Eigen::half);
REGISTER_KDNN_SIGMOID_KERNEL(bfloat16);

#undef REGISTER_KDNN_SIGMOID_KERNEL

}  // namespace tensorflow

#endif  // KERNEL_KDNN