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

// float32 uses the FTZ/DAZ-corrected functor (scalar + SIMD packetOp).
// double, Eigen::half, and bfloat16 use Eigen's standard scalar_floor_op;
// float16 subnormals are not flushed on CPU (they are normal float32 values),
// and the SIMD penalty of a corrected double/half/bfloat16 path is not
// justified for subnormal-only correctness.
// GPU registration uses functor::floor directly; GPU kernels do not run
// under CPU FTZ/DAZ settings.
REGISTER(UnaryOp, CPU, "Floor", functor::floor_cpu, float);
REGISTER3(UnaryOp, CPU, "Floor", functor::floor, Eigen::half, bfloat16,
          double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER3(UnaryOp, GPU, "Floor", functor::floor, float, Eigen::half, double);
#endif
REGISTER(UnaryOp, GPU, "Floor", functor::floor, bfloat16);
#endif
}  // namespace tensorflow
