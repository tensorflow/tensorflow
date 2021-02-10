/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(AddV2, f16, DT_HALF, Eigen::half);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(AddV2, f32, DT_FLOAT, float);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(AddV2, f64, DT_DOUBLE, double);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(AddV2, i64, DT_INT64, int64);

// Add is the same as AddV2 except for strings, which we do not support on gpu.
REGISTER_ALIASED_GPU_KERNEL(Add, AddV2, f16, f16, Eigen::half);
REGISTER_ALIASED_GPU_KERNEL(Add, AddV2, f32, f32, float);
REGISTER_ALIASED_GPU_KERNEL(Add, AddV2, f64, f64, double);
REGISTER_ALIASED_GPU_KERNEL(Add, AddV2, i64, i64, int64);

}  // namespace tensorflow
