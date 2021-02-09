/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

GENERATE_UNARY_GPU_KERNEL2(IsFinite, f16, i1, DT_BOOL, bool, Eigen::half);
REGISTER_GPU_KERNEL(IsFinite, f16, i1, Eigen::half);
GENERATE_UNARY_GPU_KERNEL2(IsFinite, f32, i1, DT_BOOL, bool, float);
REGISTER_GPU_KERNEL(IsFinite, f32, i1, float);
GENERATE_UNARY_GPU_KERNEL2(IsFinite, f64, i1, DT_BOOL, bool, double);
REGISTER_GPU_KERNEL(IsFinite, f64, i1, double);

}  // namespace tensorflow
