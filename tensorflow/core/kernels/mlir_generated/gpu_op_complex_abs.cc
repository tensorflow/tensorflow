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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

GENERATE_UNARY_GPU_KERNEL2(ComplexAbs, DT_COMPLEX64, DT_FLOAT);
REGISTER_COMPLEX_GPU_KERNEL(ComplexAbs, DT_COMPLEX64, DT_FLOAT);
GENERATE_UNARY_GPU_KERNEL2(ComplexAbs, DT_COMPLEX128, DT_DOUBLE);
REGISTER_COMPLEX_GPU_KERNEL(ComplexAbs, DT_COMPLEX128, DT_DOUBLE);

}  // namespace tensorflow
