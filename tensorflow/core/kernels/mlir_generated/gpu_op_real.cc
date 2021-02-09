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

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

GENERATE_UNARY_GPU_KERNEL2(Real, c64, f32, DT_FLOAT, float,
                           std::complex<float>);
REGISTER_COMPLEX_GPU_KERNEL(Real, c64, f32, float, std::complex<float>);
GENERATE_UNARY_GPU_KERNEL2(Real, c128, f64, DT_DOUBLE, double,
                           std::complex<double>);
REGISTER_COMPLEX_GPU_KERNEL(Real, c128, f64, double, std::complex<double>);

}  // namespace tensorflow
