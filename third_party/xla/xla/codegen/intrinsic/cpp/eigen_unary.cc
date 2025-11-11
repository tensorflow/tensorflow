/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/intrinsic/cpp/eigen_unary.h"

#include "Eigen/Core"
#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {

// Using Packet over a Map'd Array yields better llvm IR on ARM.
using Packet4f = Eigen::internal::Packet4f;

Vec4f FastTanhf(const Vec4f x) {
  Packet4f packet = static_cast<Eigen::internal::Packet4f>(x);
  Packet4f res = Eigen::internal::ptanh_float(packet);
  return *static_cast<Vec4f*>(&res);
}

Vec8d FastRqsqrtf(const Vec8d x) {
  const Eigen::Map<const Eigen::Array<double, 8, 1>> x_arr((const double*)&x);
  const Eigen::Array<double, 8, 1> res = x_arr.rsqrt();
  return *(Vec8d*)res.data();
}

}  // namespace xla::codegen
