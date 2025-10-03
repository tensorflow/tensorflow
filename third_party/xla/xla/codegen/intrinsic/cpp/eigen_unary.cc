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

Vec4f FastTanhf(const Vec4f x) {
  Eigen::Map<const Eigen::Array4f> eigen_view(
      reinterpret_cast<const float*>(&x));
  Eigen::Array4f result_array = eigen_view.tanh();
  return *reinterpret_cast<Vec4f*>(&result_array);
}

}  // namespace xla::codegen
