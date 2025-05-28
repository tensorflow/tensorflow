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

#ifndef XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_UTILS_H_
#define XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_UTILS_H_

#include <string>
#include <tuple>

#include "absl/strings/string_view.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// A key for `interpolators_` map, which represents the data type of a matrix
// multiplication.
class MatmulDTypeKey {
 public:
  // Takes key in format "{lhs_dype}x{data_dtype}->{out_dype}" (e.g.
  // bf16xf16->f32) and constructs this class.
  explicit MatmulDTypeKey(absl::string_view key);

  MatmulDTypeKey(absl::string_view lhs_dtype, absl::string_view rhs_dtype,
                 absl::string_view out_dtype);

  MatmulDTypeKey(PrimitiveType lhs_dtype, PrimitiveType rhs_dtype,
                 PrimitiveType out_dtype);

  bool operator==(const MatmulDTypeKey& other) const {
    return std::tie(lhs_dtype_, rhs_dtype_, out_dtype_) ==
           std::tie(other.lhs_dtype_, other.rhs_dtype_, other.out_dtype_);
  }

  template <typename H>
  friend H AbslHashValue(H h, const MatmulDTypeKey& key) {
    return H::combine(std::move(h), key.lhs_dtype_, key.rhs_dtype_,
                      key.out_dtype_);
  }

  // Returns true iff `type` matches lhs, out, and rhs data types.
  bool IsUniformDataType(PrimitiveType type) const;

  // Constructs a lowercase key used in interpolation lookup. The representation
  // is supposed to loosen some constraints of `HloDotInstruction` serialization
  // and has the following format:
  //
  //   lowercase_str(`lhs`)x{lowercase_str(`rhs`)}->{lowercase_str(`out`)}
  //
  // For example for LHS=bf16, RHS=f32 and OUT=f32 the key will be:
  // 'bf16xf32->f32'.
  std::string KeyString() const;

 private:
  PrimitiveType lhs_dtype_;
  PrimitiveType rhs_dtype_;
  PrimitiveType out_dtype_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_UTILS_H_
