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

#include "xla/service/gpu/model/matmul_interpolator_utils.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"

namespace xla::gpu {

std::string MatmulTypeStringRep(PrimitiveType lhs, PrimitiveType rhs,
                                PrimitiveType out) {
  std::string lhs_str = primitive_util::LowercasePrimitiveTypeName(lhs);
  std::string rhs_str = primitive_util::LowercasePrimitiveTypeName(rhs);
  std::string out_str = primitive_util::LowercasePrimitiveTypeName(out);
  return absl::StrCat(lhs_str, "x", rhs_str, "->", out_str);
}

}  // namespace xla::gpu
