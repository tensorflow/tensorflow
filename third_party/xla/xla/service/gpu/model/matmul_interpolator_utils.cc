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
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

std::string MatmulTypeStringRep(PrimitiveType lhs, PrimitiveType rhs,
                                PrimitiveType out) {
  std::string lhs_str = primitive_util::LowercasePrimitiveTypeName(lhs);
  std::string rhs_str = primitive_util::LowercasePrimitiveTypeName(rhs);
  std::string out_str = primitive_util::LowercasePrimitiveTypeName(out);
  return absl::StrCat(lhs_str, "x", rhs_str, "->", out_str);
}

}  // namespace

MatmulDTypeKey::MatmulDTypeKey(absl::string_view key) {
  std::string key_lower = absl::AsciiStrToLower(key);
  std::vector<absl::string_view> parts = absl::StrSplit(key_lower, "->");
  CHECK_EQ(parts.size(), 2);
  std::vector<absl::string_view> in_parts = absl::StrSplit(parts[0], 'x');
  CHECK_EQ(in_parts.size(), 2);
  lhs_dtype_ = *primitive_util::StringToPrimitiveType(in_parts[0]);
  rhs_dtype_ = *primitive_util::StringToPrimitiveType(in_parts[1]);
  out_dtype_ = *primitive_util::StringToPrimitiveType(parts[1]);
}

MatmulDTypeKey::MatmulDTypeKey(PrimitiveType lhs_dtype, PrimitiveType rhs_dtype,
                               PrimitiveType out_dtype)
    : lhs_dtype_(lhs_dtype), rhs_dtype_(rhs_dtype), out_dtype_(out_dtype) {}

MatmulDTypeKey::MatmulDTypeKey(absl::string_view lhs_dtype,
                               absl::string_view rhs_dtype,
                               absl::string_view out_dtype)
    : lhs_dtype_(*primitive_util::StringToPrimitiveType(lhs_dtype)),
      rhs_dtype_(*primitive_util::StringToPrimitiveType(rhs_dtype)),
      out_dtype_(*primitive_util::StringToPrimitiveType(out_dtype)) {}

bool MatmulDTypeKey::IsUniformDataType(PrimitiveType dtype) const {
  return lhs_dtype_ == dtype && rhs_dtype_ == dtype && out_dtype_ == dtype;
}

std::string MatmulDTypeKey::KeyString() const {
  return MatmulTypeStringRep(lhs_dtype_, rhs_dtype_, out_dtype_);
}

}  // namespace xla::gpu
