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

#include "xla/codegen/math/intrinsic.h"

#include <cstddef>
#include <optional>
#include <string>
#include <variant>

#include "absl/strings/str_cat.h"
#include "xla/primitive_util.h"

namespace xla::codegen {

std::string Intrinsic::Name(Type type) {
  if (auto* scalar = std::get_if<Scalar>(&type)) {
    return primitive_util::LowercasePrimitiveTypeName(scalar->type);
  }
  auto& vec = std::get<Vec>(type);
  return absl::StrCat("v", vec.width,
                      primitive_util::LowercasePrimitiveTypeName(vec.type));
}

PrimitiveType Intrinsic::ElementType(Type type) {
  if (auto* scalar = std::get_if<Scalar>(&type)) {
    return scalar->type;
  }
  auto& vec = std::get<Vec>(type);
  return vec.type;
}

std::optional<size_t> Intrinsic::Width(Type type) {
  if (auto* scalar = std::get_if<Scalar>(&type)) {
    return std::nullopt;
  }
  auto& vec = std::get<Vec>(type);
  return vec.width;
}

}  // namespace xla::codegen
