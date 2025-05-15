/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/constant_value.h"

#include <cstdint>
#include <string>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xla {

absl::StatusOr<ConstantValue> ConstantValue::FromLiteral(
    const Literal& literal) {
  CHECK_EQ(literal.shape().dimensions().size(), 0) << "Expected scalar literal";
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<ConstantValue>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<ConstantValue> {
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          return ConstantValue(
              static_cast<uint64_t>(
                  literal.GetFirstElement<
                      primitive_util::NativeTypeOf<primitive_type_constant>>()),
              /*bitwidth=*/primitive_util::BitWidth(primitive_type_constant),
              /*is_signed=*/
              primitive_util::IsSignedIntegralType(primitive_type_constant));
        }
        return InvalidArgument("Unsupported type");
      },
      literal.shape().element_type());
}

ConstantValue ConstantValue::div(const ConstantValue& other) const {
  if (!is_signed_) {
    return ConstantValue(value_ / other.value_, bitwidth_, is_signed_);
  }
  return ConstantValue(
      absl::bit_cast<uint64_t>(absl::bit_cast<int64_t>(value_) /
                               absl::bit_cast<int64_t>(other.value_)),
      bitwidth_, is_signed_);
}
ConstantValue ConstantValue::mod(const ConstantValue& other) const {
  if (!is_signed_) {
    return ConstantValue(value_ % other.value_, bitwidth_, is_signed_);
  }
  return ConstantValue(
      absl::bit_cast<uint64_t>(absl::bit_cast<int64_t>(value_) %
                               absl::bit_cast<int64_t>(other.value_)),
      bitwidth_, is_signed_);
}
ConstantValue ConstantValue::mul(const ConstantValue& other) const {
  if (!is_signed_) {
    return ConstantValue(value_ * other.value_, bitwidth_, is_signed_);
  }
  return ConstantValue(
      absl::bit_cast<uint64_t>(absl::bit_cast<int64_t>(value_) *
                               absl::bit_cast<int64_t>(other.value_)),
      bitwidth_, is_signed_);
}
bool ConstantValue::lt(const ConstantValue& other) const {
  if (!is_signed_) {
    return value_ < other.value_;
  }
  return absl::bit_cast<int64_t>(value_) <
         absl::bit_cast<int64_t>(other.value_);
}
bool ConstantValue::gt(const ConstantValue& other) const {
  if (!is_signed_) {
    return value_ > other.value_;
  }
  return absl::bit_cast<int64_t>(value_) >
         absl::bit_cast<int64_t>(other.value_);
}
std::string ConstantValue::ToString() const {
  return is_signed_ ? absl::StrCat(GetSignedValue())
                    : absl::StrCat(GetUnsignedValue());
}

}  // namespace xla
