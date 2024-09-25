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

#ifndef XLA_SERVICE_CONSTANT_VALUE_H_
#define XLA_SERVICE_CONSTANT_VALUE_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/util.h"

namespace xla {

// Class used to represent a constant. Can contain signed/unsigned values
// and values of many types type erasing the actual type and handling corner
// cases like going out of bound.
class ConstantValue {
 public:
  // Constructor makes sure the extra bits of the value are masked away. Handles
  // signed and unsigned cases.
  ConstantValue(uint64_t value, int32_t bitwidth, bool is_signed)
      : value_(is_signed
                   ? absl::bit_cast<uint64_t>(
                         absl::bit_cast<int64_t>(
                             value << (8 * sizeof(uint64_t) - bitwidth)) >>
                         (8 * sizeof(uint64_t) - bitwidth))
                   : KeepLowerBits(value, bitwidth)),
        bitwidth_(bitwidth),
        is_signed_(is_signed) {}
  static ConstantValue GetZero(int32_t bitwidth, bool is_signed) {
    return ConstantValue(0, bitwidth, is_signed);
  }
  static ConstantValue GetOne(int32_t bitwidth, bool is_signed) {
    return ConstantValue(1, bitwidth, is_signed);
  }
  static ConstantValue GetSigned(int64_t value, int32_t bitwidth) {
    return ConstantValue(absl::bit_cast<uint64_t>(value), bitwidth,
                         /*is_signed=*/true);
  }
  static ConstantValue GetUnsigned(uint64_t value, int32_t bitwidth) {
    return ConstantValue(value, bitwidth, /*is_signed=*/false);
  }
  static absl::StatusOr<ConstantValue> FromLiteral(const Literal& literal);
  ConstantValue add(const ConstantValue& other) const {
    return ConstantValue(value_ + other.value_, bitwidth_, is_signed_);
  }
  ConstantValue sub(const ConstantValue& other) const {
    return ConstantValue(value_ - other.value_, bitwidth_, is_signed_);
  }
  ConstantValue div(const ConstantValue& other) const;
  ConstantValue mod(const ConstantValue& other) const;
  ConstantValue mul(const ConstantValue& other) const;
  bool lt(const ConstantValue& other) const;
  bool gt(const ConstantValue& other) const;
  bool eq(const ConstantValue& other) const { return *this == other; }
  int64_t GetSignedValue() const { return absl::bit_cast<int64_t>(value_); }
  uint64_t GetUnsignedValue() const { return value_; }
  int32_t GetBitwidth() const { return bitwidth_; }
  bool IsSigned() const { return is_signed_; }
  bool operator==(const ConstantValue& other) const {
    return value_ == other.value_ && bitwidth_ == other.bitwidth_ &&
           is_signed_ == other.is_signed_;
  }
  std::string ToString() const;

 private:
  uint64_t value_;
  int32_t bitwidth_;
  bool is_signed_;
};

}  // namespace xla

#endif  // XLA_SERVICE_CONSTANT_VALUE_H_
