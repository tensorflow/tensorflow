/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_COMPARISON_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_COMPARISON_UTIL_H_

#include "absl/base/macros.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class Comparison {
 public:
  // Represents type of comparison
  enum class Type : uint8_t {
    kFloat,
    kFloatTotalOrder,
    kSigned,
    kUnsigned,
  };
  //
  // Represents different comparison operations.
  enum class Direction : uint8_t {
    kEq,
    kNe,
    kGe,
    kGt,
    kLe,
    kLt,
  };

  Comparison() = delete;
  explicit Comparison(Direction dir, Type type) : dir_(dir), type_(type) {}
  explicit Comparison(Direction dir, PrimitiveType type);

  Direction GetDirection() const { return dir_; }
  Type GetType() const { return type_; }

  inline bool IsEq() const { return dir_ == Direction::kEq; }
  inline bool IsNe() const { return dir_ == Direction::kNe; }
  inline bool IsGe() const { return dir_ == Direction::kGe; }
  inline bool IsGt() const { return dir_ == Direction::kGt; }
  inline bool IsLt() const { return dir_ == Direction::kLt; }
  inline bool IsFloat() const { return type_ == Type::kFloat; }
  inline bool IsFloatTotalOrder() const {
    return type_ == Type::kFloatTotalOrder;
  }
  inline bool IsSigned() const { return type_ == Type::kSigned; }
  inline bool IsUnsigned() const { return type_ == Type::kUnsigned; }

  // Returns true for comparisons, for which (a dir a) is always true.
  bool IsReflexive() const;

  // Returns true for comparisons, for which (a dir a) is always false.
  bool IsAntireflexive() const;

  // Gets the converse of the given comparison direction (e.g. >= turns to <=).
  // Useful when commuting operands to get constants into
  // immediate-accepting positions in the ISA.
  Comparison Converse() const;

  // Gets the inverse of the given comparison if it exists (e.g. >= turns to <).
  // Returns optional value because not all inversions may be supported.
  absl::optional<Comparison> Inverse() const;

  std::string ToString(std::string prefix1 = ".",
                       std::string prefix2 = ".") const;

  template <typename T, typename Comparator = bool (*)(const T, const T)>
  Comparator GetComparator() const {
    switch (GetDirection()) {
      case Direction::kEq:
        return +[](const T a, const T b) { return a == b; };
      case Direction::kNe:
        return +[](const T a, const T b) { return a != b; };
      case Direction::kGe:
        return +[](const T a, const T b) { return a >= b; };
      case Direction::kGt:
        return +[](const T a, const T b) { return a > b; };
      case Direction::kLe:
        return +[](const T a, const T b) { return a <= b; };
      case Direction::kLt:
        return +[](const T a, const T b) { return a < b; };
    }
  }

  template <typename T>
  bool Compare(const T a, const T b) const {
    return GetComparator<T>()(a, b);
  }
  static Type DefaultComparisonType(PrimitiveType t);

 private:
  static Direction Converse(Direction dir);
  static Direction Inverse(Direction dir);

  const Direction dir_;
  Type type_;
};

inline std::ostream& operator<<(std::ostream& os, const Comparison& cmp) {
  return os << cmp.ToString();
}
std::string ComparisonDirectionToString(Comparison::Direction direction);
std::string ComparisonTypeToString(Comparison::Type type);

StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction_name);

StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view compare_type_name);

using ComparisonDirection = Comparison::Direction;

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_COMPARISON_UTIL_H_
