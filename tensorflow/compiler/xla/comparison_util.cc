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

#include "tensorflow/compiler/xla/comparison_util.h"

#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

// Verifies that this is a valid Comparison: (1) not a partial ordering on
// integers, and (2) a valid PrimitiveType.
bool IsValidComparison(xla::PrimitiveType type, Comparison::Order order) {
  switch (type) {
    case F16:
    case F32:
    case BF16:
    case F64:
    case C64:
    case C128:
      return true;
    case S4:
    case S8:
    case S16:
    case S32:
    case S64:
    case PRED:
    case U4:
    case U8:
    case U16:
    case U32:
    case U64:
      return order == Comparison::Order::kTotal;
    case TUPLE:
    case OPAQUE_TYPE:
    case TOKEN:
    case PRIMITIVE_TYPE_INVALID:
    case PrimitiveType_INT_MAX_SENTINEL_DO_NOT_USE_:
    case PrimitiveType_INT_MIN_SENTINEL_DO_NOT_USE_:
    // TODO(b/259609697): Add support for comparing F8 values. F8 values are
    // comparable like any other floating-point type, but comparisons are not
    // yet implemented by any backend.
    case F8E5M2:
    case F8E4M3FN:
      return false;
  }
}

// Returns the X32 primitive type for each Type.
PrimitiveType DefaultPrimitiveType(Comparison::Type type) {
  switch (type) {
    case Comparison::Type::kFloat:
    case Comparison::Type::kFloatTotalOrder:
      return PrimitiveType::F32;
    case Comparison::Type::kSigned:
      return PrimitiveType::S32;
    case Comparison::Type::kUnsigned:
      return PrimitiveType::U32;
  }
}

// Returns the default ordering for each Comparison::Type.
Comparison::Order DefaultOrdering(Comparison::Type type) {
  switch (type) {
    case Comparison::Type::kFloat:
      return Comparison::Order::kPartial;
    case Comparison::Type::kFloatTotalOrder:
    case Comparison::Type::kSigned:
    case Comparison::Type::kUnsigned:
      return Comparison::Order::kTotal;
  }
}

// Returns the expected ordering for each primitive type.
Comparison::Order DefaultOrdering(PrimitiveType type) {
  switch (type) {
    case S4:
    case S8:
    case S16:
    case S32:
    case S64:
    case PRED:
    case U4:
    case U8:
    case U16:
    case U32:
    case U64:
      return Comparison::Order::kTotal;
    case F8E5M2:
    case F8E4M3FN:
    case BF16:
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
      return Comparison::Order::kPartial;
    default:
      LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
  }
}

// Returns the converse of `direction`.
Comparison::Direction Converse(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGe;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGt;
  }
}

// Returns the inverse of `direction`.
Comparison::Direction Inverse(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return Comparison::Direction::kNe;
    case Comparison::Direction::kNe:
      return Comparison::Direction::kEq;
    case Comparison::Direction::kGe:
      return Comparison::Direction::kLt;
    case Comparison::Direction::kGt:
      return Comparison::Direction::kLe;
    case Comparison::Direction::kLe:
      return Comparison::Direction::kGt;
    case Comparison::Direction::kLt:
      return Comparison::Direction::kGe;
  }
}

}  // namespace

std::string ComparisonDirectionToString(Comparison::Direction direction) {
  switch (direction) {
    case Comparison::Direction::kEq:
      return "EQ";
    case Comparison::Direction::kNe:
      return "NE";
    case Comparison::Direction::kGe:
      return "GE";
    case Comparison::Direction::kGt:
      return "GT";
    case Comparison::Direction::kLe:
      return "LE";
    case Comparison::Direction::kLt:
      return "LT";
    default:
      LOG(FATAL) << "Attempted to print uninitialized comparison direction";
  }
}

std::string ComparisonTypeToString(Comparison::Type type) {
  switch (type) {
    case Comparison::Type::kFloat:
      return "FLOAT";
    case Comparison::Type::kFloatTotalOrder:
      return "TOTALORDER";
    case Comparison::Type::kSigned:
      return "SIGNED";
    case Comparison::Type::kUnsigned:
      return "UNSIGNED";
  }
}

std::string ComparisonPrimitiveTypeToString(PrimitiveType type) {
  return PrimitiveType_Name(type);
}

std::string ComparisonOrderToString(Comparison::Order order) {
  switch (order) {
    case Comparison::Order::kPartial:
      return "PARTIALORDER";
    case Comparison::Order::kTotal:
      return "TOTALORDER";
  }
}

StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction) {
  static auto* map =
      new absl::flat_hash_map<std::string, Comparison::Direction>({
          {"EQ", Comparison::Direction::kEq},
          {"NE", Comparison::Direction::kNe},
          {"GE", Comparison::Direction::kGe},
          {"GT", Comparison::Direction::kGt},
          {"LE", Comparison::Direction::kLe},
          {"LT", Comparison::Direction::kLt},
      });
  auto it = map->find(direction);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison direction: %s", direction);
  }
  return it->second;
}

StatusOr<Comparison::Order> StringToComparisonOrder(absl::string_view order) {
  static auto* map = new absl::flat_hash_map<std::string, Comparison::Order>({
      {"TOTALORDER", Comparison::Order::kTotal},
      {"PARTIALORDER", Comparison::Order::kPartial},
  });
  auto it = map->find(order);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison type: %s", order);
  }
  return it->second;
}

StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view comparison) {
  static auto* map = new absl::flat_hash_map<std::string, Comparison::Type>({
      {"FLOAT", Comparison::Type::kFloat},
      {"TOTALORDER", Comparison::Type::kFloatTotalOrder},
      {"SIGNED", Comparison::Type::kSigned},
      {"UNSIGNED", Comparison::Type::kUnsigned},
  });
  auto it = map->find(comparison);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison type: %s", comparison);
  }
  return it->second;
}

Comparison::Type Comparison::DefaultComparisonType(PrimitiveType type) {
  switch (type) {
    case S4:
    case S8:
    case S16:
    case S32:
    case S64:
      return Type::kSigned;
    case PRED:
    case U4:
    case U8:
    case U16:
    case U32:
    case U64:
      return Type::kUnsigned;
    case F8E5M2:
    case F8E4M3FN:
    case F16:
    case F32:
    case BF16:
    case F64:
    case C64:
    case C128:
      return Type::kFloat;
    default:
      LOG(FATAL) << "Unexpected: " << PrimitiveType_Name(type);
  }
}

Comparison::Comparison(Direction dir, PrimitiveType type, Order order)
    : dir_(dir),
      primitive_type_(type),
      order_(order),
      type_(DefaultComparisonType(type)) {
  CHECK(IsValidComparison(primitive_type_, order_));
}

Comparison::Comparison(Direction dir, PrimitiveType type)
    : dir_(dir),
      primitive_type_(type),
      order_(DefaultOrdering(type)),
      type_(DefaultComparisonType(type)) {
  CHECK(IsValidComparison(primitive_type_, order_));
}

Comparison::Comparison(Direction dir, Type type)
    : dir_(dir),
      primitive_type_(DefaultPrimitiveType(type)),
      order_(DefaultOrdering(type)),
      type_(type) {
  CHECK(IsValidComparison(primitive_type_, order_));
}

Comparison Comparison::Converse() const {
  return Comparison(xla::Converse(dir_), primitive_type_, order_);
}

std::optional<Comparison> Comparison::Inverse() const {
  if (IsPartialOrder()) {
    // We assume comparisons don't have inverses unless they are total order,
    // e.g., a partial order floating point comparison can return true if one
    // operand is NaN.
    return std::nullopt;
  }
  switch (primitive_type_) {
    case F16:
    case F32:
    case BF16:
    case F64:
    case C64:
    case C128:
    case S4:
    case S8:
    case S16:
    case S32:
    case S64:
    case PRED:
    case U4:
    case U8:
    case U16:
    case U32:
    case U64:
      return Comparison(xla::Inverse(dir_), primitive_type_, order_);
    case TUPLE:
    case OPAQUE_TYPE:
    case TOKEN:
    case F8E5M2:
    case F8E4M3FN:
    case PRIMITIVE_TYPE_INVALID:
    case PrimitiveType_INT_MAX_SENTINEL_DO_NOT_USE_:
    case PrimitiveType_INT_MIN_SENTINEL_DO_NOT_USE_:
      return std::nullopt;
  }
}

bool Comparison::IsReflexive() const {
  switch (dir_) {
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return IsTotalOrder();
    case Direction::kNe:
    case Direction::kGt:
    case Direction::kLt:
      return false;
  }
}

bool Comparison::IsAntireflexive() const {
  switch (dir_) {
    case Direction::kNe:
      return IsTotalOrder();
    case Direction::kGt:
    case Direction::kLt:
      return true;
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return false;
  }
}

std::string Comparison::ToString(std::string prefix1, std::string prefix2,
                                 std::string prefix3) const {
  return absl::StrCat(prefix1, ComparisonDirectionToString(dir_), prefix2,
                      ComparisonPrimitiveTypeToString(primitive_type_), prefix3,
                      ComparisonOrderToString(order_));
}
}  // namespace xla
