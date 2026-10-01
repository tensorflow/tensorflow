/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/comparison_util.h"

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

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

absl::string_view ComparisonPrimitiveTypeToString(PrimitiveType type) {
  return PrimitiveType_Name(type);
}

absl::string_view ComparisonOrderToString(Comparison::Order order) {
  switch (order) {
    case Comparison::Order::kPartial:
      return "PARTIALORDER";
    case Comparison::Order::kWeak:
      return "WEAKORDER";
    case Comparison::Order::kTotal:
      return "TOTALORDER";
  }
}

absl::string_view ComparisonOrderToShortString(Comparison::Order order) {
  switch (order) {
    case Comparison::Order::kPartial:
      return "PARTIAL";
    case Comparison::Order::kWeak:
      return "WEAK";
    case Comparison::Order::kTotal:
      return "TOTAL";
  }
}

absl::StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction) {
  static auto* const map =
      new absl::flat_hash_map<absl::string_view, Comparison::Direction>({
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

absl::StatusOr<Comparison::Order> ShortStringToComparisonOrder(
    absl::string_view order) {
  static auto* const map =
      new absl::flat_hash_map<absl::string_view, Comparison::Order>({
          {"TOTAL", Comparison::Order::kTotal},
          {"WEAK", Comparison::Order::kWeak},
          {"PARTIAL", Comparison::Order::kPartial},
      });
  auto it = map->find(order);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison order: %s", order);
  }
  return it->second;
}

absl::StatusOr<Comparison::Order> ComparisonTypeToOrder(
    absl::string_view comparison_type) {
  static auto* const map =
      new absl::flat_hash_map<absl::string_view, Comparison::Order>({
          {"FLOAT", Comparison::Order::kPartial},
          {"TOTALORDER", Comparison::Order::kTotal},
          {"SIGNED", Comparison::Order::kTotal},
          {"UNSIGNED", Comparison::Order::kTotal},
      });
  auto it = map->find(comparison_type);
  if (it == map->end()) {
    return InvalidArgument("Unknown comparison type: %s", comparison_type);
  }
  return it->second;
}

// Returns the expected ordering for each primitive type.
Comparison::Order Comparison::DefaultOrdering(PrimitiveType type) {
  if (primitive_util::IsFloatingPointType(type) ||
      primitive_util::IsComplexType(type)) {
    return Comparison::Order::kPartial;
  }
  if (primitive_util::IsIntegralType(type) || type == PRED) {
    return Comparison::Order::kTotal;
  }
  LOG(FATAL) << "Unsupported type: " << PrimitiveType_Name(type);
}

Comparison::Comparison(Direction dir, PrimitiveType type, Order order)
    : dir_(dir), primitive_type_(type), order_(order) {
  CHECK(primitive_util::IsArrayType(type))
      << "Unsupported type: " << PrimitiveType_Name(type);
}

Comparison::Comparison(Direction dir, PrimitiveType type)
    : dir_(dir), primitive_type_(type), order_(DefaultOrdering(type)) {}

Comparison Comparison::Converse() const {
  return Comparison(xla::Converse(dir_), primitive_type_, order_);
}

std::optional<Comparison> Comparison::Inverse() const {
  if (IsPartialOrder()) {
    // We assume comparisons don't have inverses unless they are total or weak
    // order, e.g., a partial order floating point comparison can return false
    // for both (a < b) and (a >= b) if one operand is NaN.
    return std::nullopt;
  }
  if (primitive_util::IsArrayType(primitive_type_)) {
    return Comparison(xla::Inverse(dir_), primitive_type_, order_);
  }
  return std::nullopt;
}

bool Comparison::IsReflexive() const {
  switch (dir_) {
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return IsTotalOrder() || IsWeakOrder();
    case Direction::kNe:
    case Direction::kGt:
    case Direction::kLt:
      return false;
  }
}

bool Comparison::IsAntireflexive() const {
  switch (dir_) {
    case Direction::kNe:
      return IsTotalOrder() || IsWeakOrder();
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
