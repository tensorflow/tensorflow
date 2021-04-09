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
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

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

StatusOr<Comparison::Direction> StringToComparisonDirection(
    absl::string_view direction_name) {
  static auto* direction_map =
      new absl::flat_hash_map<string, Comparison::Direction>({
          {"EQ", Comparison::Direction::kEq},
          {"NE", Comparison::Direction::kNe},
          {"GE", Comparison::Direction::kGe},
          {"GT", Comparison::Direction::kGt},
          {"LE", Comparison::Direction::kLe},
          {"LT", Comparison::Direction::kLt},
      });
  auto it = direction_map->find(direction_name);
  if (it == direction_map->end()) {
    return InvalidArgument("Unknown comparison direction: %s", direction_name);
  }
  return it->second;
}

StatusOr<Comparison::Type> StringToComparisonType(
    absl::string_view compare_type_name) {
  static auto* type_map = new absl::flat_hash_map<string, Comparison::Type>({
      {"FLOAT", Comparison::Type::kFloat},
      {"TOTALORDER", Comparison::Type::kFloatTotalOrder},
      {"SIGNED", Comparison::Type::kSigned},
      {"UNSIGNED", Comparison::Type::kUnsigned},
  });
  auto it = type_map->find(compare_type_name);
  if (it == type_map->end()) {
    return InvalidArgument("Unknown comparison type: %s", compare_type_name);
  }
  return it->second;
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
    default:
      LOG(FATAL) << "Attempted to print incomplete comparison type";
  }
}

Comparison::Comparison(Direction dir, PrimitiveType type)
    : dir_(dir), type_(DefaultComparisonType(type)) {}

Comparison::Type Comparison::DefaultComparisonType(PrimitiveType type) {
  switch (type) {
    case S8:
    case S16:
    case S32:
    case S64:
      return Type::kSigned;
    case PRED:
    case U8:
    case U16:
    case U32:
    case U64:
      return Type::kUnsigned;
    case F16:
    case F32:
    case BF16:
    case F64:
    case C64:
    case C128:
      return Type::kFloat;
    default:
      LOG(FATAL) << "Unsupported comparison mode."
                 << PrimitiveType_Name(type) << "\n";
  }
}

Comparison Comparison::Converse() const {
  return Comparison(Converse(dir_), type_);
}

absl::optional<Comparison> Comparison::Inverse() const {
  switch (type_) {
    case Type::kFloat:
      // Floating-point comparisons don't have inverses unless total order is
      // supported (e.g. comparison can return true if one operand is NaN).
      return absl::nullopt;
    case Type::kFloatTotalOrder:
    case Type::kSigned:
    case Type::kUnsigned:
      return Comparison(Inverse(dir_), type_);
  }
}

bool Comparison::IsReflexive() const {
  switch (dir_) {
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return IsSigned() || IsUnsigned() || IsFloatTotalOrder();
    case Direction::kNe:
    case Direction::kGt:
    case Direction::kLt:
      return false;
  }
}

bool Comparison::IsAntireflexive() const {
  switch (dir_) {
    case Direction::kNe:
      return IsSigned() || IsUnsigned() || IsFloatTotalOrder();
    case Direction::kGt:
    case Direction::kLt:
      return true;
    case Direction::kEq:
    case Direction::kGe:
    case Direction::kLe:
      return false;
  }
}

/* static */ Comparison::Direction Comparison::Converse(
    Comparison::Direction dir) {
  switch (dir) {
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

/* static */ Comparison::Direction Comparison::Inverse(
    Comparison::Direction dir) {
  switch (dir) {
    case Direction::kEq:
      return Direction::kNe;
    case Direction::kNe:
      return Direction::kEq;
    case Direction::kGe:
      return Direction::kLt;
    case Direction::kGt:
      return Direction::kLe;
    case Direction::kLe:
      return Direction::kGt;
    case Direction::kLt:
      return Direction::kGe;
  }
}

std::string Comparison::ToString(std::string prefix1,
                                 std::string prefix2) const {
  return prefix1 + std::string(ComparisonDirectionToString(dir_)) + prefix2 +
         std::string(ComparisonTypeToString(type_));
}
}  // namespace xla
