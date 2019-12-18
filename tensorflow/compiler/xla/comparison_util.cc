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

std::string ComparisonDirectionToString(ComparisonDirection direction) {
  switch (direction) {
    case ComparisonDirection::kEq:
      return "EQ";
    case ComparisonDirection::kNe:
      return "NE";
    case ComparisonDirection::kGe:
      return "GE";
    case ComparisonDirection::kGt:
      return "GT";
    case ComparisonDirection::kLe:
      return "LE";
    case ComparisonDirection::kLt:
      return "LT";
  }
}

StatusOr<ComparisonDirection> StringToComparisonDirection(
    absl::string_view direction_name) {
  static auto* direction_map =
      new absl::flat_hash_map<string, ComparisonDirection>({
          {"EQ", ComparisonDirection::kEq},
          {"NE", ComparisonDirection::kNe},
          {"GE", ComparisonDirection::kGe},
          {"GT", ComparisonDirection::kGt},
          {"LE", ComparisonDirection::kLe},
          {"LT", ComparisonDirection::kLt},
      });
  auto it = direction_map->find(direction_name);
  if (it == direction_map->end()) {
    return InvalidArgument("Unknown comparison direction: %s", direction_name);
  }
  return it->second;
}

}  // namespace xla
