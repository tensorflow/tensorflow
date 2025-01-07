/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_original_value.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::string OriginalValueToStringHelper(const OriginalValue& original_value,
                                        const Shape& shape,
                                        std::vector<int64_t>& shape_index) {
  std::string result;
  if (shape.IsTuple()) {
    if (shape.tuple_shapes().empty()) {
      return "()";
    }
    absl::StrAppend(&result, "(");
    shape_index.push_back(0);
    absl::StrAppend(&result,
                    OriginalValueToStringHelper(
                        original_value, shape.tuple_shapes(0), shape_index));
    shape_index.pop_back();
    for (int64_t i = 1; i < shape.tuple_shapes().size(); ++i) {
      absl::StrAppend(&result, ", ");
      shape_index.push_back(i);
      absl::StrAppend(&result,
                      OriginalValueToStringHelper(
                          original_value, shape.tuple_shapes(i), shape_index));
      shape_index.pop_back();
    }
    absl::StrAppend(&result, ")");
    return result;
  }

  // The original_value may refer to an empty array, such as origin {}, so let's
  // check whether that's the case before accessing them. Generally speaking the
  // index _should_ be good, but let's double check.
  const auto& leaf = original_value.element(shape_index);
  if (leaf.has_value()) {
    absl::StrAppend(
        &result, "{", "\"", leaf->instruction_name, "\"",
        (leaf->shape_index.empty() ? "" : " " + leaf->shape_index.ToString()),
        "}");
  }
  return result;
}

std::string OriginalValueToString(const OriginalValue& original_value) {
  std::vector<int64_t> shape_index;
  return OriginalValueToStringHelper(original_value, original_value.shape(),
                                     shape_index);
}

OriginalValueProto OriginalValueToProto(const OriginalValue& original_value) {
  OriginalValueProto original_value_proto;
  for (const auto& leaf : original_value.leaves()) {
    OriginalArrayProto* original_array_proto =
        original_value_proto.add_leaves();
    for (const auto& index : leaf.first) {
      original_array_proto->add_leaf_shape_index(index);
    }
    *original_array_proto->mutable_instruction_name() =
        leaf.second->instruction_name;
    for (const auto& index : leaf.second->shape_index) {
      original_array_proto->add_shape_index(index);
    }
  }
  return original_value_proto;
}

}  // namespace xla
