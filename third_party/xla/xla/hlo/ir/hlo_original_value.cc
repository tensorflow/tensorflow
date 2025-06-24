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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::string OriginalValueToString(const OriginalValue& original_value,
                                  const Shape& shape,
                                  std::vector<int64_t>& shape_index) {
  std::string result;
  if (shape.IsTuple()) {
    if (shape.tuple_shapes().empty()) {
      return "()";
    }
    shape_index.push_back(0);
    absl::StrAppend(&result, "(",
                    OriginalValueToString(original_value, shape.tuple_shapes(0),
                                          shape_index));
    shape_index.pop_back();
    for (int64_t i = 1; i < shape.tuple_shapes().size(); ++i) {
      shape_index.push_back(i);
      absl::StrAppend(&result, ", ",
                      OriginalValueToString(
                          original_value, shape.tuple_shapes(i), shape_index));
      shape_index.pop_back();
    }
    absl::StrAppend(&result, ")");
    return result;
  }

  const auto& leaf = original_value.element(shape_index);
  if (leaf.has_value()) {
    absl::StrAppend(
        &result, "{", "\"", leaf->instruction_name, "\"",
        (leaf->shape_index.empty() ? "" : " " + leaf->shape_index.ToString()),
        (leaf->call_history.empty() ? "" : " \"" + leaf->call_history + "\""),
        "}");
  } else {
    absl::StrAppend(&result, "{}");
  }
  return result;
}

std::string OriginalValue::ToString() const {
  std::vector<int64_t> shape_index;
  return OriginalValueToString(*this, shape(), shape_index);
}

std::shared_ptr<OriginalValue> OriginalValue::FromProto(
    const xla::OriginalValueProto& original_value_proto) {
  xla::Shape original_value_shape(
      Shape::FromProto(original_value_proto.shape()).value_or(Shape()));
  auto original_value = std::make_shared<OriginalValue>(original_value_shape);

  for (const auto& leaf : original_value_proto.leaves()) {
    *original_value->mutable_element(ShapeIndex(leaf.leaf_shape_index())) = {
        leaf.instruction_name(), ShapeIndex(leaf.shape_index()),
        leaf.call_history()};
  }
  return original_value;
}

OriginalValueProto OriginalValue::ToProto() const {
  OriginalValueProto original_value_proto;
  *original_value_proto.mutable_shape() = shape().ToProto();
  for (const auto& leaf : leaves()) {
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
    *original_tensor_proto->mutable_call_history() = leaf.second->call_history;
  }
  return original_value_proto;
}

void CopyOriginalValue(const HloInstruction* src_instruction,
                       HloInstruction* dest_instruction, bool clone) {
  // This is not expected to happen in practice.
  if (!src_instruction || !dest_instruction ||
      !ShapeUtil::Compatible(src_instruction->shape(),
                             dest_instruction->shape())) {
    VLOG(1) << "Expect the new instruction to have the same shape with the old "
               "instruction when moving over original_value";
    return;
  }

  std::shared_ptr<OriginalValue> original_value =
      src_instruction->original_value();
  if (!original_value) {
    return;
  }

  if (!clone) {
    dest_instruction->set_original_value(original_value);
    return;
  }

  std::shared_ptr<OriginalValue> original_value_clone =
      std::make_shared<OriginalValue>(original_value->shape());
  original_value_clone->CopySubtreeFrom(*original_value, {}, {});
  dest_instruction->set_original_value(original_value_clone);
}

void DeduplicateOriginalValues(HloModule* module) {
  absl::flat_hash_set<OriginalValuePointer> unique_original_values;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (std::shared_ptr<OriginalValue> original_value =
              instruction->original_value()) {
        OriginalValuePointer original_value_ptr(original_value);
        auto p = unique_original_values.insert(original_value_ptr);
        if (!p.second) {
          // Reassign the pointer with the existing identical object and release
          // the duplicate.
          instruction->set_original_value(p.first->original_value);
        }
      }
    }
  }
}

}  // namespace xla
