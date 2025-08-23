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
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/pointer_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tuple_tree.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::string OriginalArray::ToString() const {
  std::string result;
  absl::StrAppend(&result, "\"", instruction_name, "\"",
                  (shape_index.empty() ? "" : " " + shape_index.ToString()));
  return result;
}

OriginalArrayProto OriginalArray::ToProto() const {
  OriginalArrayProto original_array_proto;
  original_array_proto.set_instruction_name(instruction_name);
  for (const auto& index : shape_index) {
    original_array_proto.add_shape_index(index);
  }
  return original_array_proto;
}

OriginalArray OriginalArray::FromProto(
    const xla::OriginalArrayProto& original_array_proto) {
  return {original_array_proto.instruction_name(),
          ShapeIndex(original_array_proto.shape_index())};
}

namespace {
using Node = TupleTree<std::optional<OriginalArray>>::Node;

std::string NodeToString(const Node& node) {
  if (node.IsLeaf()) {
    const std::optional<OriginalArray>& leaf_val = node.value();
    if (leaf_val.has_value()) {
      return absl::StrCat("{", leaf_val->ToString(), "}");
    }
    return "{}";
  }

  if (node.children().empty()) {
    return "()";
  }

  std::vector<std::string> children_str;
  for (const auto& child : node.children()) {
    children_str.push_back(NodeToString(child));
  }

  return absl::StrCat("(", absl::StrJoin(children_str, ", "), ")");
}
}  // namespace

std::string OriginalValue::ToString() const {
  auto node_or = tree_.ToNode();
  CHECK_OK(node_or.status());
  return NodeToString(*node_or);
}

OriginalValueProto OriginalValue::ToProto() const {
  OriginalValueProto original_value_proto;
  tree_.ForEachElement([&original_value_proto](
                           const ShapeIndex& index,
                           const std::optional<OriginalArray>& value) {
    OriginalValueElementProto* original_value_node_proto =
        original_value_proto.add_elements();
    for (const auto& i : index) {
      original_value_node_proto->add_shape_index(i);
    }
    if (value.has_value()) {
      *original_value_node_proto->mutable_original_array() = value->ToProto();
    }
  });
  return original_value_proto;
}

std::shared_ptr<OriginalValue> OriginalValue::FromProto(
    const xla::OriginalValueProto& original_value_proto) {
  std::vector<std::pair<ShapeIndex, std::optional<OriginalArray>>> nodes;
  for (const auto& leaf : original_value_proto.elements()) {
    ShapeIndex index(leaf.shape_index());
    if (leaf.has_original_array()) {
      nodes.emplace_back(index,
                         OriginalArray::FromProto(leaf.original_array()));
    } else {
      // This case should not happen based on ToProto, but handling defensively.
      nodes.emplace_back(index, std::nullopt);
    }
  }
  return std::make_shared<OriginalValue>(
      TupleTree<std::optional<OriginalArray>>(absl::MakeSpan(nodes)));
}

std::shared_ptr<OriginalValue> OriginalValue::CreateFromInstruction(
    const HloInstruction* instruction, absl::string_view prefix) {
  std::shared_ptr<OriginalValue> original_value =
      std::make_shared<OriginalValue>(
          TupleTree<std::optional<OriginalArray>>(instruction->shape()));

  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    const auto* tuple = instruction->operand(0);
    std::shared_ptr<OriginalValue> tuple_original_value =
        tuple->original_value();
    if (!tuple_original_value) {
      return nullptr;
    }
    original_value->CopySubtreeFrom(*tuple_original_value,
                                    {instruction->tuple_index()}, {});
  } else if (instruction->opcode() == HloOpcode::kTuple) {
    for (int64_t operand_number = 0;
         operand_number < instruction->operand_count(); ++operand_number) {
      auto element_original_value =
          instruction->operand(operand_number)->original_value();
      if (!element_original_value) {
        return nullptr;
      }
      original_value->CopySubtreeFrom(*element_original_value, {},
                                      {operand_number});
    }
  } else {
    for (auto& leaf : original_value->mutable_original_arrays()) {
      leaf.second = {absl::StrCat(prefix, instruction->name()), leaf.first};
    }
  }
  return original_value;
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
      std::make_shared<OriginalValue>();
  original_value_clone->CopySubtreeFrom(*original_value, {}, {});
  dest_instruction->set_original_value(original_value_clone);
}

void DeduplicateOriginalValues(HloModule* module) {
  absl::flat_hash_set<std::shared_ptr<OriginalValue>,
                      PointeeHash<OriginalValue>, PointeeEqual<OriginalValue>>
      unique_original_values;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (std::shared_ptr<OriginalValue> original_value =
              instruction->original_value()) {
        auto p = unique_original_values.insert(original_value);
        if (!p.second) {
          // Reassign the pointer with the existing identical object and release
          // the duplicate.
          instruction->set_original_value(*p.first);
        }
      }
    }
  }
}

}  // namespace xla
