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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
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

bool operator==(const OriginalArray& lhs, const OriginalArray& rhs) {
  return lhs.instruction_name == rhs.instruction_name &&
         lhs.shape_index == rhs.shape_index;
}

bool operator!=(const OriginalArray& lhs, const OriginalArray& rhs) {
  return !(lhs == rhs);
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

void OriginalValue::ClearInternalNodeValues() {
  if (is_synthetic_call()) {
    return;
  }
  mutable_tree()->ForEachMutableElement(
      [&](const ShapeIndex& index, std::optional<OriginalArray>* value) {
        if (!mutable_tree()->IsLeaf(index)) {
          *value = std::nullopt;
        }
      });
}

OriginalValue::OriginalValue(
    TupleTree<std::optional<OriginalArray>>::Node&& root_node)
    : data_(TupleTree<std::optional<OriginalArray>>(std::move(root_node))) {
  ClearInternalNodeValues();
}
OriginalValue::OriginalValue(TupleTree<std::optional<OriginalArray>>&& tree)
    : data_(std::move(tree)) {
  ClearInternalNodeValues();
}
OriginalValue::OriginalValue(
    const TupleTree<std::optional<OriginalArray>>& tree)
    : data_(tree) {
  ClearInternalNodeValues();
}

OriginalValue::OriginalValue(SyntheticCallType synthetic) : data_(synthetic) {}

OriginalValue OriginalValue::SyntheticCall() {
  OriginalValue result(SyntheticCallType{});
  return result;
}

std::string OriginalValue::ToString() const {
  if (is_synthetic_call()) {
    return "[synthetic_call]";
  }
  auto node_or = tree().ToNode();
  CHECK_OK(node_or.status());
  return NodeToString(*node_or);
}

bool OriginalValue::operator==(const OriginalValue& other) const {
  if (is_synthetic_call() != other.is_synthetic_call()) {
    return false;
  }
  if (is_synthetic_call()) {
    return true;  // Synthetic == Synthetic
  }
  auto this_original_arrays = original_arrays();
  auto other_original_arrays = other.original_arrays();
  return std::equal(this_original_arrays.begin(), this_original_arrays.end(),
                    other_original_arrays.begin(), other_original_arrays.end());
}

OriginalValueProto OriginalValue::ToProto() const {
  OriginalValueProto original_value_proto;
  if (is_synthetic_call()) {
    original_value_proto.set_is_synthetic_call(true);
  } else {
    tree().ForEachElement([&original_value_proto](
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
  }
  return original_value_proto;
}

std::shared_ptr<OriginalValue> OriginalValue::FromProto(
    const xla::OriginalValueProto& original_value_proto) {
  if (original_value_proto.is_synthetic_call()) {
    return std::make_shared<OriginalValue>(OriginalValue::SyntheticCall());
  }
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
  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    const auto* tuple = instruction->operand(0);
    std::shared_ptr<OriginalValue> tuple_original_value =
        tuple->original_value();
    if (!tuple_original_value || tuple_original_value->is_synthetic_call()) {
      return nullptr;
    }
    auto original_value = std::make_shared<OriginalValue>(
        TupleTree<std::optional<OriginalArray>>(instruction->shape()));
    const auto& tuple_tree = tuple_original_value->tree();
    original_value->mutable_tree()->ForEachMutableElement(
        [&](const ShapeIndex& index, std::optional<OriginalArray>* value) {
          ShapeIndex src_index({instruction->tuple_index()});
          src_index.insert(src_index.end(), index.begin(), index.end());
          *value = tuple_tree.element(src_index);
        });
    return original_value;
  }

  if (instruction->opcode() == HloOpcode::kTuple) {
    auto original_value = std::make_shared<OriginalValue>(
        TupleTree<std::optional<OriginalArray>>(instruction->shape()));
    for (int64_t i = 0; i < instruction->operand_count(); ++i) {
      const HloInstruction* operand = instruction->operand(i);
      auto op_original_value = operand->original_value();
      if (!op_original_value || op_original_value->is_synthetic_call()) {
        return nullptr;
      }
      const auto& op_tree = op_original_value->tree();
      op_tree.ForEachElement([&](const ShapeIndex& index,
                                 const std::optional<OriginalArray>& value) {
        ShapeIndex dest_index({i});
        dest_index.insert(dest_index.end(), index.begin(), index.end());
        *original_value->mutable_tree()->mutable_element(dest_index) = value;
      });
    }
    return original_value;
  }

  // Default case: create a new tree with leaves pointing to this instruction.
  auto original_value = std::make_shared<OriginalValue>(
      TupleTree<std::optional<OriginalArray>>(instruction->shape()));
  for (auto& leaf : original_value->mutable_original_arrays()) {
    leaf.second = {absl::StrCat(prefix, instruction->name()), leaf.first};
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

  if (!clone || original_value->is_synthetic_call()) {
    dest_instruction->set_original_value(original_value);
    return;
  }

  // Deep clone the tree.
  auto cloned_tree = std::make_shared<OriginalValue>(original_value->tree());
  dest_instruction->set_original_value(cloned_tree);
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

/* static */
TupleTree<std::optional<OriginalArray>>&
OriginalValue::EmptyOriginalValueTupleTree() {
  static absl::NoDestructor<TupleTree<std::optional<OriginalArray>>>
      kEmptyTupleTree;
  return *kEmptyTupleTree;
}

bool OriginalValue::IsCompatibleWith(const Shape& shape) const {
  if (is_synthetic_call()) {
    return true;
  }
  return tree().IsStructurallyCompatible(shape);
}

}  // namespace xla
