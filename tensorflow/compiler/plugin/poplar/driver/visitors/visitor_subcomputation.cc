/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_subcomputation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include <poplar/Tensor.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

SubComputationVisitor::SubComputationVisitor(
    CompilerResources& res, const ArgVectors& inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : DeferredAllocationVisitor(res),
      temp_inputs_(inputs),
      inputs_(inputs.size()),
      dependent_subcomputations_(dependent_subcomputations),
      used_tensors_(inputs.size()),
      allocated_tensors_(inputs.size()),
      has_allocation_target_(inputs.size()) {
  for (int64 i = 0; i < inputs.size(); i++) {
    inputs_[i].resize(inputs[i].size());
    used_tensors_[i].resize(inputs[i].size());
    allocated_tensors_[i].resize(inputs[i].size());
    has_allocation_target_[i].resize(inputs[i].size());
  }
}

InplaceSubComputationVisitor::InplaceSubComputationVisitor(
    CompilerResources& res, const ArgVectors& inputs,
    const TensorInputDescription& input_has_layout,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : SubComputationVisitor(res, inputs, dependent_subcomputations),
      input_has_layout_(input_has_layout) {}

bool SubComputationVisitor::InputIsUsedInThisSubComputation(
    HloParameterInstruction* inst, const std::vector<xla::Shape>& shapes,
    unsigned int index) {
  if (inst->parent()->root_instruction() == inst) {
    return true;
  }

  if (inst->user_count() == 0) {
    return false;
  }

  // Non-tuples are considered always used
  if (!inst->shape().IsTuple()) {
    return true;
  }

  // We ignore nested tuples
  if (shapes.size() != ShapeUtil::TupleElementCount(inst->shape())) {
    return true;
  }

  for (auto user : inst->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return true;
    }

    if (user->tuple_index() == index) {
      return true;
    }
  }
  return false;
}

bool SubComputationVisitor::InputIsUsedInDependentSubComputations(
    HloParameterInstruction* inst, unsigned int index) {
  const auto param_num = inst->parameter_number();
  for (const auto subcomputation : dependent_subcomputations_) {
    if (subcomputation->InputIsUsed(param_num, index)) {
      return true;
    }
  }
  return false;
}

Status SubComputationVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  HloParameterInstruction* param_inst =
      static_cast<HloParameterInstruction*>(inst);
  const auto param_num = param_inst->parameter_number();

  std::vector<xla::Shape> shapes = FlattenedXlaShape(param_inst->shape());
  auto& inputs = inputs_[param_num];
  auto& used = used_tensors_[param_num];
  auto& allocated = allocated_tensors_[param_num];
  auto& allocated_targets = has_allocation_target_[param_num];

  for (unsigned int i = 0; i < shapes.size(); i++) {
    auto& t = temp_inputs_[param_num][i];
    used[i] = InputIsUsedInThisSubComputation(param_inst, shapes, i);
    allocated[i] =
        InputIsUsedInDependentSubComputations(param_inst, i) || used[i];
    // If we have a deferred allocation then we can't add the output tensor yet
    // for the tensor mapping.
    bool add_output_tensor = true;
    if (!allocated[i]) {
      // For tensors which are not allocated we just forward them.
      inputs[i] = t;
    } else {
      // Handle the allocated tensor depending on whether this is inplace or
      // not.
      TF_ASSIGN_OR_RETURN(add_output_tensor,
                          HandleTensor(param_inst, shapes[i], i, t));
    }
    if (add_output_tensor) {
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, inputs[i]));
    }
  }

  return Status::OK();
}

StatusOr<bool> SubComputationVisitor::HandleTensor(
    HloParameterInstruction* inst, Shape& shape, const uint64 tuple_index,
    poplar::Tensor& tensor) {
  const auto param_num = inst->parameter_number();
  auto src = std::make_pair(inst, tuple_index);

  auto& inputs = inputs_[param_num];
  auto& allocated_targets = has_allocation_target_[param_num];

  // If we have a deferred allocation then we can't add the output tensor yet
  // for the tensor mapping.
  bool add_output_tensor = true;
  poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, tuple_index);

  // For used inputs we have the following cases:
  // 1. This is a deferred allocation.
  // 2. This input has an allocation target.
  // 3. This input contains constants.
  // 4. This input does not have a target.
  // For cases 2 and 3 we allocate a new tensor. For case 3 we clone the
  // layout of the input.
  if (CanDeferAllocation(inst, tuple_index)) {
    VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
            << tuple_index << ".";
    DeferAllocation(inst, tuple_index);
    add_output_tensor = false;
  } else if (HasTensorAllocationTarget(src, resources_) ||
             tensor.containsConstant()) {
    TF_ASSIGN_OR_RETURN(inputs[tuple_index],
                        AddTensor(graph, src, shape, resources_, tensor_map));
  } else {
    auto name = StrCat(GetDebugName(inst), "_in_", tuple_index);
    inputs[tuple_index] = graph.clone(tensor, name);
  }
  return add_output_tensor;
}

StatusOr<bool> InplaceSubComputationVisitor::HandleTensor(
    HloParameterInstruction* inst, Shape& shape, const uint64 tuple_index,
    poplar::Tensor& tensor) {
  const auto param_num = inst->parameter_number();
  auto src = std::make_pair(inst, tuple_index);

  auto& inputs = inputs_[param_num];
  auto& allocated_targets = has_allocation_target_[param_num];

  // If we have a deferred allocation then we can't add the output tensor yet
  // for the tensor mapping.
  bool add_output_tensor = true;
  poplar::Graph& graph = GetGraphWithOutputIndex(resources_, inst, tuple_index);

  // For inplace inputs, we can still allocated the input (and then add a copy)
  // iff there is a (deferred) allocation target and the input doesn't have a
  // layout.
  const bool input_has_layout = input_has_layout_[param_num][tuple_index];
  inputs[tuple_index] = tensor;
  if (!input_has_layout) {
    if (CanDeferAllocation(inst, tuple_index)) {
      VLOG(1) << "Deferring allocation of " << inst->name() << " sub tensor "
              << tuple_index << ".";
      DeferAllocation(inst, tuple_index);
      add_output_tensor = false;
    } else if (HasTensorAllocationTarget(src, resources_)) {
      // If the input has an allocation target, then we use that layout
      // rather than the input layout.
      TF_ASSIGN_OR_RETURN(inputs[tuple_index],
                          AddTensor(graph, src, shape, resources_, tensor_map));
      allocated_targets[tuple_index] = true;
    }
  }
  return add_output_tensor;
}

StatusOr<poplar::Tensor> SubComputationVisitor::PostProcessParameterAllocation(
    const HloInstruction* inst, int64 flat_tuple_index, const Shape&,
    poplar::Tensor tensor) {
  const auto param_num = inst->parameter_number();
  inputs_[param_num][flat_tuple_index] = tensor;
  has_allocation_target_[param_num][flat_tuple_index] = true;
  return tensor;
}

Status SubComputationVisitor::FinishVisit(HloInstruction* inst) {
  outputs_ = FindInstructionOutputs(tensor_map, inst);

  temp_inputs_.clear();

  resources_.tensor_maps[inst->parent()->name()] = std::move(tensor_map);

  return Status::OK();
}

const ArgVectors& SubComputationVisitor::inputs() { return inputs_; }

const OutVector& SubComputationVisitor::outputs() { return outputs_; }

bool SubComputationVisitor::InputIsAllocated(int64 param,
                                             unsigned int index) const {
  return allocated_tensors_[param][index];
}

bool SubComputationVisitor::InputIsUsed(int64 param, unsigned int index) const {
  return used_tensors_[param][index];
}

bool SubComputationVisitor::InputHasAllocationTarget(int64 param,
                                                     unsigned int index) const {
  return has_allocation_target_[param][index];
}

}  // namespace poplarplugin
}  // namespace xla
