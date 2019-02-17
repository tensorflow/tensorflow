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
    CompilerResources& res, const ArgVectors& inputs, bool inplace_inputs,
    const std::vector<const SubComputationVisitor*>& dependent_subcomputations)
    : FullVisitor(res),
      temp_inputs_(inputs),
      inputs_(temp_inputs_.size()),
      inplace_inputs_(inplace_inputs),
      dependent_subcomputations_(dependent_subcomputations) {}

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

  poplar::Graph& graph = GetGraph(resources_, param_inst);

  std::vector<xla::Shape> shapes = FlattenedXlaShape(param_inst->shape());
  auto& inputs = inputs_[param_num];
  auto& used = used_tensors_[param_num];
  auto& allocated = allocated_tensors_[param_num];
  inputs.resize(shapes.size());
  used.resize(shapes.size());
  allocated.resize(shapes.size());
  for (unsigned int i = 0; i < shapes.size(); i++) {
    auto src = std::make_pair(inst, i);
    auto& t = temp_inputs_[param_num][i];
    used[i] = InputIsUsedInThisSubComputation(param_inst, shapes, i);
    allocated[i] =
        InputIsUsedInDependentSubComputations(param_inst, i) || used[i];
    if (!allocated[i]) {
      inputs[i] = t;
    } else {
      if (inplace_inputs_) {
        inputs[i] = t;
      } else {
        // For used non-inplace inputs we have the following cases:
        // 1. This input has an allocation target.
        // 2. This input contains constants.
        // 3. This input does not have a target.
        // For cases 1 and 2 we allocate a new tensor. For case 3 we clone the
        // layout of the input.
        if (HasTensorAllocationTarget(src, resources_) ||
            t.containsConstant()) {
          TF_ASSIGN_OR_RETURN(inputs[i], AddTensor(graph, src, shapes[i],
                                                   resources_, tensor_map));
        } else {
          auto name = StrCat(GetDebugName(inst), "_in_", i);
          inputs[i] = graph.clone(t, name);
        }
      }
    }
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, inputs[i]));
  }

  return Status::OK();
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
  return (param < allocated_tensors_.size() &&
          index < allocated_tensors_.at(param).size() &&
          allocated_tensors_.at(param)[index]);
}

bool SubComputationVisitor::InputIsUsed(int64 param, unsigned int index) const {
  return (param < used_tensors_.size() &&
          index < used_tensors_.at(param).size() &&
          used_tensors_.at(param)[index]);
}

}  // namespace poplarplugin
}  // namespace xla
