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
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_subcomputation.h"

#include <poplar/Tensor.hpp>

using ::absl::StrCat;

namespace xla {
namespace poplarplugin {

SubComputationVisitor::SubComputationVisitor(CompilerResources& res,
                                             const ArgVectors& inputs)
    : FullVisitor(res) {
  temp_inputs_ = inputs;
  inputs_.resize(temp_inputs_.size());
  input_valid_.resize(temp_inputs_.size());
}

static bool InputIsUnused(HloInstruction* inst,
                          const std::vector<xla::Shape>& shapes,
                          unsigned int index) {
  if (inst->parent()->root_instruction() == inst) {
    return false;
  }

  if (inst->user_count() == 0) {
    return true;
  }

  // Non-tuples are considered always used
  if (!ShapeUtil::IsTuple(inst->shape())) {
    return false;
  }

  // We ignore nested tuples
  if (shapes.size() != ShapeUtil::TupleElementCount(inst->shape())) {
    return false;
  }

  for (auto user : inst->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }

    if (user->tuple_index() == index) {
      return false;
    }
  }

  return true;
}

Status SubComputationVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  poplar::Graph& graph = GetGraph(resources_, inst);

  ArgVector inputs;
  std::vector<xla::Shape> shapes = FlattenedXlaShape(inst->shape());
  std::vector<bool> valid(shapes.size());
  for (unsigned int i = 0; i < shapes.size(); i++) {
    auto& t = temp_inputs_[inst->parameter_number()][i];

    if (InputIsUnused(inst, shapes, i)) {
      valid[i] = false;
      inputs.push_back(t);
      TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, t));
    } else {
      valid[i] = true;

      if (t.containsConstant()) {
        auto src = std::make_pair(inst, i);
        poplar::Tensor out;
        TF_ASSIGN_OR_RETURN(
            out, AddTensor(graph, src, shapes[i], resources_, tensor_map));
        inputs.push_back(out);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
      } else {
        auto name = StrCat(GetDebugName(inst), "_in_", i);
        poplar::Tensor out = graph.clone(t, name);
        inputs.push_back(out);
        TF_CHECK_OK(AddOutputTensor(tensor_map, inst, i, out));
      }
    }
  }

  inputs_[inst->parameter_number()] = inputs;
  input_valid_[inst->parameter_number()] = valid;

  return Status::OK();
}

Status SubComputationVisitor::FinishVisit(HloInstruction* inst) {
  outputs_ = FindInstructionOutputs(tensor_map, inst);

  temp_inputs_.clear();

  resources_.tensor_maps[inst->parent()->name()] = std::move(tensor_map);

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
