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

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_subcomputation.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/strcat.h"

#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

SubComputationVisitor::SubComputationVisitor(poplar::Graph* graph,
                                             CompilerResources& res,
                                             int64 num_parameters)
        : FullVisitor(graph, res) {
  inputs_.resize(num_parameters);
}

Status SubComputationVisitor::HandleParameter(HloInstruction* inst) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out,
                      AddTensor(*graph_, inst, inst->shape(), resources_));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  inputs_[inst->parameter_number()] = out;
  return Status::OK();
}

Status SubComputationVisitor::FinishVisit(HloInstruction* inst) {
  int64 c = 1;
  if (ShapeUtil::IsTuple(inst->shape())) {
    c = ShapeUtil::TupleElementCount(inst->shape());
  }

  for (int64 i=0; i<c; i++) {
    poplar::Tensor out;
    TF_ASSIGN_OR_RETURN(out, FindInstructionOutput(tensor_map, inst, i));
    outputs_.push_back(out);
  }

  tensor_map.clear();

  return Status::OK();
}


}  // namespace poplarplugin
}  // namespace xla
