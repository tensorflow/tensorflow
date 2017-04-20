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

#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/poplar/driver/visitor_call.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/strcat.h"

#include "tensorflow/core/lib/core/errors.h"

#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

PoplarCallVisitor::PoplarCallVisitor(poplar::Graph* graph,
                                     const std::vector<poplar::Tensor>& inputs)
        : PoplarFullVisitor(graph),
          operands_(std::move(inputs)) {
}

Status PoplarCallVisitor::HandleParameter(HloInstruction* inst) {
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0,
                                     operands_[inst->parameter_number()]));
  return Status::OK();
}

Status PoplarCallVisitor::FinishVisit(HloInstruction* inst) {
  int64 c = 1;
  if (ShapeUtil::IsTuple(inst->shape())) {
    c = ShapeUtil::TupleElementCount(inst->shape());
  }

  for (int64 i=0; i<c; i++) {
    poplar::Tensor out;
    TF_ASSIGN_OR_RETURN(out, FindInstructionOutput(tensor_map, inst, i));
    output_.push_back(out);
  }

  return Status::OK();
}


}  // namespace poplarplugin
}  // namespace xla
