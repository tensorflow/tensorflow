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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_inline_call.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

namespace xla {
namespace poplarplugin {

InlineCallVisitor::InlineCallVisitor(CompilerResources& res,
                                     const ArgVectors& inputs)
    : FullVisitor(res), inputs_(std::move(inputs)) {}

Status InlineCallVisitor::HandleParameter(HloInstruction* inst) {
  for (uint64 t = 0; t < inputs_[inst->parameter_number()].size(); t++) {
    auto& v = inputs_[inst->parameter_number()];
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, t, v[t]));
  }
  return Status::OK();
}

Status InlineCallVisitor::FinishVisit(HloInstruction* inst) {
  outputs_ = FindInstructionOutputs(tensor_map, inst);
  resources_.tensor_maps[inst->parent()->name()] = std::move(tensor_map);

  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
