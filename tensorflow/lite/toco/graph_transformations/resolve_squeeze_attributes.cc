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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/remove_trivial_passthrough.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ResolveSqueezeAttributes::Run(Model* model,
                                                   std::size_t op_index,
                                                   bool* modified) {
  *modified = false;
  auto* squeeze_op = model->operators[op_index].get();
  if (squeeze_op->type != OperatorType::kSqueeze) {
    return ::tensorflow::OkStatus();
  }
  DCHECK_EQ(squeeze_op->inputs.size(), 1);
  DCHECK_EQ(squeeze_op->outputs.size(), 1);

  // If the output is consumed by a reshape op, it's a trivial squeeze.
  if (CountOpsWithInput(*model, squeeze_op->outputs[0]) == 1) {
    const auto* next_op = GetOpWithInput(*model, squeeze_op->outputs[0]);
    if (next_op->type == OperatorType::kReshape) {
      AddMessageF(
          "%s is trivial because its output is only consumed by a "
          "Reshape op",
          LogName(*squeeze_op));

      *modified = RemoveTrivialPassthroughOp(this, model, op_index);
      return ::tensorflow::OkStatus();
    }
  }
  return ::tensorflow::OkStatus();
}

}  // namespace toco
