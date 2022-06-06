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
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status RemoveUnusedOp::Run(Model* model, std::size_t op_index,
                                         bool* modified) {
  *modified = false;
  const auto it = model->operators.begin() + op_index;
  const auto* op = it->get();

  // Bail if any output is used, and is not an input_array of
  // the model. We allow specifying an arbitrary input_array,
  // treating the part of the graph leading up to it as unused.
  for (const auto& output : op->outputs) {
    CHECK(model->HasArray(output));
    // If this output is provided as the model's input array,
    // then we don't need this operator to produce its contents.
    if (IsInputArray(*model, output)) {
      continue;
    }
    // If this output is provided as a RNN's state array,
    // then we don't need this operator to produce its contents.
    // So far this case has only been encountered with TensorFlow
    // Fill ops used to zero-initialize RNN states, which is
    // redundant for us as we zero-initialize RNN states anyway.
    bool found_output_as_rnn_state_array = false;
    for (const auto& rnn_state : model->flags.rnn_states()) {
      if (output == rnn_state.state_array()) {
        CHECK(op->type == OperatorType::kFill ||
              op->type == OperatorType::kIdentity);
        found_output_as_rnn_state_array = true;
        break;
      }
    }
    if (found_output_as_rnn_state_array) {
      continue;
    }
    for (const std::string& output_array : model->flags.output_arrays()) {
      if (output == output_array) {
        return ::tensorflow::OkStatus();
      }
    }
    for (const auto& rnn_state : model->flags.rnn_states()) {
      if (output == rnn_state.back_edge_source_array()) {
        // The output is consumed by a RNN back-edge..
        if (!IsDiscardableArray(*model, rnn_state.back_edge_source_array()) ||
            !IsDiscardableArray(*model, rnn_state.state_array()) ||
            CountOpsWithInput(*model, rnn_state.state_array())) {
          return ::tensorflow::OkStatus();
        }
      }
    }
    if (CountOpsWithInput(*model, output)) {
      return ::tensorflow::OkStatus();
    }
  }

  if (op->unresolved_outputs) {
    AddMessageF("Not discarding %s because it has unresolved outputs.",
                LogName(*op));
    return ::tensorflow::OkStatus();
  }

  AddMessageF("Discarding %s because none of its outputs is used.",
              LogName(*op));
  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
