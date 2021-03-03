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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status ResolveTensorFlowSwitch::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  const auto switch_it = model->operators.begin() + op_index;
  const auto* switch_op = switch_it->get();
  if (switch_op->type != OperatorType::kSwitch) {
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(switch_op->inputs.size(), 2);
  CHECK_EQ(switch_op->outputs.size(), 2);
  const std::string& predicate_name = switch_op->inputs[1];
  // If the predicate array hasn't been resolved to a constant yet,
  // we need to yield.
  if (!IsConstantParameterArray(*model, predicate_name)) {
    AddMessageF(
        "Waiting for the boolean predicate of %s to be resolved to a constant",
        LogName(*switch_op));
    return ::tensorflow::Status::OK();
  }

  // The predicate should be boolean, and should consist of a single value.
  const auto& predicate_array = model->GetArray(predicate_name);
  CHECK(predicate_array.data_type == ArrayDataType::kBool);
  for (const auto& dim : predicate_array.shape().dims()) {
    CHECK_EQ(dim, 1);
  }

  // Obtain the predicate boolean value.
  const auto& predicate_data =
      predicate_array.GetBuffer<ArrayDataType::kBool>().data;
  CHECK_EQ(predicate_data.size(), 1);
  const bool predicate_value = predicate_data[0];

  // From the TensorFlow docs on .switch() in
  // third_party/tensorflow/python/ops/control_flow_ops.py
  //
  //    If `pred` is false, the `data` input is forwarded to the first output.
  //    Otherwise, the data goes to the second output.
  //
  // Note that this comment used to say the opposite and was recently fixed:
  // https://github.com/tensorflow/tensorflow/commit/bc456e361d49d1d89a74b80060c70efb51fd7d87#diff-76ab9dafbe12c20ddc3769c6b108986c
  const int selected_output_index = predicate_value ? 1 : 0;
  const int nonselected_output_index = predicate_value ? 0 : 1;

  // Update the edges of the graph ahead of removing the node:
  // edges that were pointing to the selected output, should instead
  // point to the input of the Switch node.
  for (const auto& other_op : model->operators) {
    for (auto& input : other_op->inputs) {
      if (input == switch_op->outputs[selected_output_index]) {
        input = switch_op->inputs[0];
      }
    }
  }

  // There remains to handle the edges that were pointing to the nonselected
  // output. We will just discard those edges. Concretely, at the moment,
  // our only examples of graphs with Switch nodes have them feeding into Merge
  // nodes, so what we're saying here is that we'll make the convention,
  // in our toco internal representation, that Merge nodes with only 1 input
  // are Merge nodes that have been resolved already and should be have as
  // Identity nodes, simply forwarding their input.
  //
  for (const auto& other_op : model->operators) {
    auto input_it = other_op->inputs.begin();
    while (input_it != other_op->inputs.end()) {
      if (*input_it == switch_op->outputs[nonselected_output_index]) {
        // Let us guard our assumption that only Merge nodes consume the outputs
        // of Switch nodes:
        if (other_op->type != OperatorType::kMerge) {
          return ::tensorflow::Status(
              ::tensorflow::error::FAILED_PRECONDITION,
              ::absl::StrCat(
                  "Found ", HelpfulOperatorTypeName(*other_op),
                  " as non-selected output from Switch, but only "
                  "Merge supported. Control flow ops like Switch and Merge are "
                  "not generally supported. We are working on fixing this, "
                  "please see the Github issue at "
                  "https://github.com/tensorflow/tensorflow/issues/28485."));
        }
        input_it = other_op->inputs.erase(input_it);
      } else {
        ++input_it;
      }
    }
  }

  // Remove the output arrays if they are now unused.
  for (int i = 0; i < 2; i++) {
    if (!GetOpWithInput(*model, switch_op->outputs[i])) {
      model->EraseArray(switch_op->outputs[i]);
    }
  }
  // Remove input arrays if they are only used by the switch itself and aren't
  // the output of another op (will get handled by RemoveUnusedOp in that case).
  for (const auto& input : switch_op->inputs) {
    if (CountOpsWithInput(*model, input) == 1 &&
        !GetOpWithOutput(*model, input)) {
      model->EraseArray(input);
    }
  }
  // Remove the switch node itself.
  AddMessageF("Removing already-resolved %s", LogName(*switch_op));
  DeleteOpAndArrays(model, switch_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
