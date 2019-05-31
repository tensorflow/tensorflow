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
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {

// Reroute all edges involving a given discardable array to another
// array instead. from_array is assumed to be discardable, and consequently
// this only updates operator edges (since discardable arrays only
// appear there, and not e.g. in model flags).
void Reroute(const string& from, const string& to, Model* model) {
  for (const auto& op : model->operators) {
    for (auto& output : op->outputs) {
      if (output == from) {
        output = to;
      }
    }
    for (auto& input : op->inputs) {
      if (input == from) {
        input = to;
      }
    }
  }
  const Array& from_array = model->GetArray(from);
  Array& to_array = model->GetOrCreateArray(to);
  // Preserve minmax information if to_array didn't already have any.
  if (from_array.minmax && !to_array.minmax) {
    to_array.GetOrCreateMinMax() = from_array.GetMinMax();
    // If we're copying minmax info, then we should also be copying
    // narrow_range, which affects how minmax info is to be interpreted.
    to_array.narrow_range = from_array.narrow_range;
  }
  // Separately, also preserve final_data_type if to_array didn't already
  // have any.
  if (from_array.final_data_type != ArrayDataType::kNone &&
      to_array.final_data_type == ArrayDataType::kNone) {
    to_array.final_data_type = from_array.final_data_type;
  }
}

}  // namespace

bool RemoveTrivialPassthroughOp(GraphTransformation* transformation,
                                Model* model, std::size_t op_index,
                                int input_index) {
  auto passthru_it = model->operators.begin() + op_index;
  auto* passthru_op = passthru_it->get();
  CHECK_EQ(passthru_op->outputs.size(), 1);
  CHECK_GE(passthru_op->inputs.size(), 1);

  int main_input_array_index = 0;
  if (input_index != -1) {
    main_input_array_index = input_index;
  } else {
    // We call 'main input' the unique nonconstant input array if there is one,
    // or else the 0-th input.
    int count_nonconstant_input_arrays = 0;
    for (int i = 0; i < passthru_op->inputs.size(); i++) {
      if (!model->GetArray(passthru_op->inputs[i]).buffer) {
        count_nonconstant_input_arrays++;
        if (count_nonconstant_input_arrays == 1) {
          main_input_array_index = i;
        }
      }
    }
  }

  const string main_input_name = passthru_op->inputs[main_input_array_index];
  const string output_name = passthru_op->outputs[0];

  // Build the list of all input and output arrays of the passthrough node
  // that we are considering removing. Any of these arrays is a candidate
  // for being removed as well, if nothing else references it. Doing that
  // arrays-removal together with the passthrough-node-removal proved too
  // error-prone.
  std::vector<string> removal_candidates;
  for (const string& input : passthru_op->inputs) {
    removal_candidates.push_back(input);
  }
  removal_candidates.push_back(output_name);

  if (IsDiscardableArray(*model, output_name)) {
    transformation->AddMessageF(
        "Removing %s, keeping its non-constant input array %s and removing %s",
        LogName(*passthru_op), main_input_name, output_name);
    Reroute(output_name, main_input_name, model);
  } else if (IsDiscardableArray(*model, main_input_name) &&
             !IsConstantParameterArray(*model, main_input_name)) {
    transformation->AddMessageF(
        "Removing %s, keeping its output array %s and removing non-constant "
        "input %s",
        LogName(*passthru_op), output_name, main_input_name);
    Reroute(main_input_name, output_name, model);
  } else {
    transformation->AddMessageF(
        "Cannot remove %s, neither its main input nor its output may be "
        "discarded",
        LogName(*passthru_op));
    if (passthru_op->type != OperatorType::kReshape &&
        model->GetArray(main_input_name).has_shape()) {
      // We can't remove either array but we can remove the op. Converting it to
      // a reshape gives us some hope of later on fixing that (either in the
      // final runtime or as an additional fixup step).
      //
      // Note that we don't try to insert copies in place of reshapes as the
      // copy itself is a trivial reshape and we'd go into an infinite loop!
      transformation->AddMessageF("Replacing with a copy (reshape) instead");
      InsertCopyOperator(model, main_input_name, output_name);
      // To avoid using invalidated iterator, evaluate passthru_it again.
      passthru_it = model->operators.begin() + op_index;
    } else {
      return false;
    }
  }

  // Remove the pass-through node.
  CHECK_EQ(passthru_it->get(), passthru_op);
  model->operators.erase(passthru_it);

  // Remove any array that is no longer used.
  for (const string& removal_candidate : removal_candidates) {
    bool is_referenced = false;
    for (const auto& array : model->flags.input_arrays()) {
      if (array.name() == removal_candidate) {
        is_referenced = true;
      }
    }
    for (const auto& array_name : model->flags.output_arrays()) {
      if (array_name == removal_candidate) {
        is_referenced = true;
      }
    }
    for (const auto& op : model->operators) {
      for (const string& input : op->inputs) {
        if (input == removal_candidate) {
          is_referenced = true;
        }
      }
      for (const string& output : op->outputs) {
        if (output == removal_candidate) {
          is_referenced = true;
        }
      }
    }
    if (!is_referenced) {
      model->EraseArray(removal_candidate);
    }
  }

  return true;
}

}  // namespace toco
