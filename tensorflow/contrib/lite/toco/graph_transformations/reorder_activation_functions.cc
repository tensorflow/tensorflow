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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/runtime/types.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ReorderActivationFunctions::Run(Model* model, std::size_t op_index) {
  const auto ac_it = model->operators.begin() + op_index;
  std::unique_ptr<Operator>& ac_op = *ac_it;
  DCHECK(ac_op);

  if (ac_op->type != OperatorType::kRelu6 &&
      ac_op->type != OperatorType::kRelu1 &&
      ac_op->type != OperatorType::kRelu) {
    return false;
  }

  auto exchange_it = FindOpWithOutput(*model, ac_op->inputs[0]);
  if (exchange_it == model->operators.end()) return false;
  // Find the op producing the array passed to this activation function
  std::unique_ptr<Operator>& exchange_op = *exchange_it;
  DCHECK(exchange_op);

  // Allow activation functions to move up over any operator that does not
  // change the values.
  switch (exchange_op->type) {
    case OperatorType::kExpandDims:
    case OperatorType::kSqueeze:
    case OperatorType::kTensorFlowReshape:
    case OperatorType::kTranspose:
      break;
    default:
      return false;
  }

  DCHECK_EQ(exchange_op->outputs[0], ac_op->inputs[0]);
  const auto exchange_op_input = exchange_op->inputs[0];
  const auto intermediate_array = exchange_op->outputs[0];
  const auto ac_op_output = ac_op->outputs[0];

  int count_ops_consuming_output =
      CountOpsWithInput(*model, intermediate_array);
  DCHECK_GE(count_ops_consuming_output, 1);
  if (count_ops_consuming_output > 1) {
    AddMessageF(
        "Not exchanging activation function with %s because it is consumed by "
        "more than 1 other operator",
        LogName(*exchange_op));
    return false;
  }

  // If the ac_op was originally producing an output_array we can't trivially
  // reorder as otherwise the output array name would change and break
  // downstream assumptions. To work around that we perform some renaming below
  // in that case at the cost of a bit more confusing array names in this rare
  // case.
  bool is_ac_op_output =
      std::find(model->flags.output_arrays().begin(),
                model->flags.output_arrays().end(),
                ac_op_output) != model->flags.output_arrays().end();
  if (is_ac_op_output) {
    // To preserve the output array name of the activation function we need to
    // create a temporary to use to pass between ac->ex.
    //
    // Original:
    //  (a) -> EX -> (b) -> AC -> (c)
    // Now:
    //  (a) -> AC -> (c') -> EX -> (c)
    AddMessageF(
        "Exchanging activation function %s with %s but renaming to preserve "
        "output array %s",
        LogName(*ac_op), LogName(*exchange_op), ac_op->outputs[0]);

    auto renamed_ac_op_output =
        AvailableArrayName(*model, ac_op_output + "_exchange");
    ac_op->inputs[0] = exchange_op_input;
    ac_op->outputs[0] = renamed_ac_op_output;
    model->EraseArray(exchange_op->outputs[0]);
    exchange_op->inputs[0] = renamed_ac_op_output;
    exchange_op->outputs[0] = ac_op_output;
  } else {
    // Simply swap the order and update consumers to use the exchange_op output
    // array (b).
    //
    // Original:
    //  (a) -> EX -> (b) -> AC -> (c)
    // Now:
    //  (a) -> AC -> (c) -> EX -> (b)
    AddMessageF("Exchanging activation function %s with %s", LogName(*ac_op),
                LogName(*exchange_op));

    Operator* consumer = GetFirstOpWithInput(*model, ac_op_output);
    while (consumer) {
      for (int i = 0; i < consumer->inputs.size(); ++i) {
        if (consumer->inputs[i] == ac_op_output) {
          consumer->inputs[i] = intermediate_array;
        }
      }
      consumer = GetFirstOpWithInput(*model, ac_op_output);
    }
    ac_op->inputs[0] = exchange_op_input;
    exchange_op->inputs[0] = ac_op_output;
  }

  // Clear shapes; this will allow shape propagation to fix the sizes for us.
  model->GetOrCreateArray(ac_op->outputs[0]).clear_shape();
  model->GetOrCreateArray(exchange_op->outputs[0]).clear_shape();

  // Finally, reorder operators.  Note that this only works when there are no
  // other direct descendents of the exchange_op.
  ac_op.swap(exchange_op);

  return true;
}

}  // namespace toco
