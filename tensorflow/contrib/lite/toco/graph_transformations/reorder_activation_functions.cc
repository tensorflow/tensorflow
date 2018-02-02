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

  if (exchange_op->type != OperatorType::kTensorFlowReshape) {
    return false;
  }

  DCHECK_EQ(exchange_op->outputs[0], ac_op->inputs[0]);
  const auto& exchange_op_input = exchange_op->inputs[0];
  const auto& intermediate_array = exchange_op->outputs[0];
  const auto& ac_op_output = ac_op->outputs[0];

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

  // Rewire by changing inputs, including all consumers.
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

  // Finally, reorder operators.  Note that this only works when there are no
  // other direct descendents of the exchange_op.
  ac_op.swap(exchange_op);

  return true;
}

}  // namespace toco
