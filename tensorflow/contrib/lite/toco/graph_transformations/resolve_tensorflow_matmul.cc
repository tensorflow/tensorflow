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
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

bool ResolveTensorFlowMatMul::Run(Model* model, std::size_t op_index) {
  auto matmul_it = model->operators.begin() + op_index;
  if (matmul_it->get()->type != OperatorType::kMatMul) {
    return false;
  }
  const auto* matmul_op =
      static_cast<const TensorFlowMatMulOperator*>(matmul_it->get());

  // Reorder the axes on the second input. TensorFlow uses row-major ordering
  // on both inputs, however this is inefficient for the FullyConnected
  // operator. We'll transpose the second input to be in column-major order now
  // and let constant propagation optimize things (if possible).
  auto* transpose_op = new TransposeOperator;
  transpose_op->inputs = {
      matmul_op->inputs[1],
      CreateInt32Array(
          model,
          AvailableArrayName(*model, matmul_op->inputs[1] + "/transpose/perm"),
          {1, 0})};
  transpose_op->outputs = {
      AvailableArrayName(*model, matmul_op->inputs[1] + "/transpose")};
  model->GetOrCreateArray(transpose_op->outputs[0]);
  model->operators.emplace(matmul_it, transpose_op);

  // Refresh iterator.
  matmul_it = model->operators.begin();
  for (; matmul_it != model->operators.end(); ++matmul_it) {
    if (matmul_it->get() == matmul_op) {
      break;
    }
  }
  DCHECK_EQ(matmul_it->get(), matmul_op);

  string input_lhs = matmul_op->inputs[0];
  string input_rhs = transpose_op->outputs[0];

  // Construct the new FullyConnectedOperator.
  auto* fc_op = new FullyConnectedOperator;
  fc_op->outputs = matmul_op->outputs;

  // Insert the newly constructed FullyConnectedOperator.
  model->operators.emplace(matmul_it, fc_op) + 1;

  // Find the op producing the array passed to this MatMul
  auto previous_op_it = model->operators.begin();
  bool found = false;
  for (; previous_op_it != model->operators.end(); ++previous_op_it) {
    for (const auto& output : (*previous_op_it)->outputs) {
      if (output == matmul_op->inputs[0]) {
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }
  Operator* previous_op = (found) ? previous_op_it->get() : nullptr;

  // Refresh iterator.
  matmul_it = model->operators.begin();
  for (; matmul_it != model->operators.end(); ++matmul_it) {
    if (matmul_it->get() == matmul_op) {
      break;
    }
  }
  DCHECK_EQ(matmul_it->get(), matmul_op);

  // The way that TensorFlow encodes FullyConnected ops is as a pair
  // (Reshape, MatMul), so we want to remove the Reshape op and rewrite the
  // MatMul op as a FullyConnected. However, TensorFlow skips the Reshape ops if
  // the input doesn't need reshaping, so we can't just match (Reshape, MatMul)
  // pairs.
  if (previous_op && previous_op->type == OperatorType::kReshape) {
    AddMessageF("Combining %s and %s into %s", LogName(*previous_op),
                LogName(*matmul_op), LogName(*fc_op));
    const auto& previous_op_output = previous_op->outputs[0];
    if (CountOpsWithInput(*model, previous_op_output) == 1) {
      model->EraseArray(previous_op_output);
    }
    CHECK_EQ(previous_op->inputs.size(), 2);
    input_lhs = previous_op->inputs[0];
    // Only remove Reshape node if no other node uses its output.
    if (CountOpsWithInput(*model, previous_op_output) == 1) {
      const auto& previous_op_shape = previous_op->inputs[1];
      if (CountOpsWithInput(*model, previous_op_shape) == 1 &&
          !GetOpWithOutput(*model, previous_op_shape)) {
        model->EraseArray(previous_op_shape);
      }
      model->operators.erase(previous_op_it);
    }

    // We may have just invalidated matmul_it, so let's refresh it now.
    matmul_it = model->operators.begin();
    for (; matmul_it != model->operators.end(); ++matmul_it) {
      if (matmul_it->get() == matmul_op) {
        break;
      }
    }
    CHECK(matmul_it != model->operators.end());
    CHECK(matmul_it->get() == matmul_op);
  } else {
    AddMessageF("Replacing %s by a FullyConnected operator",
                LogName(*matmul_op));
  }

  fc_op->inputs = {input_lhs, input_rhs};

  // erase the MatMul operator
  model->operators.erase(matmul_it);
  return true;
}

}  // namespace toco
