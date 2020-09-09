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

TransposeOperator* FindTransposeOpWithInput(const Model& model,
                                            const std::string& array_name) {
  for (auto it = model.operators.begin(); it != model.operators.end(); ++it) {
    Operator* op = it->get();
    if (op->type != OperatorType::kTranspose) {
      continue;
    }
    if (op->inputs[0] != array_name) {
      continue;
    }
    const auto& permutation_array = model.GetArray(op->inputs[1]);
    if (permutation_array.data_type != ArrayDataType::kInt32) {
      continue;
    }
    const auto& permutation_data =
        permutation_array.GetBuffer<ArrayDataType::kInt32>().data;
    if (permutation_data.size() != 2) {
      continue;
    }
    if (permutation_data[0] != 1 || permutation_data[1] != 0) {
      continue;
    }
    return static_cast<TransposeOperator*>(op);
  }
  return nullptr;
}

}  // namespace

::tensorflow::Status ResolveTensorFlowMatMul::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  auto matmul_it = model->operators.begin() + op_index;
  if (matmul_it->get()->type != OperatorType::kMatMul) {
    return ::tensorflow::Status::OK();
  }
  const auto* matmul_op =
      static_cast<const TensorFlowMatMulOperator*>(matmul_it->get());

  auto refresh_matmul_iterator = [&model, &matmul_it, &matmul_op]() {
    matmul_it = std::find_if(model->operators.begin(), model->operators.end(),
                             [matmul_op](const std::unique_ptr<Operator>& op) {
                               return op.get() == matmul_op;
                             });
    DCHECK_EQ(matmul_it->get(), matmul_op);
  };

  std::string input_lhs = matmul_op->inputs[0];
  std::string input_rhs = matmul_op->inputs[1];

  // Handle `transpose_a` with best effort: If the dimension of lhs is known,
  // insert a `Transpose` op.
  if (matmul_op->transpose_a) {
    Array& lhs_array = model->GetArray(input_lhs);
    if (!lhs_array.has_shape()) {
      AddMessageF(
          "Not replacing %s by a FullyConnected operator, because it has "
          "the transpose_a attribute and LHS has no shape",
          LogName(*matmul_op));
      return ::tensorflow::Status::OK();
    }

    int dimensions_count = lhs_array.shape().dimensions_count();
    if (dimensions_count < 2) {
      return ::tensorflow::errors::InvalidArgument(
          "Inputs of MatMul should have dimension >= 2. Got %d dimensions",
          dimensions_count);
    }

    // Create a permutation vector to exchange the last 2 dimensions.
    // E.g. For 4D, create [0, 1, 3, 2].
    std::vector<int> perm;
    perm.reserve(dimensions_count);
    for (int i = 0; i < dimensions_count; ++i) {
      perm.push_back(i);
    }
    std::swap(perm[dimensions_count - 1], perm[dimensions_count - 2]);

    auto* transpose_op = new TransposeOperator;
    transpose_op->inputs = {
        input_lhs,
        CreateInt32Array(
            model, AvailableArrayName(*model, input_lhs + "/transpose/perm"),
            perm)};
    transpose_op->outputs = {
        AvailableArrayName(*model, input_lhs + "/transpose")};
    model->GetOrCreateArray(transpose_op->outputs[0]);
    model->operators.emplace(matmul_it, transpose_op);
    // Sanity check
    DCHECK_EQ(transpose_op, FindTransposeOpWithInput(*model, input_lhs));
    input_lhs = transpose_op->outputs[0];

    refresh_matmul_iterator();
  }

  // TODO(b/138662017): The following code assumes that RHS is 2D. This isn't
  // always true in TensorFlow.
  //
  // Reorder the axes on the second input. TensorFlow uses row-major ordering
  // on both inputs, however this is inefficient for the FullyConnected
  // operator. We'll transpose the second input to be in column-major order now
  // and let constant propagation optimize things (if possible).
  if (!matmul_op->transpose_b) {
    // Need to transpose input_rhs, by inserting a TransposeOperator.
    // First, check if there already is a TransposeOperator transposing that
    // array, so we can just reuse it.
    auto* transpose_op = FindTransposeOpWithInput(*model, input_rhs);
    if (!transpose_op) {
      AddMessageF(
          "While replacing %s by a FullyConnected operator, created new "
          "Transpose op wrapping RHS input array %s",
          LogName(*matmul_op), input_rhs);
      // No such TransposeOperator found. Create one now.
      transpose_op = new TransposeOperator;
      transpose_op->inputs = {
          input_rhs,
          CreateInt32Array(
              model, AvailableArrayName(*model, input_rhs + "/transpose/perm"),
              {1, 0})};
      transpose_op->outputs = {
          AvailableArrayName(*model, input_rhs + "/transpose")};
      model->GetOrCreateArray(transpose_op->outputs[0]);
      model->operators.emplace(matmul_it, transpose_op);
      // Sanity check
      DCHECK_EQ(transpose_op, FindTransposeOpWithInput(*model, input_rhs));
      refresh_matmul_iterator();
    } else {
      AddMessageF(
          "While replacing %s by a FullyConnected operator, reused existing "
          "Transpose op wrapping RHS input array %s",
          LogName(*matmul_op), input_rhs);
    }
    // Re-wire: have the matmul consume the transposed array.
    input_rhs = transpose_op->outputs[0];
  }

  // Construct the new FullyConnectedOperator.
  auto* fc_op = new FullyConnectedOperator;
  fc_op->inputs = {input_lhs, input_rhs};
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
    fc_op->inputs = {input_lhs, input_rhs};
    // Only remove Reshape node if no other node uses its output.
    if (CountOpsWithInput(*model, previous_op_output) == 1) {
      DeleteOpAndArrays(model, previous_op);
    }

    // We may have just invalidated matmul_it, so let's refresh it now.
    refresh_matmul_iterator();
  } else {
    AddMessageF("Replacing %s by a FullyConnected operator",
                LogName(*matmul_op));
  }


  // erase the MatMul operator
  model->operators.erase(matmul_it);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
