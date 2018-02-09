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

namespace {

void RemoveTileOperator(Model* model, Operator* tile_op, Operator* binary_op,
                        int operand_index) {
  CHECK(tile_op->type == OperatorType::kTensorFlowTile);
  CHECK_EQ(binary_op->inputs.size(), 2);
  CHECK_EQ(tile_op->inputs.size(), 2);
  const string tile_multiplier_array = tile_op->inputs[1];
  const string tile_output_array = tile_op->outputs[0];
  binary_op->inputs[operand_index] = tile_op->inputs[0];
  auto tile_it = model->operators.begin();
  for (; tile_it != model->operators.end(); ++tile_it) {
    if (tile_it->get() == tile_op) {
      break;
    }
  }
  CHECK(tile_it != model->operators.end());
  CHECK(tile_it->get() == tile_op);
  model->operators.erase(tile_it);
  if (!CountOpsWithInput(*model, tile_multiplier_array) &&
      !GetOpWithOutput(*model, tile_multiplier_array)) {
    model->EraseArray(tile_multiplier_array);
  }
  if (!CountOpsWithInput(*model, tile_output_array)) {
    model->EraseArray(tile_output_array);
  }
}
}  // namespace

bool ResolveTensorFlowTile::Run(Model* model, std::size_t op_index) {
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();
  // Test for binary ops of types that we know how to resolve
  if (binary_op->inputs.size() != 2) {
    return false;
  }
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return false;
  }

  Operator* const op[2] = {
      GetOpWithOutput(*model, binary_op->inputs[0]),
      GetOpWithOutput(*model, binary_op->inputs[1]),
  };

  // In the unlikely case where both operands are Tile, we can't infer the
  // output
  // size without the Tile nodes, so we have to bail out.
  if (op[0] && op[0]->type == OperatorType::kTensorFlowTile && op[1] &&
      op[1]->type == OperatorType::kTensorFlowTile) {
    return false;
  }

  for (int i = 0; i < 2; i++) {
    if (op[i] && op[i]->type == OperatorType::kTensorFlowTile) {
      // We can only remove a Tile operator is no other op than the present
      // binary op was consuming its tiled output.
      if (CountOpsWithInput(*model, binary_op->inputs[i]) == 1) {
        AddMessageF("Removing %s", LogName(*op[i]));
        RemoveTileOperator(model, op[i], binary_op, i);
        return true;
      }
    }
  }
  return false;
}

}  // namespace toco
