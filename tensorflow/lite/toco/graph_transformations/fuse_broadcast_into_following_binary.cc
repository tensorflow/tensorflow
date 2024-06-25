/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Returns true if the given op is strictly a broadcasting operation.
// This is commonly seen as a Concat of the same input multiple times, and is
// often generated from Tile ops that were converted via the
// convert_trivial_tile_to_concat transformation.
bool IsBroadcastingOp(const Model& model, Operator* op) {
  // Concatenation of identical inputs is usually a broadcast.
  if (op->type == OperatorType::kConcatenation) {
    // Verify that all inputs are the same.
    for (size_t i = 1; i < op->inputs.size(); ++i) {
      if (op->inputs[i] != op->inputs[0]) {
        return false;
      }
    }
    return true;
  }

  // There are other things we could look for (Stack/etc) when needed.
  return false;
}

}  // namespace

// Finds an operation that looks like a broadcast (concat of the same sources
// along the last dimension) and drops it by relying on the ability of certain
// binary ops to perform an implicit broadcast.
::tensorflow::Status FuseBroadcastIntoFollowingBinary::Run(Model* model,
                                                           std::size_t op_index,
                                                           bool* modified) {
  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  auto* binary_op = binary_it->get();

  // Test for binary ops of types that we know how to resolve
  if (binary_op->inputs.size() != 2) {
    return absl::OkStatus();
  }
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv) {
    return absl::OkStatus();
  }

  // NOTE: either of these ops may be nullptr if the input array is constant.
  Operator* const op[2] = {
      GetOpWithOutput(*model, binary_op->inputs[0]),
      GetOpWithOutput(*model, binary_op->inputs[1]),
  };

  // Check whether either input is a broadcast-like concat.
  bool is_op_0_broadcast = op[0] && IsBroadcastingOp(*model, op[0]);
  bool is_op_1_broadcast = op[1] && IsBroadcastingOp(*model, op[1]);
  if (!is_op_0_broadcast && !is_op_1_broadcast) {
    // Neither input is a broadcast-looking thing.
    AddMessageF("Neither input looks broadcasty");
    return absl::OkStatus();
  } else if (is_op_0_broadcast && is_op_1_broadcast) {
    AddMessageF(
        "Unable to fuse broadcast into %s as both inputs (%s, %s) are "
        "broadcasts",
        LogName(*binary_op), op[0] ? LogName(*op[0]) : "(?)",
        op[1] ? LogName(*op[1]) : "(?)");
    return absl::OkStatus();
  }
  int broadcast_index = is_op_0_broadcast ? 0 : 1;

  // Just pull out the input of the broadcast op and pass it directly to the
  // binary op.
  AddMessageF("Fusing broadcast op %s into the following binary %s",
              LogName(*op[broadcast_index]), LogName(*binary_op));
  binary_op->inputs[broadcast_index] = op[broadcast_index]->inputs[0];

  // We leave the broadcast op in; it'll get cleaned up if it's not used later.
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
