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

#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

absl::Status IdentifyL2Pool::Run(Model* model, std::size_t op_index,
                                 bool* modified) {
  *modified = false;
  const auto sqrt_it = model->operators.begin() + op_index;
  const auto* sqrt_op = sqrt_it->get();
  if (sqrt_op->type != OperatorType::kSqrt) {
    return absl::OkStatus();
  }

  CHECK_EQ(sqrt_op->inputs.size(), 1);
  CHECK_EQ(sqrt_op->outputs.size(), 1);

  const AveragePoolOperator* avpool_op;
  const Operator* square_op;

  Operator* prev_to_sqrt_op = GetOpWithOutput(*model, sqrt_op->inputs[0]);
  if (prev_to_sqrt_op == nullptr) {
    AddMessageF(
        "Giving up trying to identify L2Pool subgraph: "
        "expected AveragePool op, but Sqrt op has no preceding op");
    return absl::OkStatus();
  }

  if (prev_to_sqrt_op->type != OperatorType::kAveragePool) {
    AddMessageF(
        "Giving up trying to identify L2Pool subgraph: "
        "expected AveragePool op, got %s",
        LogName(*prev_to_sqrt_op));
    return absl::OkStatus();
  }

  avpool_op = static_cast<const AveragePoolOperator*>(prev_to_sqrt_op);
  CHECK_EQ(avpool_op->inputs.size(), 1);

  square_op = GetOpWithOutput(*model, avpool_op->inputs[0]);
  CHECK_EQ(square_op->inputs.size(), 1);
  if (square_op->type != OperatorType::kSquare) {
    AddMessageF(
        "Giving up trying to identify L2Pool subgraph: "
        "expected Square op, got %s",
        LogName(*square_op));
    return absl::OkStatus();
  }

  // Create and emplace L2Pool node.
  auto* l2pool_op = new L2PoolOperator;

  l2pool_op->inputs = {square_op->inputs[0]};
  l2pool_op->outputs = sqrt_op->outputs;

  l2pool_op->padding.type = avpool_op->padding.type;
  // Note that we do not setup avpool_op->padding.fixed here.  This is done by
  // the PropagateFixedSizes graph transformation.

  l2pool_op->stride_height = avpool_op->stride_height;
  l2pool_op->stride_width = avpool_op->stride_width;
  l2pool_op->kheight = avpool_op->kheight;
  l2pool_op->kwidth = avpool_op->kwidth;
  model->operators.emplace(sqrt_it, l2pool_op);

  AddMessageF("Creating %s replacing equivalent subgraph", LogName(*l2pool_op));

  DeleteOpAndArrays(model, square_op);
  DeleteOpAndArrays(model, avpool_op);
  DeleteOpAndArrays(model, sqrt_op);

  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
