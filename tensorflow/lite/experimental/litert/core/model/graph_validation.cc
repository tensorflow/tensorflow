// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/model/graph_validation.h"

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"

namespace litert::internal {

bool ValidateLocalTopology(const LiteRtOpT& litert_op) {
  // Check number of in edges equals number of inputs and each input index
  // appears on an in edge.
  for (auto i = 0; i < litert_op.Inputs().size(); ++i) {
    const auto& litert_tensor = litert_op.Input(i);

    auto input_use =
        GetTensorUses(litert_tensor, FindUseInds(litert_tensor, litert_op));

    if (!ContainsIf(input_use.cbegin(), input_use.cend(),
                    [i](auto u) { return u.second == i; })) {
      LITERT_LOG(LITERT_WARNING,
                 "Input tensor %d not connected to op on correct index.", i);
      return false;
    }
  }

  // Similar to above for outputs.
  for (auto i = 0; i < litert_op.Outputs().size(); ++i) {
    const auto& litert_tensor = litert_op.Output(i);

    if (litert_tensor.DefiningOp() != &litert_op) {
      LITERT_LOG(LITERT_WARNING, "Output back edge doesn't refer to this op.");
      return false;
    }

    if (litert_tensor.DefiningOpOutInd() != i) {
      LITERT_LOG(LITERT_WARNING, "Output back edge ind is incorrect.");
      return false;
    }
  }

  return true;
}

bool ValidateSubgraphIO(const LiteRtSubgraphT& litert_subgraph) {
  auto num_implied_inputs = 0;
  auto num_implied_outputs = 0;
  for (auto* tensor : litert_subgraph.Tensors()) {
    const auto implied_out = tensor->NumUses() == 0;
    const auto implied_in =
        !IsConstant(*tensor) && tensor->DefiningOp() == nullptr;

    if (implied_out && implied_in) {
      LITERT_LOG(LITERT_WARNING, "Graph contains a dead tensor");
      return false;
    }

    const auto is_io = IsIO(litert_subgraph, *tensor);

    if (implied_in) {
      if (!is_io) {
        LITERT_LOG(LITERT_WARNING,
                   "Implied input not reflected in subgraph io %lu",
                   tensor - litert_subgraph.Tensors().at(0));
        return false;
      }
      ++num_implied_inputs;
    }

    if (implied_out) {
      if (!is_io) {
        LITERT_LOG(LITERT_WARNING,
                   "Implied output not reflected in subgraph io");
        return false;
      }
      ++num_implied_outputs;
    }
  }

  if (num_implied_inputs != litert_subgraph.NumInputs()) {
    LITERT_LOG(
        LITERT_WARNING,
        "Number of implied %lu inputs not equal to number of actual inputs %lu",
        num_implied_inputs, litert_subgraph.NumInputs());
    return false;
  }

  if (num_implied_outputs != litert_subgraph.NumOutputs()) {
    LITERT_LOG(LITERT_WARNING,
               "Number of implied %lu outputs not equal to number of actual "
               "outputs %lu",
               num_implied_outputs, litert_subgraph.NumOutputs());
    return false;
  }

  return true;
}

}  // namespace litert::internal
