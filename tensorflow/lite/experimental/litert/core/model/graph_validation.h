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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_GRAPH_VALIDATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_GRAPH_VALIDATION_H_

#include <algorithm>

#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_graph.h"

// Helper functions for validating the structure of IR graphs.

namespace litert::internal {

// Checks the double-linked edges to immediate neighbors are valid.
bool ValidateLocalTopology(const LiteRtOpT& litert_op);

// Runs ValidateLocalTopology across given LiteRtOp iterator.
template <class OpIt>
bool ValidateLocalTopology(OpIt start, OpIt end) {
  return std::all_of(start, end,
                     [](const auto* op) { return ValidateLocalTopology(*op); });
}

// Checks the following are bijections:
// * non-const tensor with no defining op <-> subgraph input
// * tensor with no users <-> subgraph output (assuming no side effect ops)
// These are used to figure out the i/o signatures when building a subgraph
// from scratch.
bool ValidateSubgraphIO(const LiteRtSubgraphT& litert_subgraph);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_GRAPH_VALIDATION_H_
