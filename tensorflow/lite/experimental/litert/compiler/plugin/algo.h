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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_ALGO_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_ALGO_H_

#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"

namespace litert::internal {

// Identifies sub-DAGs of ops connected w.r.t. the use-def chain. Expects
// all "ops" belong to the same Subgraph. The ops in the input
// and output will always be the same.
std::vector<std::vector<LiteRtOp>> GroupPartitions(
    const std::vector<LiteRtOp>& ops);

// Outlines "partition" from "root" into the empty subgraph "slice". Assumes
// the partition is a valid sub-DAG, and replaces it with a single
// tfl.custom_op in "root". A reference to that op is returned.
LiteRtOp OutlinePartition(LiteRtSubgraphT& root, LiteRtSubgraph slice,
                          std::vector<LiteRtOp>& partition);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_ALGO_H_
