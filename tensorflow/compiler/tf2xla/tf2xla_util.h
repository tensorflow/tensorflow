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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_

#include <unordered_map>

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// ValidateConfig returns OK iff config is valid.
Status ValidateConfig(const tf2xla::Config& config);

// Modifies <graph_def> to include placeholders for each fed tensor, and
// update references to the fed tensors to refer to the placeholders.
// The existing nodes referenced by the feeds are not removed or modified
// (except where their input edges are modified by the replacement of other
// feeds).
Status AddPlaceholdersForFeeds(
    const tf2xla::Config& config, const OpRegistryInterface* op_registry,
    std::unordered_map<string, string>* feed_remapping, GraphDef* graph_def);

// Returns in <out> a copy of <in>, pruned to only include fetches from
// <config>.
Status PruneGraphDefInto(const tf2xla::Config& config, const GraphDef& in,
                         GraphDef* out);

// Returns node:port for the given <id>.
string TensorIdToString(const tf2xla::TensorId& id);

// Updates the sharding of <n> based on the sharding of its neighbors.
// If <out_edges> is true, outgoing edges from <n> are considered; else incoming
// edges are considered.
Status SetNodeShardingFromNeighbors(Node* n, bool out_edges);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
