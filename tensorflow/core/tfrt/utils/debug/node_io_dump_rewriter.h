/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_DEBUG_NODE_IO_DUMP_REWRITER_H_
#define TENSORFLOW_CORE_TFRT_UTILS_DEBUG_NODE_IO_DUMP_REWRITER_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace tfrt_stub {

// Rewrites `graph` by inserting dump nodes for `nodes_to_dump`. During graph
// execution, the inputs and outputs of `nodes_to_dump` will be dumped to the
// folder specified by env var `TF_DUMP_GRAPH_PREFIX`.
absl::Status InsertDumpOps(
    Graph& graph, const absl::flat_hash_set<std::string>& nodes_to_dump,
    absl::string_view dump_dir = "");
// Similar to the above, but rewrites a `meta_graph_def`.
absl::Status InsertDumpOps(
    MetaGraphDef& meta_graph_def,
    const absl::flat_hash_set<std::string>& nodes_to_dump,
    absl::string_view dump_dir = "");

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_DEBUG_NODE_IO_DUMP_REWRITER_H_
