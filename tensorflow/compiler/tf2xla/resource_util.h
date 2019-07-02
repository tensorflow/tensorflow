/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
// AnalyzeResourceOpSourcePath analyzes a Tensorflow graph and finds all
// operations that creates Stack/TensorArray resources and all the operations
// that consume resource created by them.
//
// Note that _Arg nodes that introduce resources are not considered sources.
// Note again that Control Flow v1 nodes (Enter/Exit/Switch/Merge/NextIteration)
// are not supported. Graphs contain these nodes cause analysis failures.
// However Control Flow v2 nodes (While/If) will be supported.
//
// TODO(b/135628319): Support analyzing function call and functional while/if
// as pass-through ops.
//
// For example, consider following subgraph:
//
// TensorArrayOp -> Identity -> TensorArrayWriteOp
//
// It should be able to tell that TensorArrayWriteOp actually operates on the
// resource created by TensorArrayOp even though there might be
// non-resource-specific operations like Identity (or other pass-through
// operations).
//
// sources_paths maps the nodes that creates resources to all nodes that operate
// on corresponding resource, not including sources themselves. It is cleared
// upon calling this method.
Status AnalyzeResourceOpSourcePath(
    const Graph* graph,
    absl::flat_hash_map<const Node*, absl::flat_hash_set<const Node*>>*
        sources_paths);

}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_
