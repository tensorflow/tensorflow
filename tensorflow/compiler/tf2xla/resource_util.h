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
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
class ResourceUsageAnalysis {
 public:
  // NodeInfo is a triple of function_name:node_name:op to uniquely identity a
  // node in graph. ResourceUsageAnalysis uses it to represent resource sources
  // and users.
  class NodeInfo {
   public:
    absl::optional<std::string> function_name_;
    std::string node_name_;
    std::string op_;

    NodeInfo() {}

    NodeInfo(const absl::optional<std::string>& function_name,
             std::string node_name, std::string op)
        : function_name_(function_name),
          node_name_(std::move(node_name)),
          op_(std::move(op)) {}

    std::string DebugString() const {
      return absl::StrJoin({function_name_.value_or(""), node_name_, op_}, ":");
    }

    bool operator==(const NodeInfo& o) const {
      return function_name_ == o.function_name_ && node_name_ == o.node_name_ &&
             op_ == o.op_;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NodeInfo& o) {
      return H::combine(std::move(h), o.function_name_, o.node_name_, o.op_);
    }
  };

  // This method analyzes a Tensorflow graph and finds all operations that
  // create Stack/TensorArray resources and all the operations that consume
  // resource created by them.
  //
  // Note that _Arg nodes that introduce resources are not considered sources.
  // Note again that Control Flow v1 nodes
  // (Enter/Exit/Switch/Merge/NextIteration) are not supported. Graphs contain
  // these nodes cause analysis failures. However Control Flow v2 nodes
  // (While/If) will be supported.
  //
  // TODO(b/135628319): Support analyzing functional while/if as pass-through
  // ops.
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
  // source_to_path maps the nodes that creates resources to all nodes that
  // operate on the corresponding resource, not including sources themselves. It
  // is cleared upon calling this method.
  static Status Analyze(
      const Graph* graph, FunctionLibraryRuntime* lib_runtime,
      absl::flat_hash_map<NodeInfo, absl::flat_hash_set<NodeInfo>>*
          source_to_path);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_TF2XLA_RESOURCE_UTIL_H_
