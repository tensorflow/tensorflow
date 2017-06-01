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

#include "tensorflow/core/grappler/op_types.h"

namespace tensorflow {
namespace grappler {

bool IsConcat(const NodeDef& node) {
  const auto op = node.op();
  return op == "Concat" || op == "ConcatV2";
}

bool IsConstant(const NodeDef& node) {
  const auto op = node.op();
  return op == "Const";
}

bool IsDequeueOp(const NodeDef& node) {
  static const std::set<std::string> dequeue_ops = {
      "QueueDequeueManyV2", "QueueDequeueMany",   "QueueDequeueV2",
      "QueueDequeue",       "QueueDequeueUpToV2", "QueueDequeueUpTo"};
  return dequeue_ops.count(node.op()) > 0;
}

bool IsMerge(const NodeDef& node) {
  const auto op = node.op();
  return op == "Merge";
}

bool IsPlaceholder(const NodeDef& node) {
  const auto op = node.op();
  return op == "Placeholder" || op == "PlaceholderV2" ||
         op == "PlaceholderWithDefault";
}

bool IsTranspose(const NodeDef& node) {
  const auto op = node.op();
  return op == "Transpose";
}

bool IsVariable(const NodeDef& node) {
  const auto op = node.op();
  return op == "Variable" || op == "VariableV2" || op == "AutoReloadVariable" ||
         op == "VarHandleOp" || op == "TemporaryVariable";
}

}  // end namespace grappler
}  // end namespace tensorflow
