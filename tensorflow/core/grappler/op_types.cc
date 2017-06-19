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
  const auto& op = node.op();
  return op == "QueueDequeueManyV2" || op == "QueueDequeueMany" ||
         op == "QueueDequeueV2" || op == "QueueDequeue" ||
         op == "QueueDequeueUpToV2" || op == "QueueDequeueUpTo";
}

bool IsIdentity(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Identity";
}

bool IsMerge(const NodeDef& node) {
  const auto op = node.op();
  return op == "Merge";
}

bool IsNoOp(const NodeDef& node) {
  const auto op = node.op();
  return op == "NoOp";
}

bool IsPlaceholder(const NodeDef& node) {
  const auto op = node.op();
  return op == "Placeholder" || op == "PlaceholderV2" ||
         op == "PlaceholderWithDefault";
}

bool IsRecv(const NodeDef& node) {
  const auto op = node.op();
  return op == "_Recv";
}

bool IsReduction(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Sum" || op == "Prod" || op == "Min" || op == "Max" ||
         op == "Mean" || op == "Any" || op == "All";
}

bool IsSend(const NodeDef& node) {
  const auto op = node.op();
  return op == "_Send";
}

bool IsSwitch(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Switch";
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
