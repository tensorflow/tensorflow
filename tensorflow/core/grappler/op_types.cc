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

#include <unordered_set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {

bool IsAdd(const NodeDef& node) { return node.op() == "Add"; }

bool IsAddN(const NodeDef& node) { return node.op() == "AddN"; }

bool IsAvgPoolGrad(const NodeDef& node) { return node.op() == "AvgPoolGrad"; }

bool IsAssert(const NodeDef& node) { return node.op() == "Assert"; }

bool IsBiasAddGrad(const NodeDef& node) { return node.op() == "BiasAddGrad"; }

bool IsConcatOffset(const NodeDef& node) { return node.op() == "ConcatOffset"; }

bool IsConstant(const NodeDef& node) { return node.op() == "Const"; }

bool IsConv2D(const NodeDef& node) { return node.op() == "Conv2D"; }

bool IsConv2DBackpropFilter(const NodeDef& node) {
  return node.op() == "Conv2DBackpropFilter";
}

bool IsConv2DBackpropInput(const NodeDef& node) {
  return node.op() == "Conv2DBackpropInput";
}

bool IsDepthwiseConv2dNative(const NodeDef& node) {
  return node.op() == "DepthwiseConv2dNative";
}

bool IsDepthwiseConv2dNativeBackpropFilter(const NodeDef& node) {
  return node.op() == "DepthwiseConv2dNativeBackpropFilter";
}

bool IsDepthwiseConv2dNativeBackpropInput(const NodeDef& node) {
  return node.op() == "DepthwiseConv2dNativeBackpropInput";
}

bool IsDequeueOp(const NodeDef& node) {
  const auto& op = node.op();
  return op == "QueueDequeueManyV2" || op == "QueueDequeueMany" ||
         op == "QueueDequeueV2" || op == "QueueDequeue" ||
         op == "QueueDequeueUpToV2" || op == "QueueDequeueUpTo";
}

bool IsEnter(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Enter" || op == "RefEnter";
}

bool IsExit(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Exit" || op == "RefExit";
}

bool IsFloorMod(const NodeDef& node) { return node.op() == "FloorMod"; }

bool IsFusedBatchNormGradV1(const NodeDef& node) {
  return node.op() == "FusedBatchNormGrad";
}

bool IsIdentity(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Identity" || op == "RefIdentity";
}

bool IsMerge(const NodeDef& node) {
  const auto op = node.op();
  return op == "Merge" || op == "RefMerge";
}

bool IsMul(const NodeDef& node) { return node.op() == "Mul"; }

bool IsNoOp(const NodeDef& node) { return node.op() == "NoOp"; }

bool IsNextIteration(const NodeDef& node) {
  const auto& op = node.op();
  return op == "NextIteration" || op == "RefNextIteration";
}

bool IsPad(const NodeDef& node) { return node.op() == "Pad"; }

bool IsPlaceholder(const NodeDef& node) {
  const auto op = node.op();
  return op == "Placeholder" || op == "PlaceholderV2" ||
         op == "PlaceholderWithDefault";
}

bool IsRealDiv(const NodeDef& node) { return node.op() == "RealDiv"; }

bool IsReluGrad(const NodeDef& node) { return node.op() == "ReluGrad"; }

bool IsRecv(const NodeDef& node) { return node.op() == "_Recv"; }

bool IsReduction(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Sum" || op == "Prod" || op == "Min" || op == "Max" ||
         op == "Mean" || op == "Any" || op == "All";
}

bool IsReshape(const NodeDef& node) { return (node.op() == "Reshape"); }

bool IsRestore(const NodeDef& node) {
  return (node.op() == "Restore" || node.op() == "RestoreV2" ||
          node.op() == "RestoreSlice");
}

bool IsSend(const NodeDef& node) { return node.op() == "_Send"; }

bool IsSlice(const NodeDef& node) { return node.op() == "Slice"; }

bool IsSquaredDifference(const NodeDef& node) {
  return node.op() == "SquaredDifference";
}

bool IsSqueeze(const NodeDef& node) { return node.op() == "Squeeze"; }

bool IsStopGradient(const NodeDef& node) {
  const auto& op = node.op();
  return op == "StopGradient" || op == "PreventGradient";
}

bool IsSub(const NodeDef& node) { return node.op() == "Sub"; }

bool IsSum(const NodeDef& node) { return node.op() == "Sum"; }

bool IsSwitch(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Switch" || op == "RefSwitch";
}

bool IsTranspose(const NodeDef& node) { return node.op() == "Transpose"; }

bool IsVariable(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Variable" || op == "VariableV2" || op == "AutoReloadVariable" ||
         op == "VarHandleOp" || op == "ReadVariableOp";
}

bool IsFreeOfSideEffect(const NodeDef& node) {
  // Placeholders must be preserved to keep the graph feedable.
  if (IsPlaceholder(node)) {
    return false;
  }
  const OpDef* op_def = nullptr;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  if (op_def->is_stateful()) {
    return false;
  }
  // Nodes such as Assign or AssignAdd modify one of their inputs.
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return false;
    }
  }
  return true;
}

bool ModifiesFrameInfo(const NodeDef& node) {
  return IsEnter(node) || IsExit(node) || IsNextIteration(node);
}

#define OPDEF_PROPERTY_HELPER(PROPERTY_CAP, PROPERTY)                      \
  bool Is##PROPERTY_CAP(const NodeDef& node) {                             \
    if (node.op() == "Add") {                                              \
      /* Workaround for "Add" not being marked is_commutative and */       \
      /* is_aggregate. (See cl/173915048). */                              \
      const auto type = GetDataTypeFromAttr(node, "T");                    \
      return type != DT_INVALID && type != DT_STRING;                      \
    }                                                                      \
    const OpDef* op_def = nullptr;                                         \
    Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def); \
    return status.ok() && op_def->is_##PROPERTY();                         \
  }

OPDEF_PROPERTY_HELPER(Aggregate, aggregate)
OPDEF_PROPERTY_HELPER(Commutative, commutative)

bool IsInvolution(const NodeDef& node) {
  const std::unordered_set<string> involution_ops{
      "Conj", "Reciprocal", "Invert", "Neg", "LogicalNot"};
  return involution_ops.count(node.op()) > 0;
}

bool IsValuePreserving(const NodeDef& node) {
  if (NumNonControlInputs(node) == 1 && IsAggregate(node)) {
    return true;
  }
  const std::unordered_set<string> value_preserving_ops{
      "Transpose",  "Reshape",      "Identity",        "InvertPermutation",
      "Reverse",    "StopGradient", "PreventGradient", "CheckNumerics",
      "ExpandDims", "Squeeze"};
  return value_preserving_ops.count(node.op()) > 0;
}

}  // namespace grappler
}  // end namespace tensorflow
