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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

bool IsAdd(const NodeDef& node) {
  if (node.op() == "AddV2") {
    return true;
  }
  if (node.op() == "Add") {
    DataType type = node.attr().at("T").type();
    return type != DT_STRING;
  }
  return false;
}

bool IsAddN(const NodeDef& node) { return node.op() == "AddN"; }

bool IsAll(const NodeDef& node) { return node.op() == "All"; }

bool IsAngle(const NodeDef& node) { return node.op() == "Angle"; }

bool IsAny(const NodeDef& node) { return node.op() == "Any"; }

bool IsAnyDiv(const NodeDef& node) {
  return node.op() == "RealDiv" || node.op() == "Div" || node.op() == "Xdivy" ||
         node.op() == "FloorDiv" || node.op() == "TruncateDiv";
}

bool IsAnyBatchMatMul(const NodeDef& node) {
  return node.op() == "BatchMatMul" || node.op() == "BatchMatMulV2";
}

bool IsAnyMatMul(const NodeDef& node) {
  return node.op() == "MatMul" || node.op() == "SparseMatMul" ||
         IsAnyBatchMatMul(node) || IsQuantizedMatMul(node);
}

bool IsAnyMax(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Max" || op == "SegmentMax" || op == "UnsortedSegmentMax";
}

bool IsAnyMaxPool(const NodeDef& node) {
  const auto& op = node.op();
  return op == "MaxPool" || op == "MaxPoolV2" || op == "MaxPool3D" ||
         op == "MaxPoolWithArgmax" || op == "FractionalMaxPool";
}

bool IsAnyMin(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Min" || op == "SegmentMin" || op == "UnsortedSegmentMin";
}

bool IsAnySparseSegmentReduction(const NodeDef& node) {
  const auto& op = node.op();
  return op == "SparseSegmentSum" || op == "SparseSegmentSumWithNumSegments" ||
         op == "SparseSegmentMean" ||
         op == "SparseSegmentMeanWithNumSegments" ||
         op == "SparseSegmentSqrtN" ||
         op == "SparseSegmentSqrtNWithNumSegments";
}

bool IsApproximateEqual(const NodeDef& node) {
  return node.op() == "ApproximateEqual";
}

bool IsArg(const NodeDef& node) {
  return node.op() == "_Arg" || node.op() == "_DeviceArg";
}

bool IsArgMax(const NodeDef& node) { return node.op() == "ArgMax"; }

bool IsArgMin(const NodeDef& node) { return node.op() == "ArgMin"; }

bool IsAvgPoolGrad(const NodeDef& node) { return node.op() == "AvgPoolGrad"; }

bool IsAssign(const NodeDef& node) {
  return node.op() == "Assign" || node.op() == "AssignVariableOp";
}

bool IsAssert(const NodeDef& node) { return node.op() == "Assert"; }

bool IsAsString(const NodeDef& node) { return node.op() == "AsString"; }

bool IsAtan2(const NodeDef& node) { return node.op() == "Atan2"; }

bool IsBetainc(const NodeDef& node) { return node.op() == "Betainc"; }

bool IsBiasAdd(const NodeDef& node) {
  return node.op() == "BiasAdd" || node.op() == "BiasAddV1";
}

bool IsBiasAddV2(const NodeDef& node) { return node.op() == "BiasAdd"; }

bool IsBiasAddGrad(const NodeDef& node) { return node.op() == "BiasAddGrad"; }

bool IsBitcast(const NodeDef& node) { return node.op() == "Bitcast"; }

bool IsBroadcastTo(const NodeDef& node) { return node.op() == "BroadcastTo"; }

bool IsCast(const NodeDef& node) { return node.op() == "Cast"; }

bool IsCastLike(const NodeDef& node) {
  static const gtl::FlatSet<string>* const kCastLikeOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Angle", "Bucketize", "Cast", "Dequantize", "HistogramFixedWidth",
          "Imag", "IsFinite", "IsInf", "IsNan", "Quantize",
          "QuantizeDownAndShrinkRange", "QuantizeV2", "QuantizedInstanceNorm",
          "QuantizedRelu", "QuantizedRelu6", "QuantizedReluX", "Real",
          "Requantize"}));
  return kCastLikeOps->count(node.op()) > 0;
}

bool IsCheckNumerics(const NodeDef& node) {
  return node.op() == "CheckNumerics";
}

bool IsCollective(const NodeDef& node) {
  return node.op() == "CollectiveReduce" ||
         node.op() == "CollectiveBcastSend" ||
         node.op() == "CollectiveBcastRecv";
}

bool IsComplex(const NodeDef& node) { return node.op() == "Complex"; }

bool IsComplexAbs(const NodeDef& node) { return node.op() == "ComplexAbs"; }

bool IsConcat(const NodeDef& node) {
  return node.op() == "Concat" || node.op() == "ConcatV2";
}

bool IsConcatOffset(const NodeDef& node) { return node.op() == "ConcatOffset"; }

bool IsConstant(const NodeDef& node) { return node.op() == "Const"; }

bool IsConj(const NodeDef& node) { return node.op() == "Conj"; }

bool IsConjugateTranspose(const NodeDef& node) {
  return node.op() == "ConjugateTranspose";
}

bool IsControlFlow(const NodeDef& node) {
  // clang-format off
  return node.op() == "ControlTrigger" ||
         node.op() == "Enter" ||
         node.op() == "Exit" ||
         node.op() == "LoopCond" ||
         node.op() == "Merge" ||
         node.op() == "_XlaMerge" ||
         node.op() == "NextIteration" ||
         node.op() == "Switch" ||
         node.op() == "_SwitchN";
  // clang-format on
}

bool IsConv2D(const NodeDef& node) { return node.op() == "Conv2D"; }

bool IsConv2DBackpropFilter(const NodeDef& node) {
  return node.op() == "Conv2DBackpropFilter";
}

bool IsConv2DBackpropInput(const NodeDef& node) {
  return node.op() == "Conv2DBackpropInput";
}

bool IsConv3D(const NodeDef& node) { return node.op() == "Conv3D"; }

bool IsConv3DBackpropFilterV2(const NodeDef& node) {
  return node.op() == "Conv3DBackpropFilterV2";
}

bool IsConv3DBackpropInputV2(const NodeDef& node) {
  return node.op() == "Conv3DBackpropInputV2";
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

bool IsDiv(const NodeDef& node) { return node.op() == "Div"; }

bool IsDivNoNan(const NodeDef& node) { return node.op() == "DivNoNan"; }

// Returns true if node represents a unary elementwise function that is
// monotonic. If *is_non_decreasing is true, the function is non-decreasing,
// e.g. sqrt, exp. *is_non_decreasing is false, the function is non-increasing,
// e.g. inv.
bool IsElementWiseMonotonic(const NodeDef& node, bool* is_non_decreasing) {
  static const gtl::FlatSet<string>* const kMonotonicNonDecreasingOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Acosh", "Asin", "Asinh",    "Atan",     "Atanh", "Ceil",
          "Elu",   "Erf",  "Exp",      "Expm1",    "Floor", "Log",
          "Log1p", "Relu", "Relu6",    "Rint",     "Selu",  "Sigmoid",
          "Sign",  "Sinh", "Softsign", "Softplus", "Sqrt",  "Tanh",
      }));
  static const gtl::FlatSet<string>* const kMonotonicNonIncreasingOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{"Acos", "Erfc", "Neg", "Rsqrt"}));
  if (kMonotonicNonDecreasingOps->count(node.op()) > 0) {
    if (is_non_decreasing) {
      *is_non_decreasing = true;
    }
    return true;
  } else if (kMonotonicNonIncreasingOps->count(node.op()) > 0) {
    if (is_non_decreasing) {
      *is_non_decreasing = false;
    }
    return true;
  }
  return false;
}

bool IsElu(const NodeDef& node) { return node.op() == "Elu"; }

bool IsEluGrad(const NodeDef& node) { return node.op() == "EluGrad"; }

bool IsQuantizationEmulation(const NodeDef& node) {
  const auto& op = node.op();
  return absl::StartsWith(op, "QuantizeAndDequantize") ||
         absl::StartsWith(op, "FakeQuantWithMinMax");
}

bool IsEnter(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Enter" || op == "RefEnter";
}

bool IsEqual(const NodeDef& node) { return node.op() == "Equal"; }

bool IsExit(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Exit" || op == "RefExit";
}

bool IsExp(const NodeDef& node) { return node.op() == "Exp"; }

bool IsFakeParam(const NodeDef& node) { return node.op() == "FakeParam"; }

bool IsFill(const NodeDef& node) { return node.op() == "Fill"; }

bool IsFloorDiv(const NodeDef& node) { return node.op() == "FloorDiv"; }

bool IsFloorMod(const NodeDef& node) { return node.op() == "FloorMod"; }

bool IsFusedBatchNorm(const NodeDef& node) {
  const auto& op = node.op();
  return op == "FusedBatchNorm" || op == "FusedBatchNormV2" ||
         op == "FusedBatchNormV3";
}

bool IsFusedBatchNormEx(const NodeDef& node) {
  return node.op() == "_FusedBatchNormEx";
}

bool IsFusedBatchNormGrad(const NodeDef& node) {
  const auto& op = node.op();
  return op == "FusedBatchNormGrad" || op == "FusedBatchNormGradV2" ||
         op == "FusedBatchNormGradV3";
}

bool IsGather(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Gather" || op == "GatherV2" || op == "ResourceGather";
}

bool IsGreater(const NodeDef& node) { return node.op() == "Greater"; }

bool IsGreaterEqual(const NodeDef& node) { return node.op() == "GreaterEqual"; }

bool IsHostConstant(const NodeDef& node) { return node.op() == "HostConst"; }

bool IsHistogramSummary(const NodeDef& node) {
  return node.op() == "HistogramSummary";
}

bool IsIdentity(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Identity" || op == "RefIdentity";
}

bool IsIdentityN(const NodeDef& node) {
  const auto& op = node.op();
  return op == "IdentityN";
}

bool IsIdentityNSingleInput(const NodeDef& node) {
  return IsIdentityN(node) && node.attr().count("T") != 0 &&
         node.attr().at("T").list().type_size() == 1;
}

bool IsIf(const NodeDef& node) {
  const auto& op = node.op();
  return op == "If" || op == "StatelessIf";
}

bool IsIgamma(const NodeDef& node) { return node.op() == "Igamma"; }

bool IsIgammac(const NodeDef& node) { return node.op() == "Igammac"; }

bool IsImag(const NodeDef& node) { return node.op() == "Imag"; }

bool IsImmutableConst(const NodeDef& node) {
  return node.op() == "ImmutableConst";
}

bool IsInvGrad(const NodeDef& node) { return node.op() == "InvGrad"; }

bool IsLeakyRelu(const NodeDef& node) { return node.op() == "LeakyRelu"; }

bool IsLeakyReluGrad(const NodeDef& node) {
  return node.op() == "LeakyReluGrad";
}

bool IsLess(const NodeDef& node) { return node.op() == "Less"; }

bool IsLessEqual(const NodeDef& node) { return node.op() == "LessEqual"; }

bool IsLog(const NodeDef& node) { return node.op() == "Log"; }

bool IsLogicalAnd(const NodeDef& node) { return node.op() == "LogicalAnd"; }

bool IsLogicalNot(const NodeDef& node) { return node.op() == "LogicalNot"; }

bool IsLogicalOr(const NodeDef& node) { return node.op() == "LogicalOr"; }

bool IsLoopCond(const NodeDef& node) { return node.op() == "LoopCond"; }

bool IsMatMul(const NodeDef& node) { return node.op() == "MatMul"; }

bool IsMax(const NodeDef& node) { return node.op() == "Max"; }

bool IsMaximum(const NodeDef& node) { return node.op() == "Maximum"; }

bool IsMaxPoolGrad(const NodeDef& node) { return node.op() == "MaxPoolGrad"; }

bool IsMean(const NodeDef& node) { return node.op() == "Mean"; }

bool IsMerge(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Merge" || op == "RefMerge" || op == "_XlaMerge";
}

bool IsMin(const NodeDef& node) { return node.op() == "Min"; }

bool IsMinimum(const NodeDef& node) { return node.op() == "Minimum"; }

bool IsMirrorPad(const NodeDef& node) { return node.op() == "MirrorPad"; }

bool IsMirrorPadGrad(const NodeDef& node) {
  return node.op() == "MirrorPadGrad";
}

bool IsMklFusedMish(const NodeDef& node) {
  return node.op() == "_MklFusedMish";
}

bool IsMod(const NodeDef& node) { return node.op() == "Mod"; }

bool IsMul(const NodeDef& node) { return node.op() == "Mul"; }
bool IsMulNoNan(const NodeDef& node) { return node.op() == "MulNoNan"; }
bool IsAnyMul(const NodeDef& node) { return IsMul(node) || IsMulNoNan(node); }

bool IsNeg(const NodeDef& node) { return node.op() == "Neg"; }

bool IsNoOp(const NodeDef& node) { return node.op() == "NoOp"; }

bool IsNotEqual(const NodeDef& node) { return node.op() == "NotEqual"; }

bool IsNextIteration(const NodeDef& node) {
  const auto& op = node.op();
  return op == "NextIteration" || op == "RefNextIteration";
}

bool IsOnesLike(const NodeDef& node) { return node.op() == "OnesLike"; }

bool IsPack(const NodeDef& node) { return node.op() == "Pack"; }

bool IsPad(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Pad" || op == "PadV2";
}

bool IsPartitionedCall(const NodeDef& node) {
  return node.op() == "PartitionedCall";
}

bool IsPlaceholder(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Placeholder" || op == "PlaceholderV2" ||
         op == "PlaceholderWithDefault";
}

bool IsPolygamma(const NodeDef& node) { return node.op() == "Polygamma"; }

bool IsPow(const NodeDef& node) { return node.op() == "Pow"; }

bool IsPrint(const NodeDef& node) {
  return node.op() == "Print" || node.op() == "PrintV2";
}

bool IsProd(const NodeDef& node) { return node.op() == "Prod"; }

bool IsQuantizedMatMul(const NodeDef& node) {
  return node.op() == "QuantizedMatMul" || node.op() == "QuantizedMatMulV2";
}

bool IsQueue(const NodeDef& node) {
  return str_util::EndsWith(node.op(), "QueueV2");
}

bool IsRandomShuffle(const NodeDef& node) {
  return node.op() == "RandomShuffle";
}

bool IsRank(const NodeDef& node) { return node.op() == "Rank"; }

bool IsReadVariableOp(const NodeDef& node) {
  return node.op() == "ReadVariableOp";
}

bool IsReadVariablesOp(const NodeDef& node) {
  return node.op() == "_ReadVariablesOp";
}

bool IsReal(const NodeDef& node) { return node.op() == "Real"; }

bool IsRealDiv(const NodeDef& node) { return node.op() == "RealDiv"; }

bool IsReciprocalGrad(const NodeDef& node) {
  return node.op() == "ReciprocalGrad";
}

bool IsRecv(const NodeDef& node) {
  return node.op() == "_Recv" || node.op() == "_HostRecv";
}

bool IsReduction(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Sum" || op == "Prod" || op == "Min" || op == "Max" ||
         op == "Mean" || op == "Any" || op == "All";
}

bool IsRelu(const NodeDef& node) { return node.op() == "Relu"; }

bool IsRelu6(const NodeDef& node) { return node.op() == "Relu6"; }

bool IsReluGrad(const NodeDef& node) { return node.op() == "ReluGrad"; }

bool IsRelu6Grad(const NodeDef& node) { return node.op() == "Relu6Grad"; }

bool IsReshape(const NodeDef& node) { return (node.op() == "Reshape"); }

bool IsRestore(const NodeDef& node) {
  return (node.op() == "Restore" || node.op() == "RestoreV2" ||
          node.op() == "RestoreSlice");
}

bool IsRetval(const NodeDef& node) {
  return node.op() == "_Retval" || node.op() == "_DeviceRetval";
}

bool IsReverse(const NodeDef& node) {
  return node.op() == "Reverse" || node.op() == "ReverseV2";
}

bool IsReverseV2(const NodeDef& node) { return node.op() == "ReverseV2"; }

bool IsRsqrt(const NodeDef& node) { return node.op() == "Rsqrt"; }

bool IsRsqrtGrad(const NodeDef& node) { return node.op() == "RsqrtGrad"; }

bool IsSelect(const NodeDef& node) {
  return node.op() == "Select" || node.op() == "SelectV2";
}

bool IsSeluGrad(const NodeDef& node) { return node.op() == "SeluGrad"; }

bool IsSend(const NodeDef& node) {
  return node.op() == "_Send" || node.op() == "_HostSend";
}

bool IsShape(const NodeDef& node) { return node.op() == "Shape"; }

bool IsShapeN(const NodeDef& node) { return node.op() == "ShapeN"; }

bool IsShuffle(const NodeDef& node) { return node.op() == "Shuffle"; }

bool IsSigmoid(const NodeDef& node) { return node.op() == "Sigmoid"; }

bool IsSigmoidGrad(const NodeDef& node) { return node.op() == "SigmoidGrad"; }

bool IsSize(const NodeDef& node) { return node.op() == "Size"; }

bool IsSlice(const NodeDef& node) { return node.op() == "Slice"; }

bool IsSnapshot(const NodeDef& node) { return node.op() == "Snapshot"; }

bool IsSoftmax(const NodeDef& node) { return node.op() == "Softmax"; }

bool IsSoftplusGrad(const NodeDef& node) { return node.op() == "SoftplusGrad"; }

bool IsSoftsignGrad(const NodeDef& node) { return node.op() == "SoftsignGrad"; }

bool IsSplit(const NodeDef& node) { return node.op() == "Split"; }

bool IsSplitV(const NodeDef& node) { return node.op() == "SplitV"; }

bool IsSqrt(const NodeDef& node) { return node.op() == "Sqrt"; }

bool IsSqrtGrad(const NodeDef& node) { return node.op() == "SqrtGrad"; }

bool IsSquare(const NodeDef& node) { return node.op() == "Square"; }

bool IsSquaredDifference(const NodeDef& node) {
  return node.op() == "SquaredDifference";
}

bool IsSqueeze(const NodeDef& node) { return node.op() == "Squeeze"; }

bool IsStackOp(const NodeDef& node) {
  return node.op() == "Stack" || node.op() == "StackV2";
}
bool IsStackCloseOp(const NodeDef& node) {
  return node.op() == "StackClose" || node.op() == "StackCloseV2";
}
bool IsStackPushOp(const NodeDef& node) {
  return node.op() == "StackPush" || node.op() == "StackPushV2";
}
bool IsStackPopOp(const NodeDef& node) {
  return node.op() == "StackPop" || node.op() == "StackPopV2";
}

bool IsStatefulPartitionedCall(const NodeDef& node) {
  return node.op() == "StatefulPartitionedCall";
}

bool IsStopGradient(const NodeDef& node) {
  const auto& op = node.op();
  return op == "StopGradient" || op == "PreventGradient";
}

bool IsStridedSlice(const NodeDef& node) { return node.op() == "StridedSlice"; }

bool IsStridedSliceGrad(const NodeDef& node) {
  return node.op() == "StridedSliceGrad";
}

bool IsStringToHashBucketFast(const NodeDef& node) {
  return node.op() == "StringToHashBucketFast";
}

bool IsSub(const NodeDef& node) { return node.op() == "Sub"; }

bool IsSum(const NodeDef& node) { return node.op() == "Sum"; }

bool IsSwitch(const NodeDef& node) {
  const auto& op = node.op();
  return op == "_SwitchN" || op == "Switch" || op == "RefSwitch";
}

bool IsSymbolicGradient(const NodeDef& node) {
  return node.op() == "SymbolicGradient";
}

bool IsTanh(const NodeDef& node) { return node.op() == "Tanh"; }

bool IsTanhGrad(const NodeDef& node) { return node.op() == "TanhGrad"; }

bool IsTensorArray(const NodeDef& node) {
  static const gtl::FlatSet<string>* const kTensorArrayOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "TensorArray",
          "TensorArrayV2",
          "TensorArrayV3",
          "TensorArrayGrad",
          "TensorArrayGradV2",
          "TensorArrayGradV3",
          "TensorArrayGradWithShape",
          "TensorArrayWrite",
          "TensorArrayWriteV2",
          "TensorArrayWriteV3",
          "TensorArrayRead",
          "TensorArrayReadV2",
          "TensorArrayReadV3",
          "TensorArrayConcat",
          "TensorArrayConcatV2",
          "TensorArrayConcatV3",
          "TensorArraySplit",
          "TensorArraySplitV2",
          "TensorArraySplitV3",
          "TensorArraySize",
          "TensorArraySizeV2",
          "TensorArraySizeV3",
          "TensorArrayClose",
          "TensorArrayCloseV2",
          "TensorArrayCloseV3",
      }));
  return kTensorArrayOps->count(node.op()) > 0;
}

bool IsTile(const NodeDef& node) { return node.op() == "Tile"; }

bool IsTranspose(const NodeDef& node) { return node.op() == "Transpose"; }

bool IsTruncateDiv(const NodeDef& node) { return node.op() == "TruncateDiv"; }

bool IsTruncateMod(const NodeDef& node) { return node.op() == "TruncateMod"; }

bool IsUnique(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Unique" || op == "UniqueV2";
}

bool IsUnpack(const NodeDef& node) { return node.op() == "Unpack"; }

bool IsVariable(const NodeDef& node) {
  const auto& op = node.op();
  return op == "Variable" || op == "VariableV2" || op == "AutoReloadVariable" ||
         op == "VarHandleOp" || op == "ReadVariableOp" ||
         op == "_VarHandlesOp" || op == "_ReadVariablesOp";
}

bool IsWhile(const NodeDef& node) {
  const auto& op = node.op();
  return op == "While" || op == "StatelessWhile";
}

bool IsXdivy(const NodeDef& node) { return node.op() == "Xdivy"; }

bool IsZerosLike(const NodeDef& node) { return node.op() == "ZerosLike"; }

bool IsZeta(const NodeDef& node) { return node.op() == "Zeta"; }

namespace {
bool GetBoolAttr(const NodeDef& node, const string& name) {
  return node.attr().count(name) > 0 && node.attr().at(name).b();
}
}  // namespace

bool IsPersistent(const NodeDef& node) {
  return IsConstant(node) || IsVariable(node) || IsHostConstant(node);
}

bool HasRefInput(const NodeDef& node) {
  const OpDef* op_def;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  // Nodes such as Assign or AssignAdd modify one of their inputs.
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return true;
    }
  }
  return false;
}

bool IsDataset(const NodeDef& node) {
  const string& op = node.op();
  // See `GetNodeClassForOp` in core/graph/graph.cc.
  return op == "IteratorGetNext" || op == "IteratorGetNextSync" ||
         op == "DatasetToSingleElement" || op == "ReduceDataset";
}

bool IsStateful(const NodeDef node, const OpRegistryInterface* op_registry) {
  const OpDef* op_def = nullptr;
  const string& op_name = node.op();
  Status status = op_registry->LookUpOpDef(op_name, &op_def);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to lookup OpDef for " << op_name
                 << ". Error: " << status.message();
    return false;
  }
  return op_def->is_stateful();
}

bool IsStateful(const NodeDef node) {
  return IsStateful(node, OpRegistry::Global());
}

bool IsFreeOfSideEffect(const NodeDef& node,
                        const OpRegistryInterface* op_registry) {
  // Placeholders must be preserved to keep the graph feedable.
  if (IsPlaceholder(node)) {
    return false;
  }
  const OpDef* op_def = nullptr;
  const string& op_name = node.op();
  Status status = op_registry->LookUpOpDef(op_name, &op_def);
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
  // Queue ops modify the queue which is a side effect.
  if (node.op().find("Queue") != string::npos) {
    return false;
  }
  // Sending a tensor via a network is a side effect.
  if (IsSend(node)) {
    return false;
  }
  return !ModifiesInputsInPlace(node);
}

bool IsFreeOfSideEffect(const NodeDef& node) {
  return IsFreeOfSideEffect(node, OpRegistry::Global());
}

bool ModifiesInputsInPlace(const NodeDef& node) {
  // Some nodes do in-place updates on regular tensor inputs.
  const string& op_name = node.op();

  // Ops that modify resource variables effectively modify one of their inputs.
  if (op_name == "AssignVariableOp" || op_name == "AssignAddVariableOp" ||
      op_name == "AssignSubVariableOp" || op_name == "ResourceScatterUpdate" ||
      op_name == "ResourceScatterAdd" || op_name == "ResourceScatterSub" ||
      op_name == "ResourceScatterMul" || op_name == "ResourceScatterDiv" ||
      op_name == "ResourceScatterMin" || op_name == "ResourceScatterMax") {
    return false;
  }

  string lower_op_name = op_name;
  std::transform(lower_op_name.begin(), lower_op_name.end(),
                 lower_op_name.begin(), ::tolower);
  if (absl::StrContains(lower_op_name, "inplace")) {
    return true;
  }
  return GetBoolAttr(node, "in_place") || GetBoolAttr(node, "inplace");
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
  static const gtl::FlatSet<string>* const kInvolutionOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{"Conj", "Reciprocal", "Invert",
                                              "Neg", "LogicalNot"}));
  return kInvolutionOps->count(node.op()) > 0;
}

bool IsValueAndOrderAndShapePreserving(const NodeDef& node) {
  if (NumNonControlInputs(node) == 1 && IsAggregate(node)) {
    return true;
  }
  static const gtl::FlatSet<string>* const kValueAndOrderAndShapePreservingOps =
      CHECK_NOTNULL((new const gtl::FlatSet<string>{
          "CheckNumerics",
          "DebugGradientIdentity",
          "DeepCopy",
          "Enter",
          "Exit",
          "PreventGradient",
          "Print",
          "Snapshot",
          "StopGradient",
      }));
  return kValueAndOrderAndShapePreservingOps->count(node.op()) > 0 ||
         IsIdentity(node);
}

bool IsValueAndOrderPreserving(const NodeDef& node) {
  if (NumNonControlInputs(node) == 1 && IsAggregate(node)) {
    return true;
  }
  static const gtl::FlatSet<string>* const kValueAndOrderPreservingOps =
      CHECK_NOTNULL((new const gtl::FlatSet<string>{
          "ExpandDims",
          "Reshape",
          "Squeeze",
      }));
  return kValueAndOrderPreservingOps->count(node.op()) > 0 ||
         IsValueAndOrderAndShapePreserving(node);
}

bool IsValuePreserving(const NodeDef& node) {
  static const gtl::FlatSet<string>* const kValuePreservingOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "InvertPermutation",
          "Reverse",
          "ReverseV2",
          "Roll",
          "Transpose",
          "DepthToSpace",
          "SpaceToDepth",
          "BatchToSpace",
          "BatchToSpaceND",
          "SpaceToBatch",
          "SpaceToBatchND",
      }));
  return IsValueAndOrderPreserving(node) ||
         kValuePreservingOps->count(node.op()) > 0;
}

bool IsUnaryElementWise(const NodeDef& node) {
  static const gtl::FlatSet<string>* const kElementWiseOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Abs",      "Acos",     "Acosh",      "Asin",       "Asinh",
          "Atan",     "Atanh",    "Ceil",       "ComplexAbs", "Conj",
          "Cos",      "Cosh",     "Digamma",    "Elu",        "Erf",
          "Erfc",     "Exp",      "Expm1",      "Floor",      "Inv",
          "Invert",   "Isinf",    "Isnan",      "Isfinite",   "Lgamma",
          "Log",      "Log1p",    "LogicalNot", "Neg",        "Reciprocal",
          "Relu",     "Relu6",    "Rint",       "Round",      "Selu",
          "Rsqrt",    "Sigmoid",  "Sign",       "Sin",        "SinH",
          "Softplus", "Softsign", "Sqrt",       "Square",     "Tan",
          "Tanh",
      }));
  return kElementWiseOps->count(node.op()) > 0 ||
         IsValueAndOrderAndShapePreserving(node);
}

bool HasOpDef(const NodeDef& node) {
  const OpDef* op_def = nullptr;
  return OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok();
}

bool IsIdempotent(const NodeDef& node) {
  return IsValueAndOrderAndShapePreserving(node) && IsFreeOfSideEffect(node) &&
         !ModifiesFrameInfo(node);
}

bool NeverForwardsInputs(const NodeDef& node) {
  static const gtl::FlatSet<string>* const kNonForwardingOps = CHECK_NOTNULL(
      (new gtl::FlatSet<string>{"ArgMax",
                                "ArgMin",
                                "AudioSpectrogram",
                                "AvgPool",
                                "BatchMatMul",
                                "BatchMatMulV2",
                                "BatchNormWithGlobalNormalization",
                                "BatchToSpace",
                                "BatchToSpaceND",
                                "Bincount",
                                "BroadcastArgs",
                                "BroadcastGradientArgs",
                                "Bucketize",
                                "CTCBeamSearchDecoder",
                                "CTCGreedyDecoder",
                                "CTCLoss",
                                "CompareAndBitpack",
                                "ComplexAbs",
                                "Concat",
                                "ConcatOffset",
                                "ConcatV2",
                                "Conv2D",
                                "Copy",
                                "CopyHost",
                                "Cross",
                                "CudnnRNN",
                                "CudnnRNNBackprop",
                                "CudnnRNNBackpropV2",
                                "CudnnRNNBackpropV3",
                                "CudnnRNNCanonicalToParams",
                                "CudnnRNNCanonicalToParamsV2",
                                "CudnnRNNParamsSize",
                                "CudnnRNNParamsToCanonical",
                                "CudnnRNNParamsToCanonicalV2",
                                "CudnnRNNV2",
                                "CudnnRNNV3",
                                "CumProd",
                                "CumSum",
                                "DebugNanCount",
                                "DebugNumericSummary",
                                "DecodeProtoV2",
                                "DecodeWav",
                                "DeepCopy",
                                "DepthToSpace",
                                "Dequantize",
                                "Diag",
                                "DiagPart",
                                "EditDistance",
                                "Empty",
                                "EncodeProtoV2",
                                "EncodeWav",
                                "ExtractImagePatches",
                                "ExtractVolumePatches",
                                "Fill",
                                "Gather",
                                "GatherNd",
                                "GatherV2",
                                "HistogramFixedWidth",
                                "InvertPermutation",
                                "IsInf",
                                "IsNan",
                                "Isfinite",
                                "LinSpace",
                                "LowerBound",
                                "MatMul",
                                "MatrixDiag",
                                "MatrixDiagPart",
                                "MatrixDiagPartV2",
                                "MatrixDiagV2",
                                "Mfcc",
                                "Multinomial",
                                "OneHot",
                                "Pack",
                                "ParameterizedTruncatedNormal",
                                "PopulationCount",
                                "RandomGamma",
                                "RandomPoisson",
                                "RandomPoissonV2",
                                "RandomStandardNormal",
                                "RandomUniform",
                                "RandomUniformInt",
                                "Range",
                                "Rank",
                                "RequantizationRange",
                                "Requantize",
                                "ReverseSequence",
                                "Shape",
                                "ShapeN",
                                "Size",
                                "SpaceToBatch",
                                "SpaceToBatchND",
                                "SpaceToDepth",
                                "SparseMatMul",
                                "Split",
                                "SplitV",
                                "TruncatedNormal",
                                "Unique",
                                "UniqueV2",
                                "UniqueWithCounts",
                                "UniqueWithCountsV2",
                                "Unpack",
                                "UnravelIndex",
                                "UpperBound",
                                "Where"}));
  const string& op_name = node.op();
  return kNonForwardingOps->count(op_name) > 0 ||
         absl::StrContains(op_name, "Segment") ||
         absl::StartsWith(op_name, "Quantize");
}

bool IsXlaLaunch(const NodeDef& node) { return node.op() == "XlaLaunch"; }

}  // namespace grappler
}  // end namespace tensorflow
