/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TRANSFORMS_UTILS_OP_CAT_HELPER_H_
#define TENSORFLOW_CORE_TRANSFORMS_UTILS_OP_CAT_HELPER_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/core/ir/tf_op_wrapper.h"

namespace mlir {
namespace tfg {

// A Helper class to identify if an op belongs to certain op category.
class OpCatHelper {
 public:
  explicit OpCatHelper(MLIRContext *context) : context_(context) {}

  bool IsAdd(TFOp op);
  bool IsAddN(TFOp op);
  bool IsAll(TFOp op);
  bool IsAngle(TFOp op);
  bool IsAny(TFOp op);
  bool IsAnyDiv(TFOp op);
  bool IsAnyBatchMatMul(TFOp op);
  bool IsAnyMatMul(TFOp op);
  bool IsAnyMax(TFOp op);
  bool IsAnyMaxPool(TFOp op);
  bool IsAnyMin(TFOp op);
  bool IsAnyMul(TFOp op);
  bool IsAnySparseSegmentReduction(TFOp op);
  bool IsApproximateEqual(TFOp op);
  bool IsArg(TFOp op);
  bool IsArgMax(TFOp op);
  bool IsArgMin(TFOp op);
  bool IsAssert(TFOp op);
  bool IsAssign(TFOp op);
  bool IsAsString(TFOp op);
  bool IsAtan2(TFOp op);
  bool IsAvgPoolGrad(TFOp op);
  bool IsBetainc(TFOp op);
  bool IsBiasAdd(TFOp op);
  bool IsBiasAddV2(TFOp op);
  bool IsBiasAddGrad(TFOp op);
  bool IsBitcast(TFOp op);
  bool IsBroadcastTo(TFOp op);
  bool IsCast(TFOp op);
  bool IsCheckNumerics(TFOp op);
  bool IsCollective(TFOp op);
  bool IsComplex(TFOp op);
  bool IsComplexAbs(TFOp op);
  bool IsConcat(TFOp op);
  bool IsConcatV2(TFOp op);
  bool IsConcatOffset(TFOp op);
  bool IsConj(TFOp op);
  bool IsConjugateTranspose(TFOp op);
  bool IsConstant(TFOp op);
  bool IsControlFlow(TFOp op);
  bool IsConv2D(TFOp op);
  bool IsConv2DBackpropFilter(TFOp op);
  bool IsConv2DBackpropInput(TFOp op);
  bool IsConv3D(TFOp op);
  bool IsConv3DBackpropFilterV2(TFOp op);
  bool IsConv3DBackpropInputV2(TFOp op);
  bool IsDepthwiseConv2dNative(TFOp op);
  bool IsDepthwiseConv2dNativeBackpropFilter(TFOp op);
  bool IsDepthwiseConv2dNativeBackpropInput(TFOp op);
  bool IsDequeueOp(TFOp op);
  bool IsDiv(TFOp op);
  bool IsDivNoNan(TFOp op);
  bool IsElementWiseMonotonic(TFOp op, bool *is_non_decreasing);
  bool IsElu(TFOp op);
  bool IsEluGrad(TFOp op);
  bool IsQuantizationEmulation(TFOp op);
  bool IsEnter(TFOp op);
  bool IsEqual(TFOp op);
  bool IsExit(TFOp op);
  bool IsExp(TFOp op);
  bool IsFakeParam(TFOp op);
  bool IsFill(TFOp op);
  bool IsFloorDiv(TFOp op);
  bool IsFloorMod(TFOp op);
  bool IsFusedBatchNorm(TFOp op);
  bool IsFusedBatchNormEx(TFOp op);
  bool IsFusedBatchNormGrad(TFOp op);
  bool IsGather(TFOp op);
  bool IsGreater(TFOp op);
  bool IsGreaterEqual(TFOp op);
  bool IsHistogramSummary(TFOp op);
  bool IsHostConstant(TFOp op);
  bool IsIdentity(TFOp op);
  bool IsIdentityN(TFOp op);
  bool IsIdentityNSingleInput(TFOp op);
  bool IsIf(TFOp op);
  bool IsIgamma(TFOp op);
  bool IsIgammac(TFOp op);
  bool IsImag(TFOp op);
  bool IsImmutableConst(TFOp op);
  bool IsInvGrad(TFOp op);
  bool IsLeakyRelu(TFOp op);
  bool IsLeakyReluGrad(TFOp op);
  bool IsLess(TFOp op);
  bool IsLessEqual(TFOp op);
  bool IsLog(TFOp op);
  bool IsLogicalAnd(TFOp op);
  bool IsLogicalNot(TFOp op);
  bool IsLogicalOr(TFOp op);
  bool IsLoopCond(TFOp op);
  bool IsMatMul(TFOp op);
  bool IsMax(TFOp op);
  bool IsMaxPoolGrad(TFOp op);
  bool IsMaximum(TFOp op);
  bool IsMean(TFOp op);
  bool IsMerge(TFOp op);
  bool IsMin(TFOp op);
  bool IsMinimum(TFOp op);
  bool IsMirrorPad(TFOp op);
  bool IsMirrorPadGrad(TFOp op);
  bool IsMod(TFOp op);
  bool IsMul(TFOp op);
  bool IsMulNoNan(TFOp op);
  bool IsNeg(TFOp op);
  bool IsNextIteration(TFOp op);
  bool IsNoOp(TFOp op);
  bool IsNotEqual(TFOp op);
  bool IsOnesLike(TFOp op);
  bool IsPack(TFOp op);
  bool IsPad(TFOp op);
  bool IsPartitionedCall(TFOp op);
  bool IsPlaceholder(TFOp op);
  bool IsPolygamma(TFOp op);
  bool IsPow(TFOp op);
  bool IsPrint(TFOp op);
  bool IsProd(TFOp op);
  bool IsQuantizedMatMul(TFOp op);
  bool IsQueue(TFOp op);
  bool IsRandomShuffle(TFOp op);
  bool IsRank(TFOp op);
  bool IsReadVariableOp(TFOp op);
  bool IsReadVariablesOp(TFOp op);
  bool IsReal(TFOp op);
  bool IsRealDiv(TFOp op);
  bool IsReciprocalGrad(TFOp op);
  bool IsRecv(TFOp op);
  bool IsReduction(TFOp op);
  bool IsRelu(TFOp op);
  bool IsRelu6(TFOp op);
  bool IsRelu6Grad(TFOp op);
  bool IsReluGrad(TFOp op);
  bool IsReshape(TFOp op);
  bool IsRestore(TFOp op);
  bool IsRetval(TFOp op);
  bool IsReverse(TFOp op);
  bool IsReverseV2(TFOp op);
  bool IsRsqrt(TFOp op);
  bool IsRsqrtGrad(TFOp op);
  bool IsSelect(TFOp op);
  bool IsSeluGrad(TFOp op);
  bool IsSend(TFOp op);
  bool IsShape(TFOp op);
  bool IsShapeN(TFOp op);
  bool IsShuffle(TFOp op);
  bool IsSigmoid(TFOp op);
  bool IsSigmoidGrad(TFOp op);
  bool IsSize(TFOp op);
  bool IsSlice(TFOp op);
  bool IsSnapshot(TFOp op);
  bool IsSoftmax(TFOp op);
  bool IsSoftplusGrad(TFOp op);
  bool IsSoftsignGrad(TFOp op);
  bool IsSplit(TFOp op);
  bool IsSplitV(TFOp op);
  bool IsSqrt(TFOp op);
  bool IsSqrtGrad(TFOp op);
  bool IsSquare(TFOp op);
  bool IsSquaredDifference(TFOp op);
  bool IsSqueeze(TFOp op);
  bool IsStackCloseOp(TFOp op);
  bool IsStackOp(TFOp op);
  bool IsStackPopOp(TFOp op);
  bool IsStackPushOp(TFOp op);
  bool IsStatefulPartitionedCall(TFOp op);
  bool IsStopGradient(TFOp op);
  bool IsStridedSlice(TFOp op);
  bool IsStridedSliceGrad(TFOp op);
  bool IsStringToHashBucketFast(TFOp op);
  bool IsSub(TFOp op);
  bool IsSum(TFOp op);
  bool IsSwitch(TFOp op);
  bool IsSymbolicGradient(TFOp op);
  bool IsTanh(TFOp op);
  bool IsTanhGrad(TFOp op);
  bool IsTensorArray(TFOp op);
  bool IsTile(TFOp op);
  bool IsTranspose(TFOp op);
  bool IsTruncateDiv(TFOp op);
  bool IsTruncateMod(TFOp op);
  bool IsUnique(TFOp op);
  bool IsUnpack(TFOp op);
  bool IsVariable(TFOp op);
  bool IsWhile(TFOp op);
  bool IsXdivy(TFOp op);
  bool IsXlaLaunch(TFOp op);
  bool IsZerosLike(TFOp op);
  bool IsZeta(TFOp op);
  bool IsAggregate(TFOp op);
  bool IsCommutative(TFOp op);

  // Returns true if it's a splat tensor type and has the splat value 1.
  bool IsOnes(TFOp op);
  // Returns true if it's a splat tensor type and has the splat value 0.
  bool IsZeros(TFOp op);

  // Returns true if the op is known to use persistent memory to store its
  // value.
  bool IsPersistent(TFOp op);

  // Returns true if the op belongs to the NC_DATASET class (see graph/graph.h).
  bool IsDataset(TFOp op);

 private:
  MLIRContext *context_;
};

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_UTILS_OP_CAT_HELPER_H_
