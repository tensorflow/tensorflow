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

#ifndef TENSORFLOW_CORE_GRAPPLER_OP_TYPES_H_
#define TENSORFLOW_CORE_GRAPPLER_OP_TYPES_H_

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
bool IsAdd(const NodeDef& node);
bool IsAddN(const NodeDef& node);
bool IsAll(const NodeDef& node);
bool IsAngle(const NodeDef& node);
bool IsAny(const NodeDef& node);
bool IsAnyDiv(const NodeDef& node);
bool IsAnyBatchMatMul(const NodeDef& node);
bool IsAnyMatMul(const NodeDef& node);
bool IsAnyMax(const NodeDef& node);
bool IsAnyMaxPool(const NodeDef& node);
bool IsAnyMin(const NodeDef& node);
bool IsAnyMul(const NodeDef& node);
bool IsAnySparseSegmentReduction(const NodeDef& node);
bool IsApproximateEqual(const NodeDef& node);
bool IsArg(const NodeDef& node);
bool IsArgMax(const NodeDef& node);
bool IsArgMin(const NodeDef& node);
bool IsAssert(const NodeDef& node);
bool IsAssign(const NodeDef& node);
bool IsAsString(const NodeDef& node);
bool IsAtan2(const NodeDef& node);
bool IsAvgPoolGrad(const NodeDef& node);
bool IsBetainc(const NodeDef& node);
bool IsBiasAdd(const NodeDef& node);
bool IsBiasAddV2(const NodeDef& node);
bool IsBiasAddGrad(const NodeDef& node);
bool IsBitcast(const NodeDef& node);
bool IsBroadcastTo(const NodeDef& node);
bool IsCast(const NodeDef& node);
bool IsCheckNumerics(const NodeDef& node);
bool IsCollective(const NodeDef& node);
bool IsComplex(const NodeDef& node);
bool IsComplexAbs(const NodeDef& node);
bool IsConcat(const NodeDef& node);
bool IsConcatOffset(const NodeDef& node);
bool IsConj(const NodeDef& node);
bool IsConjugateTranspose(const NodeDef& node);
bool IsConstant(const NodeDef& node);
bool IsControlFlow(const NodeDef& node);
bool IsConv2D(const NodeDef& node);
bool IsConv2DBackpropFilter(const NodeDef& node);
bool IsConv2DBackpropInput(const NodeDef& node);
bool IsConv3D(const NodeDef& node);
bool IsConv3DBackpropFilterV2(const NodeDef& node);
bool IsConv3DBackpropInputV2(const NodeDef& node);
bool IsDepthwiseConv2dNative(const NodeDef& node);
bool IsDepthwiseConv2dNativeBackpropFilter(const NodeDef& node);
bool IsDepthwiseConv2dNativeBackpropInput(const NodeDef& node);
bool IsDequeueOp(const NodeDef& node);
bool IsDiv(const NodeDef& node);
bool IsDivNoNan(const NodeDef& node);
bool IsElementWiseMonotonic(const NodeDef& node, bool* is_non_decreasing);
bool IsElu(const NodeDef& node);
bool IsEluGrad(const NodeDef& node);
bool IsQuantizationEmulation(const NodeDef& node);
bool IsEnter(const NodeDef& node);
bool IsEqual(const NodeDef& node);
bool IsExit(const NodeDef& node);
bool IsExp(const NodeDef& node);
bool IsFakeParam(const NodeDef& node);
bool IsFill(const NodeDef& node);
bool IsFloorDiv(const NodeDef& node);
bool IsFloorMod(const NodeDef& node);
bool IsFusedBatchNorm(const NodeDef& node);
bool IsFusedBatchNormEx(const NodeDef& node);
bool IsFusedBatchNormGrad(const NodeDef& node);
bool IsGather(const NodeDef& node);
bool IsGreater(const NodeDef& node);
bool IsGreaterEqual(const NodeDef& node);
bool IsHistogramSummary(const NodeDef& node);
bool IsHostConstant(const NodeDef& node);
bool IsIdentity(const NodeDef& node);
bool IsIdentityN(const NodeDef& node);
bool IsIdentityNSingleInput(const NodeDef& node);
bool IsIf(const NodeDef& node);
bool IsIgamma(const NodeDef& node);
bool IsIgammac(const NodeDef& node);
bool IsImag(const NodeDef& node);
bool IsImmutableConst(const NodeDef& node);
bool IsInvGrad(const NodeDef& node);
bool IsLeakyRelu(const NodeDef& node);
bool IsLeakyReluGrad(const NodeDef& node);
bool IsLess(const NodeDef& node);
bool IsLessEqual(const NodeDef& node);
bool IsLog(const NodeDef& node);
bool IsLogicalAnd(const NodeDef& node);
bool IsLogicalNot(const NodeDef& node);
bool IsLogicalOr(const NodeDef& node);
bool IsLoopCond(const NodeDef& node);
bool IsMatMul(const NodeDef& node);
bool IsMax(const NodeDef& node);
bool IsMaxPoolGrad(const NodeDef& node);
bool IsMaximum(const NodeDef& node);
bool IsMean(const NodeDef& node);
bool IsMerge(const NodeDef& node);
bool IsMin(const NodeDef& node);
bool IsMinimum(const NodeDef& node);
bool IsMirrorPad(const NodeDef& node);
bool IsMirrorPadGrad(const NodeDef& node);
bool IsMklFusedMish(const NodeDef& node);
bool IsMod(const NodeDef& node);
bool IsMul(const NodeDef& node);
bool IsMulNoNan(const NodeDef& node);
bool IsNeg(const NodeDef& node);
bool IsNextIteration(const NodeDef& node);
bool IsNoOp(const NodeDef& node);
bool IsNotEqual(const NodeDef& node);
bool IsOnesLike(const NodeDef& node);
bool IsPack(const NodeDef& node);
bool IsPack(const NodeDef& node);
bool IsPad(const NodeDef& node);
bool IsPartitionedCall(const NodeDef& node);
bool IsPlaceholder(const NodeDef& node);
bool IsPolygamma(const NodeDef& node);
bool IsPow(const NodeDef& node);
bool IsPrint(const NodeDef& node);
bool IsProd(const NodeDef& node);
bool IsQuantizedMatMul(const NodeDef& node);
bool IsQueue(const NodeDef& node);
bool IsRandomShuffle(const NodeDef& node);
bool IsRank(const NodeDef& node);
bool IsReadVariableOp(const NodeDef& node);
bool IsReadVariablesOp(const NodeDef& node);
bool IsReal(const NodeDef& node);
bool IsRealDiv(const NodeDef& node);
bool IsReciprocalGrad(const NodeDef& node);
bool IsRecv(const NodeDef& node);
bool IsReduction(const NodeDef& node);
bool IsRelu(const NodeDef& node);
bool IsRelu6(const NodeDef& node);
bool IsRelu6Grad(const NodeDef& node);
bool IsReluGrad(const NodeDef& node);
bool IsReshape(const NodeDef& node);
bool IsRestore(const NodeDef& node);
bool IsRetval(const NodeDef& node);
bool IsReverse(const NodeDef& node);
bool IsReverseV2(const NodeDef& node);
bool IsRsqrt(const NodeDef& node);
bool IsRsqrtGrad(const NodeDef& node);
bool IsSelect(const NodeDef& node);
bool IsSeluGrad(const NodeDef& node);
bool IsSend(const NodeDef& node);
bool IsShape(const NodeDef& node);
bool IsShapeN(const NodeDef& node);
bool IsShuffle(const NodeDef& node);
bool IsSigmoid(const NodeDef& node);
bool IsSigmoidGrad(const NodeDef& node);
bool IsSize(const NodeDef& node);
bool IsSlice(const NodeDef& node);
bool IsSnapshot(const NodeDef& node);
bool IsSoftmax(const NodeDef& node);
bool IsSoftplusGrad(const NodeDef& node);
bool IsSoftsignGrad(const NodeDef& node);
bool IsSplit(const NodeDef& node);
bool IsSplitV(const NodeDef& node);
bool IsSqrt(const NodeDef& node);
bool IsSqrtGrad(const NodeDef& node);
bool IsSquare(const NodeDef& node);
bool IsSquaredDifference(const NodeDef& node);
bool IsSqueeze(const NodeDef& node);
bool IsStackCloseOp(const NodeDef& node);
bool IsStackOp(const NodeDef& node);
bool IsStackPopOp(const NodeDef& node);
bool IsStackPushOp(const NodeDef& node);
bool IsStatefulPartitionedCall(const NodeDef& node);
bool IsStopGradient(const NodeDef& node);
bool IsStridedSlice(const NodeDef& node);
bool IsStridedSliceGrad(const NodeDef& node);
bool IsStringToHashBucketFast(const NodeDef& node);
bool IsSub(const NodeDef& node);
bool IsSum(const NodeDef& node);
bool IsSwitch(const NodeDef& node);
bool IsSymbolicGradient(const NodeDef& node);
bool IsTanh(const NodeDef& node);
bool IsTanhGrad(const NodeDef& node);
bool IsTensorArray(const NodeDef& node);
bool IsTile(const NodeDef& node);
bool IsTranspose(const NodeDef& node);
bool IsTruncateDiv(const NodeDef& node);
bool IsTruncateMod(const NodeDef& node);
bool IsUnique(const NodeDef& node);
bool IsUnpack(const NodeDef& node);
bool IsVariable(const NodeDef& node);
bool IsWhile(const NodeDef& node);
bool IsXdivy(const NodeDef& node);
bool IsXlaLaunch(const NodeDef& node);
bool IsZerosLike(const NodeDef& node);
bool IsZeta(const NodeDef& node);

// Return true if the op is an aggregation (e.g. Add, AddN).
// Returns false if it could not be determined to be so.
bool IsAggregate(const NodeDef& node);

// Return true if the op is commutative (e.g. Mul, Add).
// Returns false if it could not be determined to be so.
bool IsCommutative(const NodeDef& node);

// Returns true if the node is known to use persistent memory to store its
// value.
bool IsPersistent(const NodeDef& node);

// Returns true if the node belongs to the NC_DATASET class (see graph/graph.h).
bool IsDataset(const NodeDef& node);

// Returns true if the node op is marked as stateful, or if it was not found in
// op_registry.
bool IsStateful(const NodeDef& node, const OpRegistryInterface* op_registry);
bool IsStateful(const NodeDef& node);  // use OpRegistry::Global()

bool IsFreeOfSideEffect(const NodeDef& node,
                        const OpRegistryInterface* op_registry);
bool IsFreeOfSideEffect(const NodeDef& node);  // use OpRegistry::Global()

// Returns true if the takes a tensor reference as input.
// Returns false if the op type is unknown.
bool HasRefInput(const NodeDef& node);

bool ModifiesFrameInfo(const NodeDef& node);

// Returns true if the op is known to write to one or more of its inputs.
bool ModifiesInputsInPlace(const NodeDef& node);

// Returns true if the op is an element-wise involution, i.e. if it is its
// own inverse such that f(f(x)) == x.
bool IsInvolution(const NodeDef& node);

// Returns true if the op preserves the order and value of elements
// and shape of its first input tensor.
bool IsValueAndOrderAndShapePreserving(const NodeDef& node);

// Returns true if the op preserves the order and value of elements in its
// first input tensor and possible changes its shape.
bool IsValueAndOrderPreserving(const NodeDef& node);

// Returns true if the op in node only rearranges the order of elements in its
// first input tensor and possible changes its shape. More precisely, this
// function returns true if the op commutes with all element-wise operations.
bool IsValuePreserving(const NodeDef& node);

// Returns true if node is idempotent w.r.t. its first input, i.e. if
// Op(Op(x, y, z), y, z) = Op(x, y, z).
bool IsIdempotent(const NodeDef& node);

bool IsUnaryElementWise(const NodeDef& node);

// Returns true if we can find an opdef corresponding to the op of the node.
bool HasOpDef(const NodeDef& node);

// Returns true if the op changes the scalar type of its first input elements
// and preserves the number of elements.
bool IsCastLike(const NodeDef& node);

// Returns true if this op never forwards any of its inputs, i.e. always
// allocates buffers for its inputs.
bool NeverForwardsInputs(const NodeDef& node);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OP_TYPES_H_
