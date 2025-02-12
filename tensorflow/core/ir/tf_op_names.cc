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

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"

namespace mlir {
namespace tfg {

bool TFGraphDialect::IsAdd(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();

  if (op_name == add_v2_) return true;
  if (op_name == add_)
    return !mlir::isa<StringType>(op->getAttrOfType<TypeAttr>("T").getValue());
  return false;
}

bool TFGraphDialect::IsAddN(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == add_n_;
}

bool TFGraphDialect::IsAll(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == all_;
}

bool TFGraphDialect::IsAngle(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == angle_;
}

bool TFGraphDialect::IsAny(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == any_;
}

bool TFGraphDialect::IsAnyDiv(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == real_div_ || op_name == div_ || IsXdivy(op) ||
         op_name == floor_div_ || op_name == truncate_div_;
}

bool TFGraphDialect::IsAnyBatchMatMul(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == batch_matmul_ || op_name == batch_matmul_v2_;
}

bool TFGraphDialect::IsAnyMatMul(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == matmul_ || op_name == sparse_matmul_ ||
         IsAnyBatchMatMul(op) || IsQuantizedMatMul(op);
}

bool TFGraphDialect::IsAnyMax(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == max_ || op_name == segment_max_ ||
         op_name == unsorted_segment_max_;
}

bool TFGraphDialect::IsAnyMaxPool(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == max_pool_ || op_name == max_pool_v2_ ||
         op_name == max_pool_3d_ || op_name == max_pool_with_argmax_ ||
         op_name == fractional_max_pool_;
}

bool TFGraphDialect::IsAnyMin(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == min_ || op_name == segment_min_ ||
         op_name == unsorted_segment_min_;
}

bool TFGraphDialect::IsAnySparseSegmentReduction(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sparse_segment_sum_ ||
         op_name == sparse_segment_sum_with_num_segments_ ||
         op_name == sparse_segment_mean_ ||
         op_name == sparse_segment_mean_with_num_segments_ ||
         op_name == sparse_segment_sqrtn_ ||
         op_name == sparse_segment_sqrtn_with_num_segments_;
}

bool TFGraphDialect::IsApproximateEqual(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == approximate_equal_;
}

bool TFGraphDialect::IsArg(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == arg_ || op_name == device_arg_;
}

bool TFGraphDialect::IsArgMax(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == arg_max_;
}

bool TFGraphDialect::IsArgMin(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == arg_min_;
}

bool TFGraphDialect::IsAvgPoolGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == arg_pool_grad_;
}

bool TFGraphDialect::IsAssign(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == assign_ || op_name == assign_variable_op_;
}

bool TFGraphDialect::IsAssert(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == assert_;
}

bool TFGraphDialect::IsAsString(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == as_string_;
}

bool TFGraphDialect::IsAtan2(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == atan2_;
}

bool TFGraphDialect::IsBetainc(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == betainc_;
}

bool TFGraphDialect::IsBiasAdd(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == bias_add_ || op_name == bias_add_v1_;
}

bool TFGraphDialect::IsBiasAddV2(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == bias_add_;
}

bool TFGraphDialect::IsBiasAddGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == bias_add_grad_;
}

bool TFGraphDialect::IsBitcast(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == bitcast_;
}

bool TFGraphDialect::IsBroadcastTo(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == broadcast_to_;
}

bool TFGraphDialect::IsCast(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == cast_;
}

bool TFGraphDialect::IsCheckNumerics(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == check_numerics_;
}

bool TFGraphDialect::IsCollective(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == collective_reduce_ || op_name == collective_bcast_send_ ||
         op_name == collective_bcast_recv_;
}

bool TFGraphDialect::IsComplex(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == complex_;
}

bool TFGraphDialect::IsComplexAbs(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == complex_abs_;
}

bool TFGraphDialect::IsConcat(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == concat_ || IsConcatV2(op);
}

bool TFGraphDialect::IsConcatV2(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == concat_v2_;
}

bool TFGraphDialect::IsConcatOffset(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == concat_offset_;
}

bool TFGraphDialect::IsConstant(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == const_;
}

bool TFGraphDialect::IsConj(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conj_;
}

bool TFGraphDialect::IsConjugateTranspose(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conjugate_transpose_;
}

// TODO(chiahungduan): Should we use certain helpers like IsEnter().
bool TFGraphDialect::IsControlFlow(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();

  return op_name == control_trigger_ || op_name == enter_ || op_name == exit_ ||
         op_name == loop_cond_ || op_name == merge_ || op_name == xla_merge_ ||
         op_name == next_iteration_ || op_name == switch_ ||
         op_name == switch_n_;
}

bool TFGraphDialect::IsConv2D(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conv_2d_;
}

bool TFGraphDialect::IsConv2DBackpropFilter(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conv_2d_back_prop_filter_;
}

bool TFGraphDialect::IsConv2DBackpropInput(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conv_2d_back_prop_input_;
}

bool TFGraphDialect::IsConv3D(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conv_3d_;
}

bool TFGraphDialect::IsConv3DBackpropFilterV2(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conv_3d_back_prop_filter_v2_;
}

bool TFGraphDialect::IsConv3DBackpropInputV2(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == conv_3d_back_prop_input_v2_;
}

bool TFGraphDialect::IsDepthwiseConv2dNative(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == depth_wise_conv_2d_native_;
}

bool TFGraphDialect::IsDepthwiseConv2dNativeBackpropFilter(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == depth_wise_conv_2d_native_back_prop_filter_;
}

bool TFGraphDialect::IsDepthwiseConv2dNativeBackpropInput(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == depth_wise_conv_2d_native_back_prop_input_;
}

bool TFGraphDialect::IsDequeueOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == queue_dequeue_ || op_name == queue_dequeue_v2_ ||
         op_name == queue_dequeue_many_ || op_name == queue_dequeue_many_v2_ ||
         op_name == queue_dequeue_upto_ || op_name == queue_dequeue_upto_v2_;
}

bool TFGraphDialect::IsDiv(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == div_;
}

bool TFGraphDialect::IsDivNoNan(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == div_no_nan_;
}

bool TFGraphDialect::IsElu(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == elu_;
}

bool TFGraphDialect::IsEluGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == elu_grad_;
}

bool TFGraphDialect::IsQuantizationEmulation(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == quantize_and_dequantize_ ||
         op_name == quantize_and_dequantize_v2_ ||
         op_name == quantize_and_dequantize_v3_ ||
         op_name == quantize_and_dequantize_v4_ ||
         op_name == quantize_and_dequantize_v4_grad_ ||
         op_name == fake_quant_with_min_max_args_ ||
         op_name == fake_quant_with_min_max_args_gradient_ ||
         op_name == fake_quant_with_min_max_vars_ ||
         op_name == fake_quant_with_min_max_vars_gradient_ ||
         op_name == fake_quant_with_min_max_vars_per_channel_ ||
         op_name == fake_quant_with_min_max_vars_per_channel_gradient_;
}

bool TFGraphDialect::IsEnter(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == enter_ || op_name == ref_enter_;
}

bool TFGraphDialect::IsEqual(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == equal_;
}

bool TFGraphDialect::IsExit(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == exit_ || op_name == ref_exit_;
}

bool TFGraphDialect::IsExp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == exp_;
}

bool TFGraphDialect::IsFakeParam(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == fake_param_;
}

bool TFGraphDialect::IsFill(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == fill_;
}

bool TFGraphDialect::IsFloorDiv(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == floor_div_;
}

bool TFGraphDialect::IsFloorMod(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == floor_mod_;
}

bool TFGraphDialect::IsFusedBatchNorm(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == fused_batch_norm_ || op_name == fused_batch_norm_v2_ ||
         op_name == fused_batch_norm_v3_;
}

bool TFGraphDialect::IsFusedBatchNormEx(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == fused_batch_norm_ex_;
}

bool TFGraphDialect::IsFusedBatchNormGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == fused_batch_norm_grad_ ||
         op_name == fused_batch_norm_grad_v2_ ||
         op_name == fused_batch_norm_grad_v3_;
}

bool TFGraphDialect::IsGather(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == gather_ || op_name == gather_v2_ ||
         op_name == resource_gather_;
}

bool TFGraphDialect::IsGreater(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == greater_;
}

bool TFGraphDialect::IsGreaterEqual(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == greater_equal_;
}

bool TFGraphDialect::IsHostConstant(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == host_const_;
}

bool TFGraphDialect::IsHistogramSummary(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == histogram_summary_;
}

bool TFGraphDialect::IsIdentity(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == identity_ || op_name == ref_identity_;
}

bool TFGraphDialect::IsIdentityN(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == identity_n_;
}

bool TFGraphDialect::IsIdentityNSingleInput(TFOp op) const {
  if (!IsIdentityN(op)) return false;
  auto array_attr = op->getAttrOfType<ArrayAttr>("T");
  if (!array_attr) return false;
  // TODO(chiahungduan): Do we need to check the content of array_attr?
  return array_attr.size() == 1;
}

bool TFGraphDialect::IsIf(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == if_ || op_name == stateless_if_;
}

bool TFGraphDialect::IsIgamma(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == igamma_;
}

bool TFGraphDialect::IsIgammac(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == igammac_;
}

bool TFGraphDialect::IsImag(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == imag_;
}

bool TFGraphDialect::IsImmutableConst(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == immutable_const_;
}

bool TFGraphDialect::IsInvGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == inv_grad_;
}

bool TFGraphDialect::IsLeakyRelu(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == leaky_relu_;
}

bool TFGraphDialect::IsLeakyReluGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == leaky_relu_grad_;
}

bool TFGraphDialect::IsLess(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == less_;
}

bool TFGraphDialect::IsLessEqual(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == less_equal_;
}

bool TFGraphDialect::IsLog(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == log_;
}

bool TFGraphDialect::IsLogicalAnd(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == logical_and_;
}

bool TFGraphDialect::IsLogicalNot(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == logical_not_;
}

bool TFGraphDialect::IsLogicalOr(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == logical_or_;
}

bool TFGraphDialect::IsLoopCond(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == loop_cond_;
}

bool TFGraphDialect::IsMatMul(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == matmul_;
}

bool TFGraphDialect::IsMax(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == max_;
}

bool TFGraphDialect::IsMaximum(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == maximum_;
}

bool TFGraphDialect::IsMaxPoolGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == max_pool_grad_;
}

bool TFGraphDialect::IsMean(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == mean_;
}

bool TFGraphDialect::IsMerge(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == merge_ || op_name == ref_merge_ || op_name == xla_merge_;
}

bool TFGraphDialect::IsMin(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == min_;
}

bool TFGraphDialect::IsMinimum(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == minimum_;
}

bool TFGraphDialect::IsMirrorPad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == mirror_pad_;
}

bool TFGraphDialect::IsMirrorPadGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == mirror_pad_grad_;
}

bool TFGraphDialect::IsMod(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == mod_;
}

bool TFGraphDialect::IsMul(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == mul_;
}
bool TFGraphDialect::IsMulNoNan(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == mul_no_nan_;
}
bool TFGraphDialect::IsAnyMul(TFOp op) const {
  return IsMul(op) || IsMulNoNan(op);
}

bool TFGraphDialect::IsNeg(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == neg_;
}

bool TFGraphDialect::IsNoOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == no_op_;
}

bool TFGraphDialect::IsNotEqual(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == not_equal_;
}

bool TFGraphDialect::IsNextIteration(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == next_iteration_ || op_name == ref_next_iteration_;
}

bool TFGraphDialect::IsOnesLike(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == ones_like_;
}

bool TFGraphDialect::IsPack(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == pack_;
}

bool TFGraphDialect::IsPad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == pad_ || op_name == pad_v2_;
}

bool TFGraphDialect::IsPartitionedCall(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == partitioned_call_;
}

bool TFGraphDialect::IsPlaceholder(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == placeholder_ || op_name == placeholder_v2_ ||
         op_name == placeholder_with_default_;
}

bool TFGraphDialect::IsPolygamma(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == poly_gamma_;
}

bool TFGraphDialect::IsPow(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == pow_;
}

bool TFGraphDialect::IsPrint(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == print_ || op_name == print_v2_;
}

bool TFGraphDialect::IsProd(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == prod_;
}

bool TFGraphDialect::IsQuantizedMatMul(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == quantized_matmul_ || op_name == quantized_matmul_v2_;
}

bool TFGraphDialect::IsQueue(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == random_shuffle_queue_v2_ || op_name == fifo_queue_v2_ ||
         op_name == padding_fifo_queue_v2_ || op_name == priority_queue_v2_;
}

bool TFGraphDialect::IsRandomShuffle(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == random_shuffle_;
}

bool TFGraphDialect::IsRank(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == rank_;
}

bool TFGraphDialect::IsReadVariableOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == read_variable_op_;
}

bool TFGraphDialect::IsReadVariablesOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == read_variables_op_;
}

bool TFGraphDialect::IsReal(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == real_;
}

bool TFGraphDialect::IsRealDiv(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == real_div_;
}

bool TFGraphDialect::IsReciprocalGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == reciprocal_grad_;
}

bool TFGraphDialect::IsRecv(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == recv_ || op_name == host_recv_;
}

bool TFGraphDialect::IsReduction(TFOp op) const {
  return IsSum(op) || IsProd(op) || IsMin(op) || IsMax(op) || IsMean(op) ||
         IsAny(op) || IsAll(op);
}

bool TFGraphDialect::IsRelu(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == relu_;
}

bool TFGraphDialect::IsRelu6(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == relu6_;
}

bool TFGraphDialect::IsReluGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == relu_grad_;
}

bool TFGraphDialect::IsRelu6Grad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == relu6_grad_;
}

bool TFGraphDialect::IsReshape(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == reshape_;
}

bool TFGraphDialect::IsRestore(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == restore_ || op_name == restore_v2_ ||
         op_name == restore_slice_;
}

bool TFGraphDialect::IsReturn(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == return_;
}

bool TFGraphDialect::IsRetval(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == retval_ || op_name == device_retval_;
}

bool TFGraphDialect::IsReverse(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == reverse_ || IsReverseV2(op);
}

bool TFGraphDialect::IsReverseV2(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == reverse_v2_;
}

bool TFGraphDialect::IsRsqrt(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == rsqrt_;
}

bool TFGraphDialect::IsRsqrtGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == rsqrt_grad_;
}

bool TFGraphDialect::IsSelect(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == select_ || op_name == select_v2_;
}

bool TFGraphDialect::IsSeluGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == selu_grad_;
}

bool TFGraphDialect::IsSend(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == send_ || op_name == host_send_;
}

bool TFGraphDialect::IsShape(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == shape_;
}

bool TFGraphDialect::IsShapeN(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == shape_n_;
}

bool TFGraphDialect::IsShuffle(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == shuffle_;
}

bool TFGraphDialect::IsSigmoid(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sigmoid_;
}

bool TFGraphDialect::IsSigmoidGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sigmoid_grad_;
}

bool TFGraphDialect::IsSize(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == size_;
}

bool TFGraphDialect::IsSlice(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == slice_;
}

bool TFGraphDialect::IsSnapshot(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == snapshot_;
}

bool TFGraphDialect::IsSoftmax(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == softmax_;
}

bool TFGraphDialect::IsSoftplus(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == softplus_;
}

bool TFGraphDialect::IsSoftplusGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == softplus_grad_;
}

bool TFGraphDialect::IsSoftsignGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == softsign_grad_;
}

bool TFGraphDialect::IsSplit(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == split_;
}

bool TFGraphDialect::IsSplitV(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == split_v_;
}

bool TFGraphDialect::IsSqrt(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sqrt_;
}

bool TFGraphDialect::IsSqrtGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sqrt_grad_;
}

bool TFGraphDialect::IsSquare(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == square_;
}

bool TFGraphDialect::IsSquaredDifference(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == squared_difference_;
}

bool TFGraphDialect::IsSqueeze(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == squeeze_;
}

bool TFGraphDialect::IsStackOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == stack_ || op_name == stack_v2_;
}

bool TFGraphDialect::IsStackCloseOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == stack_close_ || op_name == stack_close_v2_;
}

bool TFGraphDialect::IsStackPushOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == stack_push_ || op_name == stack_push_v2_;
}

bool TFGraphDialect::IsStackPopOp(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == stack_pop_ || op_name == stack_pop_v2_;
}

bool TFGraphDialect::IsStatefulPartitionedCall(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == stateful_partitioned_call_;
}

bool TFGraphDialect::IsStopGradient(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == stop_gradient_ || op_name == prevent_gradient_;
}

bool TFGraphDialect::IsStridedSlice(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == strided_slice_;
}

bool TFGraphDialect::IsStridedSliceGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == strided_slice_grad_;
}

bool TFGraphDialect::IsStringToHashBucketFast(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == string_to_hashbucket_fast_;
}

bool TFGraphDialect::IsSub(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sub_;
}

bool TFGraphDialect::IsSum(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == sum_;
}

bool TFGraphDialect::IsSwitch(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == switch_ || op_name == switch_n_ || op_name == ref_switch_;
}

bool TFGraphDialect::IsSymbolicGradient(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == symbolic_gradient_;
}

bool TFGraphDialect::IsTanh(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == tanh_;
}

bool TFGraphDialect::IsTanhGrad(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == tanh_grad_;
}

bool TFGraphDialect::IsTile(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == tile_;
}

bool TFGraphDialect::IsTranspose(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == transpose_;
}

bool TFGraphDialect::IsTruncateDiv(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == truncate_div_;
}

bool TFGraphDialect::IsTruncateMod(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == truncate_mod_;
}

bool TFGraphDialect::IsUnique(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == unique_ || op_name == unique_v2_;
}

bool TFGraphDialect::IsUnpack(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == unpack_;
}

bool TFGraphDialect::IsVariable(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == variable_ || op_name == variable_v2_ ||
         op_name == auto_reload_variable_ || op_name == var_handle_op_ ||
         op_name == var_handles_op_ || IsReadVariableOp(op) ||
         IsReadVariablesOp(op);
}

bool TFGraphDialect::IsWhile(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == while_ || op_name == stateless_while_;
}

bool TFGraphDialect::IsXdivy(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == xdivy_;
}

bool TFGraphDialect::IsZerosLike(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == zeros_like_;
}

bool TFGraphDialect::IsZeta(TFOp op) const {
  StringAttr op_name = op->getName().getIdentifier();
  return op_name == zeta_;
}

}  // namespace tfg
}  // namespace mlir
