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
#include "tensorflow/core/transforms/utils/op_cat_helper.h"

#include "absl/strings/match.h"
#include "llvm/ADT/TypeSwitch.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/platform/str_util.h"

namespace mlir {
namespace tfg {

namespace {

bool SplatElementsAttrHasValue(SplatElementsAttr attr, float v) {
  Type type = attr.getElementType();

#define IF_SPLAT_VALUE_IS(DTYPE, VALUE)                                \
  if (attr.getSplatValue<tensorflow::EnumToDataType<DTYPE>::Type>() == \
      tensorflow::EnumToDataType<DTYPE>::Type(VALUE))                  \
    return true;

  if (type.isInteger(1)) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_BOOL, v);
  } else if (type.isSignedInteger()) {
    if (type.isInteger(8)) {
      IF_SPLAT_VALUE_IS(tensorflow::DT_INT8, v);
    } else if (type.isInteger(16)) {
      IF_SPLAT_VALUE_IS(tensorflow::DT_INT16, v);
    } else if (type.isInteger(32)) {
      IF_SPLAT_VALUE_IS(tensorflow::DT_INT32, v);
    } else if (type.isInteger(64)) {
      IF_SPLAT_VALUE_IS(tensorflow::DT_INT64, v);
    }
  } else if (type.isUnsignedInteger()) {
    if (type.isInteger(8)) IF_SPLAT_VALUE_IS(tensorflow::DT_UINT8, v);
    if (type.isInteger(16)) IF_SPLAT_VALUE_IS(tensorflow::DT_UINT16, v);
  } else if (type.isF16()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_HALF, v);
  } else if (type.isF32()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_FLOAT, v);
  } else if (type.isF64()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_DOUBLE, v);
  } else if (type.isBF16()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_BFLOAT16, v);
  } else if (type.isa<ComplexType>()) {
    ComplexType complex_type = type.cast<ComplexType>();
    if (complex_type.getElementType().isF32()) {
      IF_SPLAT_VALUE_IS(tensorflow::DT_COMPLEX64, v);
    } else if (complex_type.getElementType().isF64()) {
      IF_SPLAT_VALUE_IS(tensorflow::DT_COMPLEX128, v);
    }
  } else if (type.isa<tf_type::Qint8Type>()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_QINT8, v);
  } else if (type.isa<tf_type::Qint16Type>()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_QINT16, v);
  } else if (type.isa<tf_type::Qint32Type>()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_QINT32, v);
  } else if (type.isa<tf_type::Quint8Type>()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_QUINT8, v);
  } else if (type.isa<tf_type::Quint16Type>()) {
    IF_SPLAT_VALUE_IS(tensorflow::DT_QUINT16, v);
  }
#undef IF_SPLAT_VALUE_IS
  return false;
}

}  // namespace

bool OpCatHelper::IsAdd(TFOp op) {
  static StringRef add_v2 = "AddV2";
  static StringRef add = "Add";
  StringRef op_name = op->getName().stripDialect();

  if (op_name == add_v2) return true;
  if (op_name == add) return !op->getAttrOfType<StringAttr>("T");
  return false;
}

bool OpCatHelper::IsAddN(TFOp op) {
  static StringRef add_n = "AddN";
  StringRef op_name = op->getName().stripDialect();
  return op_name == add_n;
}

bool OpCatHelper::IsAll(TFOp op) {
  static StringRef all = "All";
  StringRef op_name = op->getName().stripDialect();
  return op_name == all;
}

bool OpCatHelper::IsAngle(TFOp op) {
  static StringRef angle = "Angle";
  StringRef op_name = op->getName().stripDialect();
  return op_name == angle;
}

bool OpCatHelper::IsAny(TFOp op) {
  static StringRef any = "Any";
  StringRef op_name = op->getName().stripDialect();
  return op_name == any;
}

bool OpCatHelper::IsAnyDiv(TFOp op) {
  static StringRef real_div = "RealDiv";
  static StringRef div = "Div";
  static StringRef floor_div = "FloorDiv";
  static StringRef truncate_div = "TruncateDiv";
  StringRef op_name = op->getName().stripDialect();
  return op_name == real_div || op_name == div || IsXdivy(op) ||
         op_name == floor_div || op_name == truncate_div;
}

bool OpCatHelper::IsAnyBatchMatMul(TFOp op) {
  static StringRef batch_matmul = "BatchMatMul";
  static StringRef batch_matmul_v2 = "BatchMatMulV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == batch_matmul || op_name == batch_matmul_v2;
}

bool OpCatHelper::IsAnyMatMul(TFOp op) {
  static StringRef matmul = "MatMul";
  static StringRef sparse_matmul = "SparseMatMul";
  StringRef op_name = op->getName().stripDialect();
  return op_name == matmul || op_name == sparse_matmul ||
         IsAnyBatchMatMul(op) || IsQuantizedMatMul(op);
}

bool OpCatHelper::IsAnyMax(TFOp op) {
  static StringRef max = "Max";
  static StringRef segment_max = "SegmentMax";
  static StringRef unsorted_segment_max = "UnsortedSegmentMax";
  StringRef op_name = op->getName().stripDialect();
  return op_name == max || op_name == segment_max ||
         op_name == unsorted_segment_max;
}

bool OpCatHelper::IsAnyMaxPool(TFOp op) {
  static StringRef max_pool = "MaxPool";
  static StringRef max_pool_v2 = "MaxPoolV2";
  static StringRef max_pool_3d = "MaxPool3D";
  static StringRef max_pool_with_argmax = "MaxPoolWithArgmax";
  static StringRef fractional_max_pool = "FractionalMaxPool";
  StringRef op_name = op->getName().stripDialect();
  return op_name == max_pool || op_name == max_pool_v2 ||
         op_name == max_pool_3d || op_name == max_pool_with_argmax ||
         op_name == fractional_max_pool;
}

bool OpCatHelper::IsAnyMin(TFOp op) {
  static StringRef min = "Min";
  static StringRef segment_min = "SegmentMin";
  static StringRef unsorted_segment_min = "UnsortedSegmentMin";
  StringRef op_name = op->getName().stripDialect();
  return op_name == min || op_name == segment_min ||
         op_name == unsorted_segment_min;
}

bool OpCatHelper::IsAnySparseSegmentReduction(TFOp op) {
  static StringRef sparse_segment_sum = "SparseSegmentSum";
  static StringRef sparse_segment_sum_with_num_segments =
      "SparseSegmentSumWithNumSegments";
  static StringRef sparse_segment_mean = "SparseSegmentMean";
  static StringRef sparse_segment_mean_with_num_segments =
      "SparseSegmentMeanWithNumSegments";
  static StringRef sparse_segment_sqrtn = "SparseSegmentSqrtN";
  static StringRef sparse_segment_sqrtn_with_num_segments =
      "SparseSegmentSqrtNWithNumSegments";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sparse_segment_sum ||
         op_name == sparse_segment_sum_with_num_segments ||
         op_name == sparse_segment_mean ||
         op_name == sparse_segment_mean_with_num_segments ||
         op_name == sparse_segment_sqrtn ||
         op_name == sparse_segment_sqrtn_with_num_segments;
}

bool OpCatHelper::IsApproximateEqual(TFOp op) {
  static StringRef approximate_equal = "ApproximateEqual";
  StringRef op_name = op->getName().stripDialect();
  return op_name == approximate_equal;
}

bool OpCatHelper::IsArg(TFOp op) {
  static StringRef _arg = "_Arg";
  static StringRef _device_arg = "_DeviceArg";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _arg || op_name == _device_arg;
}

bool OpCatHelper::IsArgMax(TFOp op) {
  static StringRef arg_max = "ArgMax";
  StringRef op_name = op->getName().stripDialect();
  return op_name == arg_max;
}

bool OpCatHelper::IsArgMin(TFOp op) {
  static StringRef arg_min = "ArgMin";
  StringRef op_name = op->getName().stripDialect();
  return op_name == arg_min;
}

bool OpCatHelper::IsAvgPoolGrad(TFOp op) {
  static StringRef arg_pool_grad = "AvgPoolGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == arg_pool_grad;
}

bool OpCatHelper::IsAssign(TFOp op) {
  static StringRef assign = "Assign";
  static StringRef assign_variable_op = "AssignVariableOp";
  StringRef op_name = op->getName().stripDialect();
  return op_name == assign || op_name == assign_variable_op;
}

bool OpCatHelper::IsAssert(TFOp op) {
  static StringRef _assert = "Assert";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _assert;
}

bool OpCatHelper::IsAsString(TFOp op) {
  static StringRef as_string = "AsString";
  StringRef op_name = op->getName().stripDialect();
  return op_name == as_string;
}

bool OpCatHelper::IsAtan2(TFOp op) {
  static StringRef atan2 = "Atan2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == atan2;
}

bool OpCatHelper::IsBetainc(TFOp op) {
  static StringRef betainc = "Betainc";
  StringRef op_name = op->getName().stripDialect();
  return op_name == betainc;
}

bool OpCatHelper::IsBiasAdd(TFOp op) {
  static StringRef bias_add = "BiasAdd";
  static StringRef bias_add_v1 = "BiasAddV1";
  StringRef op_name = op->getName().stripDialect();
  return op_name == bias_add || op_name == bias_add_v1;
}

bool OpCatHelper::IsBiasAddV2(TFOp op) {
  static StringRef bias_add = "BiasAdd";
  StringRef op_name = op->getName().stripDialect();
  return op_name == bias_add;
}

bool OpCatHelper::IsBiasAddGrad(TFOp op) {
  static StringRef bias_add_grad = "BiasAddGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == bias_add_grad;
}

bool OpCatHelper::IsBitcast(TFOp op) {
  static StringRef bitcast = "Bitcast";
  StringRef op_name = op->getName().stripDialect();
  return op_name == bitcast;
}

bool OpCatHelper::IsBroadcastTo(TFOp op) {
  static StringRef broadcast_to = "BroadcastTo";
  StringRef op_name = op->getName().stripDialect();
  return op_name == broadcast_to;
}

bool OpCatHelper::IsCast(TFOp op) {
  static StringRef cast = "Cast";
  StringRef op_name = op->getName().stripDialect();
  return op_name == cast;
}

bool OpCatHelper::IsCheckNumerics(TFOp op) {
  static StringRef check_numerics = "CheckNumerics";
  StringRef op_name = op->getName().stripDialect();
  return op_name == check_numerics;
}

bool OpCatHelper::IsCollective(TFOp op) {
  static StringRef collective_reduce = "CollectiveReduce";
  static StringRef collective_bcast_send = "CollectiveBcastSend";
  static StringRef collective_bcast_recv = "CollectiveBcastRecv";
  StringRef op_name = op->getName().stripDialect();
  return op_name == collective_reduce || op_name == collective_bcast_send ||
         op_name == collective_bcast_recv;
}

bool OpCatHelper::IsComplex(TFOp op) {
  static StringRef _complex = "Complex";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _complex;
}

bool OpCatHelper::IsComplexAbs(TFOp op) {
  static StringRef complex_abs = "ComplexAbs";
  StringRef op_name = op->getName().stripDialect();
  return op_name == complex_abs;
}

bool OpCatHelper::IsConcat(TFOp op) {
  static StringRef concat = "Concat";
  StringRef op_name = op->getName().stripDialect();
  return op_name == concat || IsConcatV2(op);
}

bool OpCatHelper::IsConcatV2(TFOp op) {
  static StringRef concat_v2 = "ConcatV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == concat_v2;
}

bool OpCatHelper::IsConcatOffset(TFOp op) {
  static StringRef concat_offset = "ConcatOffset";
  StringRef op_name = op->getName().stripDialect();
  return op_name == concat_offset;
}

bool OpCatHelper::IsConstant(TFOp op) {
  static StringRef _const = "Const";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _const;
}

bool OpCatHelper::IsConj(TFOp op) {
  static StringRef conj = "Conj";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conj;
}

bool OpCatHelper::IsConjugateTranspose(TFOp op) {
  static StringRef conjugate_transpose = "ConjugateTranspose";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conjugate_transpose;
}

// TODO(chiahungduan): Should we use certain helpers like IsEnter().
bool OpCatHelper::IsControlFlow(TFOp op) {
  static StringRef control_trigger = "ControlTrigger";
  static StringRef enter = "Enter";
  static StringRef exit = "Exit";
  static StringRef loop_cond = "LoopCond";
  static StringRef merge = "Merge";
  static StringRef _xla_merge = "_XlaMerge";
  static StringRef next_iteration = "NextIteration";
  static StringRef _switch = "Switch";
  static StringRef _switch_n = "_SwitchN";
  StringRef op_name = op->getName().stripDialect();

  return op_name == control_trigger || op_name == enter || op_name == exit ||
         op_name == loop_cond || op_name == merge || op_name == _xla_merge ||
         op_name == next_iteration || op_name == _switch ||
         op_name == _switch_n;
}

bool OpCatHelper::IsConv2D(TFOp op) {
  static StringRef conv_2d = "Conv2D";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conv_2d;
}

bool OpCatHelper::IsConv2DBackpropFilter(TFOp op) {
  static StringRef conv_2d_back_prop_filter = "Conv2DBackpropFilter";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conv_2d_back_prop_filter;
}

bool OpCatHelper::IsConv2DBackpropInput(TFOp op) {
  static StringRef conv_2d_back_prop_input = "Conv2DBackpropInput";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conv_2d_back_prop_input;
}

bool OpCatHelper::IsConv3D(TFOp op) {
  static StringRef conv_3d = "Conv3D";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conv_3d;
}

bool OpCatHelper::IsConv3DBackpropFilterV2(TFOp op) {
  static StringRef conv_3d_back_prop_filter_v2 = "Conv3DBackpropFilterV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conv_3d_back_prop_filter_v2;
}

bool OpCatHelper::IsConv3DBackpropInputV2(TFOp op) {
  static StringRef conv_3d_back_prop_input_v2 = "Conv3DBackpropInputV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == conv_3d_back_prop_input_v2;
}

bool OpCatHelper::IsDepthwiseConv2dNative(TFOp op) {
  static StringRef depth_wise_conv_2d_native = "DepthwiseConv2dNative";
  StringRef op_name = op->getName().stripDialect();
  return op_name == depth_wise_conv_2d_native;
}

bool OpCatHelper::IsDepthwiseConv2dNativeBackpropFilter(TFOp op) {
  static StringRef depth_wise_conv_2d_native_back_prop_filter =
      "DepthwiseConv2dNativeBackpropFilter";
  StringRef op_name = op->getName().stripDialect();
  return op_name == depth_wise_conv_2d_native_back_prop_filter;
}

bool OpCatHelper::IsDepthwiseConv2dNativeBackpropInput(TFOp op) {
  static StringRef depth_wise_conv_2d_native_back_prop_input =
      "DepthwiseConv2dNativeBackpropInput";
  StringRef op_name = op->getName().stripDialect();
  return op_name == depth_wise_conv_2d_native_back_prop_input;
}

bool OpCatHelper::IsDequeueOp(TFOp op) {
  static StringRef queue_dequeue = "QueueDequeue";
  static StringRef queue_dequeue_v2 = "QueueDequeueV2";
  static StringRef queue_dequeue_many = "QueueDequeueMany";
  static StringRef queue_dequeue_many_v2 = "QueueDequeueManyV2";
  static StringRef queue_dequeue_upto = "QueueDequeueUpTo";
  static StringRef queue_dequeue_upto_v2 = "QueueDequeueUpToV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == queue_dequeue || op_name == queue_dequeue_v2 ||
         op_name == queue_dequeue_many || op_name == queue_dequeue_many_v2 ||
         op_name == queue_dequeue_upto || op_name == queue_dequeue_upto_v2;
}

bool OpCatHelper::IsDiv(TFOp op) {
  static StringRef div = "Div";
  StringRef op_name = op->getName().stripDialect();
  return op_name == div;
}

bool OpCatHelper::IsDivNoNan(TFOp op) {
  static StringRef div_no_nan = "DivNoNan";
  StringRef op_name = op->getName().stripDialect();
  return op_name == div_no_nan;
}

bool OpCatHelper::IsElu(TFOp op) {
  static StringRef elu = "Elu";
  StringRef op_name = op->getName().stripDialect();
  return op_name == elu;
}

bool OpCatHelper::IsEluGrad(TFOp op) {
  static StringRef elu_grad = "EluGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == elu_grad;
}

bool OpCatHelper::IsQuantizationEmulation(TFOp op) {
  StringRef op_name = op->getName().stripDialect();
  return absl::StartsWith(op_name.data(), "QuantizeAndDequantize") ||
         absl::StartsWith(op_name.data(), "FakeQuantWithMinMax");
}

bool OpCatHelper::IsEnter(TFOp op) {
  static StringRef enter = "Enter";
  static StringRef ref_enter = "RefEnter";
  StringRef op_name = op->getName().stripDialect();
  return op_name == enter || op_name == ref_enter;
}

bool OpCatHelper::IsEqual(TFOp op) {
  static StringRef equal = "Equal";
  StringRef op_name = op->getName().stripDialect();
  return op_name == equal;
}

bool OpCatHelper::IsExit(TFOp op) {
  static StringRef exit = "Exit";
  static StringRef ref_exit = "RefExit";
  StringRef op_name = op->getName().stripDialect();
  return op_name == exit || op_name == ref_exit;
}

bool OpCatHelper::IsExp(TFOp op) {
  static StringRef exp = "Exp";
  StringRef op_name = op->getName().stripDialect();
  return op_name == exp;
}

bool OpCatHelper::IsFakeParam(TFOp op) {
  static StringRef fake_param = "FakeParam";
  StringRef op_name = op->getName().stripDialect();
  return op_name == fake_param;
}

bool OpCatHelper::IsFill(TFOp op) {
  static StringRef fill = "Fill";
  StringRef op_name = op->getName().stripDialect();
  return op_name == fill;
}

bool OpCatHelper::IsFloorDiv(TFOp op) {
  static StringRef floor_div = "FloorDiv";
  StringRef op_name = op->getName().stripDialect();
  return op_name == floor_div;
}

bool OpCatHelper::IsFloorMod(TFOp op) {
  static StringRef floor_mod = "FloorMod";
  StringRef op_name = op->getName().stripDialect();
  return op_name == floor_mod;
}

bool OpCatHelper::IsFusedBatchNorm(TFOp op) {
  static StringRef fused_batch_norm = "FusedBatchNorm";
  static StringRef fused_batch_norm_v2 = "FusedBatchNormV2";
  static StringRef fused_batch_norm_v3 = "FusedBatchNormV3";
  StringRef op_name = op->getName().stripDialect();
  return op_name == fused_batch_norm || op_name == fused_batch_norm_v2 ||
         op_name == fused_batch_norm_v3;
}

bool OpCatHelper::IsFusedBatchNormEx(TFOp op) {
  static StringRef _fused_batch_norm_ex = "_FusedBatchNormEx";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _fused_batch_norm_ex;
}

bool OpCatHelper::IsFusedBatchNormGrad(TFOp op) {
  static StringRef fused_batch_norm_grad = "FusedBatchNormGrad";
  static StringRef fused_batch_norm_grad_v2 = "FusedBatchNormGradV2";
  static StringRef fused_batch_norm_grad_v3 = "FusedBatchNormGradV3";
  StringRef op_name = op->getName().stripDialect();
  return op_name == fused_batch_norm_grad ||
         op_name == fused_batch_norm_grad_v2 ||
         op_name == fused_batch_norm_grad_v3;
}

bool OpCatHelper::IsGather(TFOp op) {
  static StringRef gather = "Gather";
  static StringRef gather_v2 = "GatherV2";
  static StringRef resource_gather = "ResourceGather";
  StringRef op_name = op->getName().stripDialect();
  return op_name == gather || op_name == gather_v2 ||
         op_name == resource_gather;
}

bool OpCatHelper::IsGreater(TFOp op) {
  static StringRef greater = "Greater";
  StringRef op_name = op->getName().stripDialect();
  return op_name == greater;
}

bool OpCatHelper::IsGreaterEqual(TFOp op) {
  static StringRef greater_equal = "GreaterEqual";
  StringRef op_name = op->getName().stripDialect();
  return op_name == greater_equal;
}

bool OpCatHelper::IsHostConstant(TFOp op) {
  static StringRef host_const = "HostConst";
  StringRef op_name = op->getName().stripDialect();
  return op_name == host_const;
}

bool OpCatHelper::IsHistogramSummary(TFOp op) {
  static StringRef histogram_summary = "HistogramSummary";
  StringRef op_name = op->getName().stripDialect();
  return op_name == histogram_summary;
}

bool OpCatHelper::IsIdentity(TFOp op) {
  static StringRef identity = "Identity";
  static StringRef ref_identity = "RefIdentity";
  StringRef op_name = op->getName().stripDialect();
  return op_name == identity || op_name == ref_identity;
}

bool OpCatHelper::IsIdentityN(TFOp op) {
  static StringRef identity_n = "IdentityN";
  StringRef op_name = op->getName().stripDialect();
  return op_name == identity_n;
}

bool OpCatHelper::IsIdentityNSingleInput(TFOp op) {
  if (!IsIdentityN(op)) return false;
  auto array_attr = op->getAttrOfType<ArrayAttr>("T");
  if (!array_attr) return false;
  // TODO(chiahungduan): Do we need to check the content of array_attr?
  return array_attr.size() == 1;
}

bool OpCatHelper::IsIf(TFOp op) {
  static StringRef _if = "If";
  static StringRef stateless_if = "StatelessIf";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _if || op_name == stateless_if;
}

bool OpCatHelper::IsIgamma(TFOp op) {
  static StringRef igamma = "Igamma";
  StringRef op_name = op->getName().stripDialect();
  return op_name == igamma;
}

bool OpCatHelper::IsIgammac(TFOp op) {
  static StringRef igammac = "Igammac";
  StringRef op_name = op->getName().stripDialect();
  return op_name == igammac;
}

bool OpCatHelper::IsImag(TFOp op) {
  static StringRef imag = "Imag";
  StringRef op_name = op->getName().stripDialect();
  return op_name == imag;
}

bool OpCatHelper::IsImmutableConst(TFOp op) {
  static StringRef immutable_const = "ImmutableConst";
  StringRef op_name = op->getName().stripDialect();
  return op_name == immutable_const;
}

bool OpCatHelper::IsInvGrad(TFOp op) {
  static StringRef inv_grad = "InvGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == inv_grad;
}

bool OpCatHelper::IsLeakyRelu(TFOp op) {
  static StringRef leaky_relu = "LeakyRelu";
  StringRef op_name = op->getName().stripDialect();
  return op_name == leaky_relu;
}

bool OpCatHelper::IsLeakyReluGrad(TFOp op) {
  static StringRef leaky_relu_grad = "LeakyReluGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == leaky_relu_grad;
}

bool OpCatHelper::IsLess(TFOp op) {
  static StringRef less = "Less";
  StringRef op_name = op->getName().stripDialect();
  return op_name == less;
}

bool OpCatHelper::IsLessEqual(TFOp op) {
  static StringRef less_equal = "LessEqual";
  StringRef op_name = op->getName().stripDialect();
  return op_name == less_equal;
}

bool OpCatHelper::IsLog(TFOp op) {
  static StringRef log = "Log";
  StringRef op_name = op->getName().stripDialect();
  return op_name == log;
}

bool OpCatHelper::IsLogicalAnd(TFOp op) {
  static StringRef logical_and = "LogicalAnd";
  StringRef op_name = op->getName().stripDialect();
  return op_name == logical_and;
}

bool OpCatHelper::IsLogicalNot(TFOp op) {
  static StringRef logical_not = "LogicalNot";
  StringRef op_name = op->getName().stripDialect();
  return op_name == logical_not;
}

bool OpCatHelper::IsLogicalOr(TFOp op) {
  static StringRef logical_or = "LogicalOr";
  StringRef op_name = op->getName().stripDialect();
  return op_name == logical_or;
}

bool OpCatHelper::IsLoopCond(TFOp op) {
  static StringRef loop_cond = "LoopCond";
  StringRef op_name = op->getName().stripDialect();
  return op_name == loop_cond;
}

bool OpCatHelper::IsMatMul(TFOp op) {
  static StringRef matmul = "MatMul";
  StringRef op_name = op->getName().stripDialect();
  return op_name == matmul;
}

bool OpCatHelper::IsMax(TFOp op) {
  static StringRef max = "Max";
  StringRef op_name = op->getName().stripDialect();
  return op_name == max;
}

bool OpCatHelper::IsMaximum(TFOp op) {
  static StringRef maximum = "Maximum";
  StringRef op_name = op->getName().stripDialect();
  return op_name == maximum;
}

bool OpCatHelper::IsMaxPoolGrad(TFOp op) {
  static StringRef max_pool_grad = "MaxPoolGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == max_pool_grad;
}

bool OpCatHelper::IsMean(TFOp op) {
  static StringRef mean = "Mean";
  StringRef op_name = op->getName().stripDialect();
  return op_name == mean;
}

bool OpCatHelper::IsMerge(TFOp op) {
  static StringRef merge = "Merge";
  static StringRef ref_merge = "RefMerge";
  static StringRef _xla_merge = "_XlaMerge";
  StringRef op_name = op->getName().stripDialect();
  return op_name == merge || op_name == ref_merge || op_name == _xla_merge;
}

bool OpCatHelper::IsMin(TFOp op) {
  static StringRef min = "Min";
  StringRef op_name = op->getName().stripDialect();
  return op_name == min;
}

bool OpCatHelper::IsMinimum(TFOp op) {
  static StringRef minimum = "Minimum";
  StringRef op_name = op->getName().stripDialect();
  return op_name == minimum;
}

bool OpCatHelper::IsMirrorPad(TFOp op) {
  static StringRef mirror_pad = "MirrorPad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == mirror_pad;
}

bool OpCatHelper::IsMirrorPadGrad(TFOp op) {
  static StringRef mirror_pad_grad = "MirrorPadGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == mirror_pad_grad;
}

bool OpCatHelper::IsMod(TFOp op) {
  static StringRef mod = "Mod";
  StringRef op_name = op->getName().stripDialect();
  return op_name == mod;
}

bool OpCatHelper::IsMul(TFOp op) {
  static StringRef mul = "Mul";
  StringRef op_name = op->getName().stripDialect();
  return op_name == mul;
}
bool OpCatHelper::IsMulNoNan(TFOp op) {
  static StringRef mul_no_nan = "MulNoNan";
  StringRef op_name = op->getName().stripDialect();
  return op_name == mul_no_nan;
}
bool OpCatHelper::IsAnyMul(TFOp op) { return IsMul(op) || IsMulNoNan(op); }

bool OpCatHelper::IsNeg(TFOp op) {
  static StringRef neg = "Neg";
  StringRef op_name = op->getName().stripDialect();
  return op_name == neg;
}

bool OpCatHelper::IsNoOp(TFOp op) {
  static StringRef no_op = "NoOp";
  StringRef op_name = op->getName().stripDialect();
  return op_name == no_op;
}

bool OpCatHelper::IsNotEqual(TFOp op) {
  static StringRef not_equal = "NotEqual";
  StringRef op_name = op->getName().stripDialect();
  return op_name == not_equal;
}

bool OpCatHelper::IsNextIteration(TFOp op) {
  static StringRef next_iteration = "NextIteration";
  static StringRef ref_next_iteration = "RefNextIteration";
  StringRef op_name = op->getName().stripDialect();
  return op_name == next_iteration || op_name == ref_next_iteration;
}

bool OpCatHelper::IsOnesLike(TFOp op) {
  static StringRef ones_like = "OnesLike";
  StringRef op_name = op->getName().stripDialect();
  return op_name == ones_like;
}

bool OpCatHelper::IsPack(TFOp op) {
  static StringRef pack = "Pack";
  StringRef op_name = op->getName().stripDialect();
  return op_name == pack;
}

bool OpCatHelper::IsPad(TFOp op) {
  static StringRef pad = "Pad";
  static StringRef pad_v2 = "PadV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == pad || op_name == pad_v2;
}

bool OpCatHelper::IsPartitionedCall(TFOp op) {
  static StringRef partitioned_call = "PartitionedCall";
  StringRef op_name = op->getName().stripDialect();
  return op_name == partitioned_call;
}

bool OpCatHelper::IsPlaceholder(TFOp op) {
  static StringRef placeholder = "Placeholder";
  static StringRef placeholder_v2 = "PlaceholderV2";
  static StringRef placeholder_with_default = "PlaceholderWithDefault";
  StringRef op_name = op->getName().stripDialect();
  return op_name == placeholder || op_name == placeholder_v2 ||
         op_name == placeholder_with_default;
}

bool OpCatHelper::IsPolygamma(TFOp op) {
  static StringRef poly_gamma = "Polygamma";
  StringRef op_name = op->getName().stripDialect();
  return op_name == poly_gamma;
}

bool OpCatHelper::IsPow(TFOp op) {
  static StringRef pow = "Pow";
  StringRef op_name = op->getName().stripDialect();
  return op_name == pow;
}

bool OpCatHelper::IsPrint(TFOp op) {
  static StringRef print = "Print";
  static StringRef print_v2 = "PrintV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == print || op_name == print_v2;
}

bool OpCatHelper::IsProd(TFOp op) {
  static StringRef Prod = "Prod";
  StringRef op_name = op->getName().stripDialect();
  return op_name == Prod;
}

bool OpCatHelper::IsQuantizedMatMul(TFOp op) {
  static StringRef quantized_matmul = "QuantizedMatMul";
  static StringRef quantized_matmul_v2 = "QuantizedMatMulV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == quantized_matmul || op_name == quantized_matmul_v2;
}

bool OpCatHelper::IsQueue(TFOp op) {
  // TODO(chiahungduan): Check if we can use "Queue" helper functions.
  return tensorflow::str_util::EndsWith(op->getName().stripDialect().data(),
                                        "QueueV2");
}

bool OpCatHelper::IsRandomShuffle(TFOp op) {
  static StringRef random_shuffle = "RandomShuffle";
  StringRef op_name = op->getName().stripDialect();
  return op_name == random_shuffle;
}

bool OpCatHelper::IsRank(TFOp op) {
  static StringRef rank = "Rank";
  StringRef op_name = op->getName().stripDialect();
  return op_name == rank;
}

bool OpCatHelper::IsReadVariableOp(TFOp op) {
  static StringRef read_variable_op = "ReadVariableOp";
  StringRef op_name = op->getName().stripDialect();
  return op_name == read_variable_op;
}

bool OpCatHelper::IsReadVariablesOp(TFOp op) {
  static StringRef _read_variables_op = "_ReadVariablesOp";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _read_variables_op;
}

bool OpCatHelper::IsReal(TFOp op) {
  static StringRef real = "Real";
  StringRef op_name = op->getName().stripDialect();
  return op_name == real;
}

bool OpCatHelper::IsRealDiv(TFOp op) {
  static StringRef real_div = "RealDiv";
  StringRef op_name = op->getName().stripDialect();
  return op_name == real_div;
}

bool OpCatHelper::IsReciprocalGrad(TFOp op) {
  static StringRef reciprocal_grad = "ReciprocalGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == reciprocal_grad;
}

bool OpCatHelper::IsRecv(TFOp op) {
  static StringRef _recv = "_Recv";
  static StringRef _host_recv = "_HostRecv";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _recv || op_name == _host_recv;
}

bool OpCatHelper::IsReduction(TFOp op) {
  return IsSum(op) || IsProd(op) || IsMin(op) || IsMax(op) || IsMean(op) ||
         IsAny(op) || IsAll(op);
}

bool OpCatHelper::IsRelu(TFOp op) {
  static StringRef relu = "Relu";
  StringRef op_name = op->getName().stripDialect();
  return op_name == relu;
}

bool OpCatHelper::IsRelu6(TFOp op) {
  static StringRef relu6 = "Relu6";
  StringRef op_name = op->getName().stripDialect();
  return op_name == relu6;
}

bool OpCatHelper::IsReluGrad(TFOp op) {
  static StringRef relu_grad = "ReluGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == relu_grad;
}

bool OpCatHelper::IsRelu6Grad(TFOp op) {
  static StringRef relu6_grad = "Relu6Grad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == relu6_grad;
}

bool OpCatHelper::IsReshape(TFOp op) {
  static StringRef reshape = "Reshape";
  StringRef op_name = op->getName().stripDialect();
  return op_name == reshape;
}

bool OpCatHelper::IsRestore(TFOp op) {
  static StringRef restore = "Restore";
  static StringRef restore_v2 = "RestoreV2";
  static StringRef restore_slice = "RestoreSlice";
  StringRef op_name = op->getName().stripDialect();
  return op_name == restore || op_name == restore_v2 ||
         op_name == restore_slice;
}

bool OpCatHelper::IsRetval(TFOp op) {
  static StringRef _retval = "_Retval";
  static StringRef _device_retval = "_DeviceRetval";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _retval || op_name == _device_retval;
}

bool OpCatHelper::IsReverse(TFOp op) {
  static StringRef reverse = "Reverse";
  StringRef op_name = op->getName().stripDialect();
  return op_name == reverse || IsReverseV2(op);
}

bool OpCatHelper::IsReverseV2(TFOp op) {
  static StringRef reverse_v2 = "ReverseV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == reverse_v2;
}

bool OpCatHelper::IsRsqrt(TFOp op) {
  static StringRef rsqrt = "Rsqrt";
  StringRef op_name = op->getName().stripDialect();
  return op_name == rsqrt;
}

bool OpCatHelper::IsRsqrtGrad(TFOp op) {
  static StringRef rsqrt_grad = "RsqrtGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == rsqrt_grad;
}

bool OpCatHelper::IsSelect(TFOp op) {
  static StringRef select = "Select";
  static StringRef select_v2 = "SelectV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == select || op_name == select_v2;
}

bool OpCatHelper::IsSeluGrad(TFOp op) {
  static StringRef selu_grad = "SeluGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == selu_grad;
}

bool OpCatHelper::IsSend(TFOp op) {
  static StringRef _send = "_Send";
  static StringRef _host_send = "_HostSend";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _send || op_name == _host_send;
}

bool OpCatHelper::IsShape(TFOp op) {
  static StringRef shape = "Shape";
  StringRef op_name = op->getName().stripDialect();
  return op_name == shape;
}

bool OpCatHelper::IsShapeN(TFOp op) {
  static StringRef shapeN = "ShapeN";
  StringRef op_name = op->getName().stripDialect();
  return op_name == shapeN;
}

bool OpCatHelper::IsShuffle(TFOp op) {
  static StringRef shuffle = "Shuffle";
  StringRef op_name = op->getName().stripDialect();
  return op_name == shuffle;
}

bool OpCatHelper::IsSigmoid(TFOp op) {
  static StringRef sigmoid = "Sigmoid";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sigmoid;
}

bool OpCatHelper::IsSigmoidGrad(TFOp op) {
  static StringRef sigmoid_grad = "SigmoidGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sigmoid_grad;
}

bool OpCatHelper::IsSize(TFOp op) {
  static StringRef size = "Size";
  StringRef op_name = op->getName().stripDialect();
  return op_name == size;
}

bool OpCatHelper::IsSlice(TFOp op) {
  static StringRef slice = "Slice";
  StringRef op_name = op->getName().stripDialect();
  return op_name == slice;
}

bool OpCatHelper::IsSnapshot(TFOp op) {
  static StringRef snapshot = "Snapshot";
  StringRef op_name = op->getName().stripDialect();
  return op_name == snapshot;
}

bool OpCatHelper::IsSoftmax(TFOp op) {
  static StringRef softmax = "Softmax";
  StringRef op_name = op->getName().stripDialect();
  return op_name == softmax;
}

bool OpCatHelper::IsSoftplusGrad(TFOp op) {
  static StringRef softplus_grad = "SoftplusGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == softplus_grad;
}

bool OpCatHelper::IsSoftsignGrad(TFOp op) {
  static StringRef softsign_grad = "SoftsignGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == softsign_grad;
}

bool OpCatHelper::IsSplit(TFOp op) {
  static StringRef split = "Split";
  StringRef op_name = op->getName().stripDialect();
  return op_name == split;
}

bool OpCatHelper::IsSplitV(TFOp op) {
  static StringRef splitV = "SplitV";
  StringRef op_name = op->getName().stripDialect();
  return op_name == splitV;
}

bool OpCatHelper::IsSqrt(TFOp op) {
  static StringRef sqrt = "Sqrt";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sqrt;
}

bool OpCatHelper::IsSqrtGrad(TFOp op) {
  static StringRef sqrt_grad = "SqrtGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sqrt_grad;
}

bool OpCatHelper::IsSquare(TFOp op) {
  static StringRef square = "Square";
  StringRef op_name = op->getName().stripDialect();
  return op_name == square;
}

bool OpCatHelper::IsSquaredDifference(TFOp op) {
  static StringRef squared_difference = "SquaredDifference";
  StringRef op_name = op->getName().stripDialect();
  return op_name == squared_difference;
}

bool OpCatHelper::IsSqueeze(TFOp op) {
  static StringRef squeeze = "Squeeze";
  StringRef op_name = op->getName().stripDialect();
  return op_name == squeeze;
}

bool OpCatHelper::IsStackOp(TFOp op) {
  static StringRef stack = "Stack";
  static StringRef stack_v2 = "StackV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == stack || op_name == stack_v2;
}
bool OpCatHelper::IsStackCloseOp(TFOp op) {
  static StringRef stack_close = "StackClose";
  static StringRef stack_close_v2 = "StackCloseV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == stack_close || op_name == stack_close_v2;
}
bool OpCatHelper::IsStackPushOp(TFOp op) {
  static StringRef stack_push = "StackPush";
  static StringRef stack_push_v2 = "StackPushV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == stack_push || op_name == stack_push_v2;
}
bool OpCatHelper::IsStackPopOp(TFOp op) {
  static StringRef stack_pop = "StackPop";
  static StringRef stack_pop_v2 = "StackPopV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == stack_pop || op_name == stack_pop_v2;
}

bool OpCatHelper::IsStatefulPartitionedCall(TFOp op) {
  static StringRef stateful_partitioned_call = "StatefulPartitionedCall";
  StringRef op_name = op->getName().stripDialect();
  return op_name == stateful_partitioned_call;
}

bool OpCatHelper::IsStopGradient(TFOp op) {
  static StringRef stop_gradient = "StopGradient";
  static StringRef prevent_gradient = "PreventGradient";
  StringRef op_name = op->getName().stripDialect();
  return op_name == stop_gradient || op_name == prevent_gradient;
}

bool OpCatHelper::IsStridedSlice(TFOp op) {
  static StringRef strided_slice = "StridedSlice";
  StringRef op_name = op->getName().stripDialect();
  return op_name == strided_slice;
}

bool OpCatHelper::IsStridedSliceGrad(TFOp op) {
  static StringRef strided_slice_grad = "StridedSliceGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == strided_slice_grad;
}

bool OpCatHelper::IsStringToHashBucketFast(TFOp op) {
  static StringRef string_to_hashbucket_fast = "StringToHashBucketFast";
  StringRef op_name = op->getName().stripDialect();
  return op_name == string_to_hashbucket_fast;
}

bool OpCatHelper::IsSub(TFOp op) {
  static StringRef sub = "Sub";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sub;
}

bool OpCatHelper::IsSum(TFOp op) {
  static StringRef sum = "Sum";
  StringRef op_name = op->getName().stripDialect();
  return op_name == sum;
}

bool OpCatHelper::IsSwitch(TFOp op) {
  static StringRef _switch = "Switch";
  static StringRef _switch_n = "_SwitchN";
  static StringRef ref_switch = "RefSwitch";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _switch || op_name == _switch_n || op_name == ref_switch;
}

bool OpCatHelper::IsSymbolicGradient(TFOp op) {
  static StringRef symbolic_gradient = "SymbolicGradient";
  StringRef op_name = op->getName().stripDialect();
  return op_name == symbolic_gradient;
}

bool OpCatHelper::IsTanh(TFOp op) {
  static StringRef tanh = "Tanh";
  StringRef op_name = op->getName().stripDialect();
  return op_name == tanh;
}

bool OpCatHelper::IsTanhGrad(TFOp op) {
  static StringRef tanh_grad = "TanhGrad";
  StringRef op_name = op->getName().stripDialect();
  return op_name == tanh_grad;
}

bool OpCatHelper::IsTile(TFOp op) {
  static StringRef tile = "Tile";
  StringRef op_name = op->getName().stripDialect();
  return op_name == tile;
}

bool OpCatHelper::IsTranspose(TFOp op) {
  static StringRef transpose = "Transpose";
  StringRef op_name = op->getName().stripDialect();
  return op_name == transpose;
}

bool OpCatHelper::IsTruncateDiv(TFOp op) {
  static StringRef truncate_div = "TruncateDiv";
  StringRef op_name = op->getName().stripDialect();
  return op_name == truncate_div;
}

bool OpCatHelper::IsTruncateMod(TFOp op) {
  static StringRef truncate_mod = "TruncateMod";
  StringRef op_name = op->getName().stripDialect();
  return op_name == truncate_mod;
}

bool OpCatHelper::IsUnique(TFOp op) {
  static StringRef unique = "Unique";
  static StringRef unique_v2 = "UniqueV2";
  StringRef op_name = op->getName().stripDialect();
  return op_name == unique || op_name == unique_v2;
}

bool OpCatHelper::IsUnpack(TFOp op) {
  static StringRef unpack = "Unpack";
  StringRef op_name = op->getName().stripDialect();
  return op_name == unpack;
}

bool OpCatHelper::IsVariable(TFOp op) {
  static StringRef variable = "Variable";
  static StringRef variable_v2 = "VariableV2";
  static StringRef auto_reload_variable = "AutoReloadVariable";
  static StringRef var_handle_op = "VarHandleOp";
  static StringRef _var_handles_op = "_VarHandlesOp";
  StringRef op_name = op->getName().stripDialect();
  return op_name == variable || op_name == variable_v2 ||
         op_name == auto_reload_variable || op_name == var_handle_op ||
         op_name == _var_handles_op || IsReadVariableOp(op) ||
         IsReadVariablesOp(op);
}

bool OpCatHelper::IsWhile(TFOp op) {
  static StringRef _while = "While";
  static StringRef stateless_while = "StatelessWhile";
  StringRef op_name = op->getName().stripDialect();
  return op_name == _while || op_name == stateless_while;
}

bool OpCatHelper::IsXdivy(TFOp op) {
  static StringRef xdivy = "Xdivy";
  StringRef op_name = op->getName().stripDialect();
  return op_name == xdivy;
}

bool OpCatHelper::IsZerosLike(TFOp op) {
  static StringRef zeros_like = "ZerosLike";
  StringRef op_name = op->getName().stripDialect();
  return op_name == zeros_like;
}

bool OpCatHelper::IsZeta(TFOp op) {
  static StringRef zeta = "Zeta";
  StringRef op_name = op->getName().stripDialect();
  return op_name == zeta;
}

bool OpCatHelper::IsAggregate(TFOp op) {
  if (IsAdd(op)) {
    // TODO(chiahungduan): type != tensorflow::DT_INVALID
    return !op->getAttrOfType<TypeAttr>("T").getValue().isa<StringType>();
  }
  const tensorflow::OpDef *op_def = nullptr;
  tensorflow::Status status = tensorflow::OpRegistry::Global()->LookUpOpDef(
      op->getName().stripDialect().data(), &op_def);
  return status.ok() && op_def->is_aggregate();
}

bool OpCatHelper::IsCommutative(TFOp op) {
  if (IsAdd(op)) {
    // TODO(chiahungduan): type != tensorflow::DT_INVALID
    return !op->getAttrOfType<TypeAttr>("T").getValue().isa<StringType>();
  }
  const tensorflow::OpDef *op_def = nullptr;
  tensorflow::Status status = tensorflow::OpRegistry::Global()->LookUpOpDef(
      op->getName().stripDialect().data(), &op_def);
  return status.ok() && op_def->is_commutative();
}

bool OpCatHelper::IsOnes(TFOp op) {
  if (IsOnesLike(op)) return true;
  if (IsZerosLike(op)) return false;

  if (IsFill(op)) {
    TFOp value_op = op->getOperand(1).getDefiningOp();
    return !value_op && IsOnes(value_op);
  }

  if (!IsConstant(op)) return false;

  SplatElementsAttr const_attr = op->getAttrOfType<SplatElementsAttr>("value");
  if (!const_attr) return false;

  return SplatElementsAttrHasValue(const_attr, 1);
}

bool OpCatHelper::IsZeros(TFOp op) {
  if (IsOnesLike(op)) return false;
  if (IsZerosLike(op)) return true;

  if (IsFill(op)) {
    TFOp value_op = op->getOperand(1).getDefiningOp();
    return !value_op && IsZeros(value_op);
  }

  if (!IsConstant(op)) return false;

  SplatElementsAttr const_attr = op->getAttrOfType<SplatElementsAttr>("value");
  if (!const_attr) return false;

  return SplatElementsAttrHasValue(const_attr, 0);
}

bool OpCatHelper::IsPersistent(TFOp op) {
  return IsConstant(op) || IsVariable(op) || IsHostConstant(op);
}

bool OpCatHelper::IsDataset(TFOp op) {
  static StringRef iterator_get_next = "IteratorGetNext";
  static StringRef iterator_get_next_sync = "IteratorGetNextSync";
  static StringRef dataset_to_single_element = "DatasetToSingleElement";
  static StringRef reduce_data_set = "ReduceDataset";
  StringRef op_name = op->getName().stripDialect();
  // See `GetNodeClassForOp` in core/graph/graph.cc.
  return op_name == iterator_get_next || op_name == iterator_get_next_sync ||
         op_name == dataset_to_single_element || op_name == reduce_data_set;
}

}  // namespace tfg
}  // namespace mlir
