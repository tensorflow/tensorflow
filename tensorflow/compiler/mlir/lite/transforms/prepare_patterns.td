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

include "tensorflow/compiler/mlir/tensorflow/transforms/optimize.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"

def FalseBoolAttr : AttrConstraint<CPred<"!$_self.getValue()">>;

def DenseElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<DenseElementsAttr>()">,
  "non-opaque constant tensor">;

def CreateGatherNdOp : NativeCodeCall<
    "$_builder.create<TF::GatherNdOp>($0.getLoc(), $0.getType(), $1, $2)">;

def CreateTFCastOpI32 : NativeCodeCall<
    "CreateTFCastOpI32(&$_builder, $_loc, $0, $1)">;

// Convert to std constant for statically shaped, non-opaque constants.
def ConvertTfConst : Pat<(TF_ConstOp:$res DenseElementsAttr:$value), (Arith_ConstantOp $value),
          [(AnyStaticShapeTensor $res)]>;

// Convert the legacy BatchNormWithGlobalNormalization to other ops.
def ConvertBatchNormWithGlobalNormalization_1 : Pat<
  (TF_BatchNormWithGlobalNormalizationOp
    $t, $m, $v, $beta, $gamma, F32Attr:$variance_epsilon, ConstBoolAttrFalse:$scale_after_normalization),
  (TF_AddOp
    (TF_MulOp $t, (TF_RsqrtOp:$rsqrt (TF_AddOp $v, (TF_ConstOp $variance_epsilon)))),
    (TF_SubOp $beta, (TF_MulOp $m, $rsqrt)))>;
def ConvertBatchNormWithGlobalNormalization_2 : Pat<
  (TF_BatchNormWithGlobalNormalizationOp
    $t, $m, $v, $beta, $gamma, F32Attr:$variance_epsilon, ConstBoolAttrTrue:$scale_after_normalization),
  (TF_AddOp
    (TF_MulOp $t, (TF_MulOp:$mul (TF_RsqrtOp (TF_AddOp $v, (TF_ConstOp $variance_epsilon))), $gamma)),
    (TF_SubOp $beta, (TF_MulOp $m, $mul)))>;

class TFi32<int v> : ConstantAttr<I32ElementsAttr, !cast<string>(v)>;

// Matmul without transpose on b to matmul with explicit transpose op and
// transposed b.
def ConvertMatmulWithoutTransposeToWithTranspose :
      Pat<(TF_MatMulOp $a, $b, ConstBoolAttrFalse:$at, ConstBoolAttrFalse, $grad_a, $grad_b),
          (TF_MatMulOp $a, (TF_TransposeOp $b, (TF_SubOp (TF_RangeOp
             /*start=*/(TF_RankOp $b),
             /*limit=*/(TF_ConstOp TFi32<0>),
             /*delta=*/(TF_ConstOp TFi32<-1>)), (TF_ConstOp TFi32<1>))),
           $at, ConstBoolAttrTrue, $grad_a, $grad_b)>;

// Matmul with transpose on a to matmul with explicit transpose op and a not
// transposed.
def ConvertMatmulWithTranspose : Pat<(TF_MatMulOp $a, $b, ConstBoolAttrTrue, $bt, $grad_a, $grad_b),
          (TF_MatMulOp (TF_TransposeOp $a, (TF_SubOp (TF_RangeOp
             /*start=*/(TF_RankOp $a),
             /*limit=*/(TF_ConstOp TFi32<0>),
             /*delta=*/(TF_ConstOp TFi32<-1>)), (TF_ConstOp TFi32<1>))), $b,
           ConstBoolAttrFalse, $bt, $grad_a, $grad_b)>;

// Partially supported in TFLite, treated as passthrough IdentityOp
def ConvertCheckNumerics : Pat<(TF_CheckNumericsOp $arg, $msg), (TF_IdentityOp $arg)>;
def ConvertSnapshot : Pat<(TF_SnapshotOp $arg), (TF_IdentityOp $arg)>;
def ConvertStopGradient : Pat<(TF_StopGradientOp $arg), (TF_IdentityOp $arg)>;
def ConvertPlaceholderWithDefault : Pat<(TF_PlaceholderWithDefaultOp $arg), (TF_IdentityOp $arg)>;

//===----------------------------------------------------------------------===//
// Op removal patterns.
//===----------------------------------------------------------------------===//
def RemovePreventGradient : Pat<(TF_PreventGradientOp $op, $msg), (replaceWithValue $op)>;
def RemoveXlaSharding : Pat<(TF_XlaShardingOp $a, $b, $c), (replaceWithValue $a)>;
def RemoveIdentityN : Pat<(TF_IdentityNOp $arg), (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// Op quantization pass-through patterns.
//===----------------------------------------------------------------------===//
// Casts result type of $1 to a quantized type by using the quantization
// parameters from the type in $0.
class UpdateShapeWithAxis<int i> : NativeCodeCall<
  "quant::CastQuantizedTypeAttrFromExpressedType($_builder, $0, $1.getType(), " # i # ")">;

class CanUpdateShapeWithAxis<int i> : Constraint<
  CPred<"quant::CastQuantizedTypeAttrFromExpressedType($_builder, $0, $1.getType(), " # i # ")">>;

class UsedBy<string op> : Constraint<
  CPred<"llvm::isa<mlir::TFL::" # op # "Op>(*$0.getUsers().begin())">>;

// When the op is passing-through, the output types of the quantized ops need
// to be updated as well. Since the quantize op manages its own type by the
// "qtype" attribute, we should update the type shape in this attribute.
def ReorderTransposeDequantQuant :
      Pat<(TF_TransposeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $perm),
          (TFL_DequantizeOp (TFL_QuantizeOp
                                (TF_TransposeOp $input, $perm),
                                (UpdateShapeWithAxis<-1> $qtype, $old_value))),
          [(CanUpdateShapeWithAxis<-1> $qtype, $old_value)]>;

def ReorderReshapeDequantQuant : Pat<(TF_ReshapeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $shape),
          (TFL_DequantizeOp
              (TFL_QuantizeOp (TF_ReshapeOp $input, $shape),
              (UpdateShapeWithAxis<-1> $qtype, $old_value))),
          [(CanUpdateShapeWithAxis<-1> $qtype, $old_value)]>;

// The axis is set to 0 because the transpose is from the legalization of
// tf.conv2d and the new channel axis is the first dimension.
def ReorderTransposeDequantQuantUsedByConv :
      Pat<(TF_TransposeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $perm),
          (TFL_DequantizeOp (TFL_QuantizeOp
                                (TF_TransposeOp $input, $perm),
                                (UpdateShapeWithAxis<0> $qtype, $old_value))),
  [(UsedBy<"Conv2D"> $old_value),
   (CanUpdateShapeWithAxis<0> $qtype, $old_value)],
  [], (addBenefit 10)>;

// The axis is set to 3, because this reshape is from the legalization of
// tf.depthwiseconvnative and the new channel axis is the last dimension.
def ReorderReshapeDequantQuantUsedByDepthwiseConv :
      Pat<(TF_ReshapeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $shape),
          (TFL_DequantizeOp
              (TFL_QuantizeOp (TF_ReshapeOp $input, $shape),
              (UpdateShapeWithAxis<3> $qtype, $old_value))),
  [(UsedBy<"DepthwiseConv2D"> $old_value),
   (CanUpdateShapeWithAxis<3> $qtype, $old_value)],
  [], (addBenefit 10)>;

// The Rank op produces result which is independent with the quantization
// parameters of the input, so we can remove the quantization ops.
def OptimizeAwayRankDequantQuant :
      Pat<(TF_RankOp (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype))),
          (TF_RankOp $input)>;

// `k` is expected to be 0, other values are not supported currently.
def ConvertMatrixSetDiagV2ToMatrixSetDiag :
      Pat<(TF_MatrixSetDiagV2Op $input, $diagonal,
            (Arith_ConstantOp ConstantAttr<I32ElementsAttr, "{0}">)),
          (TF_MatrixSetDiagOp $input, $diagonal)>;

// `align` attribute can be ignored because we only support converting
// `MatrixSetDiagV3` to `MatrixSetDiag` with default `k` inputs.
def ConvertMatrixSetDiagV3ToMatrixSetDiag :
      Pat<(TF_MatrixSetDiagV3Op $input, $diagonal,
            (Arith_ConstantOp ConstantAttr<I32ElementsAttr, "{0}">),
            $align),
          (TF_MatrixSetDiagOp $input, $diagonal)>;

//===----------------------------------------------------------------------===//
// TensorScatterAdd op patterns.
//===----------------------------------------------------------------------===//

def LowerTensorScatterAdd: Pat<
  (TF_TensorScatterAddOp $input, $indices, $updates),
  (TF_TensorScatterUpdateOp $input, $indices,
     (TF_AddOp $updates, (CreateGatherNdOp $updates, $input, $indices)))>;

//===----------------------------------------------------------------------===//
// AddV2 op patterns.
//===----------------------------------------------------------------------===//

// This is mostly safe, we can use i64 but it will break Darwinn case.
// TODO(b/199054442): Change to i64 when Darwinn supports i64.
def LowerUInt32AddV2 : Pat<
  (TF_AddV2Op $lhs, $rhs),
  (TF_CastOp
    (TF_AddV2Op
      (CreateTFCastOpI32 $lhs, /*truncate=*/ConstBoolAttrFalse),
      (CreateTFCastOpI32 $rhs, /*truncate=*/ConstBoolAttrFalse)),
     /*truncate=*/ConstBoolAttrFalse),
  [(TensorOf<[TF_Uint32]> $lhs), (TensorOf<[TF_Uint32]> $rhs)]>;

//===----------------------------------------------------------------------===//
// Fill op patterns.
//===----------------------------------------------------------------------===//

def RemoveRedundantShapeOp :
  Pat<(TF_ShapeOp (TF_FillOp $shape, $fill)), (replaceWithValue $shape)>;
