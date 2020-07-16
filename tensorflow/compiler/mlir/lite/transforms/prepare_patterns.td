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

def NonOpaqueElementsAttr : ElementsAttrBase<
  CPred<"!$_self.isa<OpaqueElementsAttr>()">,
  "non-opaque constant tensor">;

// Convert to std constant for statically shaped, non-opaque constants.
def : Pat<(TF_ConstOp:$res NonOpaqueElementsAttr:$value), (ConstantOp $value),
          [(AnyStaticShapeTensor $res)]>;

// Convert the legacy BatchNormWithGlobalNormalization to other ops.
def : Pat<
  (TF_BatchNormWithGlobalNormalizationOp
    $t, $m, $v, $beta, $gamma, F32Attr:$variance_epsilon, ConstBoolAttrFalse:$scale_after_normalization),
  (TF_AddOp
    (TF_MulOp $t, (TF_RsqrtOp:$rsqrt (TF_AddOp $v, (TF_ConstOp $variance_epsilon)))),
    (TF_SubOp $beta, (TF_MulOp $m, $rsqrt)))>;
def : Pat<
  (TF_BatchNormWithGlobalNormalizationOp
    $t, $m, $v, $beta, $gamma, F32Attr:$variance_epsilon, ConstBoolAttrTrue:$scale_after_normalization),
  (TF_AddOp
    (TF_MulOp $t, (TF_MulOp:$mul (TF_RsqrtOp (TF_AddOp $v, (TF_ConstOp $variance_epsilon))), $gamma)),
    (TF_SubOp $beta, (TF_MulOp $m, $mul)))>;

// Converts tf.FusedBatchNorm & tf.FusedBatchNormV3 into a sequence of more primitive arithmetic
// operations. Specifically, performs the following calculation:
//
//   (x - mean) * scale / sqrt(variance + epsilon) + offset
//
// Let multiplier = scale / sqrt(variance + epsilon),
// to compute
//   (x - mean) * scale / sqrt(variance + epsilon) + offset,
// is then to compute
//   (x * multiplier) + (offset - mean * multiplier).
def : Pattern<
    (TF_FusedBatchNormOp:$root
        $x, $scale, $offset, $mean, $variance,
        F32Attr:$epsilon, $exponential_avg_factor,
        $data_format, FalseBoolAttr:$is_training),
    [(TF_AddOp
        (TF_MulOp
            $x,
            (TF_MulOp:$multiplier
                $scale,
                (TF_RsqrtOp
                    (TF_AddOp $variance,
                              (TF_ConstOp $epsilon))))),
        (TF_SubOp $offset, (TF_MulOp $mean, $multiplier))),
     // We already guaranteed that the last four results has no use so it does
     // not matter what value we provide here for replacement.
     /*batch_mean=*/(replaceWithValue $x),
     /*batch_variance=*/(replaceWithValue $x),
     /*reserve_space_1=*/(replaceWithValue $x),
     /*reserve_space_2=*/(replaceWithValue $x)],
    [(HasNoUseOf:$root__1), (HasNoUseOf:$root__2),
     (HasNoUseOf:$root__3), (HasNoUseOf:$root__4)]>;

def : Pattern<
    (TF_FusedBatchNormV3Op:$root
        $x, $scale, $offset, $mean, $variance,
        F32Attr:$epsilon, $exponential_avg_factor,
        $data_format, FalseBoolAttr:$is_training),
    [(TF_AddOp
        (TF_MulOp
            $x,
            (TF_MulOp:$multiplier
                $scale,
                (TF_RsqrtOp
                    (TF_AddOp $variance,
                              (TF_ConstOp $epsilon))))),
        (TF_SubOp $offset, (TF_MulOp $mean, $multiplier))),
     // We already guaranteed that the last five results have no use so it does
     // not matter what value we provide here for replacement.
     /*batch_mean=*/(replaceWithValue $x),
     /*batch_variance=*/(replaceWithValue $x),
     /*reserve_space_1=*/(replaceWithValue $x),
     /*reserve_space_2=*/(replaceWithValue $x),
     /*reserve_space_3=*/(replaceWithValue $x)],
    [(HasNoUseOf:$root__1), (HasNoUseOf:$root__2),
     (HasNoUseOf:$root__3), (HasNoUseOf:$root__4),
     (HasNoUseOf:$root__5)]>;

class TFi32<int v> : ConstantAttr<I32ElementsAttr, !cast<string>(v)>;

// Matmul without transpose on b to matmul with explicit transpose op and
// transposed b.
def : Pat<(TF_MatMulOp $a, $b, ConstBoolAttrFalse:$at, ConstBoolAttrFalse),
          (TF_MatMulOp $a, (TF_TransposeOp $b, (TF_SubOp (TF_RangeOp
             /*start=*/(TF_RankOp $b),
             /*limit=*/(TF_ConstOp TFi32<0>),
             /*delta=*/(TF_ConstOp TFi32<-1>)), (TF_ConstOp TFi32<1>))),
           $at, ConstBoolAttrTrue)>;

// Matmul with transpose on a to matmul with explicit transpose op and a not
// transposed.
def : Pat<(TF_MatMulOp $a, $b, ConstBoolAttrTrue, $bt),
          (TF_MatMulOp (TF_TransposeOp $a, (TF_SubOp (TF_RangeOp
             /*start=*/(TF_RankOp $a),
             /*limit=*/(TF_ConstOp TFi32<0>),
             /*delta=*/(TF_ConstOp TFi32<-1>)), (TF_ConstOp TFi32<1>))), $b,
           ConstBoolAttrFalse, $bt)>;

// Partially supported in TFLite, treated as passthrough IdentityOp
def : Pat<(TF_CheckNumericsOp $arg, $msg), (TF_IdentityOp $arg)>;
def : Pat<(TF_SnapshotOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_StopGradientOp $arg), (TF_IdentityOp $arg)>;
def : Pat<(TF_PlaceholderWithDefaultOp $arg), (TF_IdentityOp $arg)>;

//===----------------------------------------------------------------------===//
// Op removal patterns.
//===----------------------------------------------------------------------===//
def : Pat<(TF_IdentityOp $arg), (replaceWithValue $arg)>;
def : Pat<(TF_IdentityNOp $arg), (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// Op quantization pass-through patterns.
//===----------------------------------------------------------------------===//
// TODO(fengliuai): Implement similar rule in the QuantizePass if the constant
// folding hook of tfl.transpose and tfl.reshape are implemented.
def : Pat<(TF_TransposeOp
              (TF_FakeQuantWithMinMaxVarsOp
                  $input, $min, $max, $num_bits, $narrow_range),
              $perm),
          (TF_FakeQuantWithMinMaxVarsOp (TF_TransposeOp $input, $perm),
              $min, $max, $num_bits, $narrow_range)>;

def : Pat<(TF_ReshapeOp
              (TF_FakeQuantWithMinMaxVarsOp
                  $input, $min, $max, $num_bits, $narrow_range),
              $shape),
          (TF_FakeQuantWithMinMaxVarsOp (TF_ReshapeOp $input, $shape),
              $min, $max, $num_bits, $narrow_range)>;

// Casts result type of $1 to a quantized type by using the quantization
// parameters from the type in $0.
class UpdateShapeWithAxis<int i> : NativeCodeCall<
  "quant::CastQuantizedTypeAttrFromExpressedType($_builder, $0, $1.getType(), " # i # ")">;

class UsedBy<string op> : Constraint<
  CPred<"llvm::isa<mlir::TFL::" # op # "Op>(*$0.getUsers().begin())">>;

// When the op is passing-through, the output types of the quantized ops need
// to be updated as well. Since the quantize op manages its own type by the
// "qtype" attribute, we should update the type shape in this attribute.
def : Pat<(TF_TransposeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $perm),
          (TFL_DequantizeOp (TFL_QuantizeOp
                                (TF_TransposeOp $input, $perm),
                                (UpdateShapeWithAxis<-1> $qtype, $old_value)))>;
def : Pat<(TF_ReshapeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $shape),
          (TFL_DequantizeOp
              (TFL_QuantizeOp (TF_ReshapeOp $input, $shape),
              (UpdateShapeWithAxis<-1> $qtype, $old_value)))>;

// The axis is set to 0 because the transpose is from the legalization of
// tf.conv2d and the new channel axis is the first dimension.
def : Pat<(TF_TransposeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $perm),
          (TFL_DequantizeOp (TFL_QuantizeOp
                                (TF_TransposeOp $input, $perm),
                                (UpdateShapeWithAxis<0> $qtype, $old_value))),
  [(UsedBy<"Conv2D"> $old_value)], (addBenefit 10)>;
// The axis is set to 3, because this reshape is from the legalization of
// tf.depthwiseconvnative and the new channel axis is the last dimension.
def : Pat<(TF_ReshapeOp:$old_value
              (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype)), $shape),
          (TFL_DequantizeOp
              (TFL_QuantizeOp (TF_ReshapeOp $input, $shape),
              (UpdateShapeWithAxis<3> $qtype, $old_value))),
  [(UsedBy<"DepthwiseConv2D"> $old_value)], (addBenefit 10)>;

// The Rank op produces result which is independent with the quantization
// parameters of the input, so we can remove the quantization ops.
def : Pat<(TF_RankOp (TFL_DequantizeOp (TFL_QuantizeOp $input, $qtype))),
          (TF_RankOp $input)>;

// `k` is expected to be 0, other values are not supported currently.
def : Pat<(TF_MatrixSetDiagV2Op $input, $diagonal,
            (ConstantOp ConstantAttr<I32ElementsAttr, "{0}">)),
          (TF_MatrixSetDiagOp $input, $diagonal)>;

// `align` attribute can be ignored because we only support converting
// `MatrixSetDiagV3` to `MatrixSetDiag` with default `k` inputs.
def : Pat<(TF_MatrixSetDiagV3Op $input, $diagonal,
            (ConstantOp ConstantAttr<I32ElementsAttr, "{0}">),
            $align),
          (TF_MatrixSetDiagOp $input, $diagonal)>;

