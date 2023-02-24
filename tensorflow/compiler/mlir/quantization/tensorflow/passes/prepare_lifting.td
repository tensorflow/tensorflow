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

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

// Converts arith.constant ops from freezing passes back to tf.Const ops.
def ConvertArithConstToTfConst : Pat<
  (Arith_ConstantOp:$res DenseElementsAttr:$value),
  (TF_ConstOp $value),
  [(AnyStaticShapeTensor $res)]>;

// Converts tf.FusedBatchNormV3 into a sequence of more primitive arithmetic
// operations. Specifically, performs the following calculation:
//
//   (x - mean) * scale / sqrt(variance + epsilon) + offset
//
// Let multiplier = scale / sqrt(variance + epsilon),
// to compute
//   (x - mean) * scale / sqrt(variance + epsilon) + offset,
// is then to compute
//   (x * multiplier) + (offset - mean * multiplier).
//
// TODO(b/228916181): There is a known issue with this DDR rule that it doesn't
// take into account broadcasting conditions. If the issue needs to be handled,
// see tensorflow/compiler/mlir/lite/transforms/prepare_tf.cc
def FoldFusedBatchNormV3: Pattern<
    (TF_FusedBatchNormV3Op:$root
        $x, $scale, $offset, $mean, $variance,
        F32Attr:$epsilon, $exponential_avg_factor,
        $data_format, IsFalseBoolAttr:$is_training),
    [(TF_AddV2Op
        (TF_MulOp
            $x,
            (TF_MulOp:$multiplier
                $scale,
                (TF_RsqrtOp
                    (TF_AddV2Op $variance,
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

class HasEqualElementSize<list<int> shape_1, list<int> shape_2> : Constraint<
  CPred<"quant::HasEqualElementSize($0, $1,"
  "llvm::ArrayRef<int>({" # !interleave(shape_1, ", ") # "}),"
  "llvm::ArrayRef<int>({" # !interleave(shape_2, ", ") # "}))">,
  "Checks if the given dimensions contain the same number of elements.">;

def HasEqualShape : Constraint<CPred<
  "$0.getType().cast<ShapedType>().hasRank() && "
  "$1.getType().cast<ShapedType>().hasRank() && "
  "$0.getType().cast<ShapedType>().getShape() == $1.getType().cast<ShapedType>().getShape()">,
  "Checks if the shapes of tensors are same.">;

// Make the 1D value $0 broadcastable with the shape of $1.
def MakeOneDimValueBroadcastable : NativeCodeCall<
  "MakeOneDimValueBroadcastable($_builder, $_loc, $0, $1.getType().cast<ShapedType>())">;

// Match convolution op with "NHWC" data format or matmul op.
def SupportedAffineOpMatcher : NativeCodeCall<
  "MatchSupportedAffineOp($_self, $0, $1, $2)">;

// Checks if a value can be symetrically quantized.
def CanBeSymmetricallyQuantized : Constraint<CPred<"CanBeSymmetricallyQuantized($0)">>;

// Multiplies the value followed by a FakeQuant op and adjusts its params.
def MultiplyFakeQuantValue : NativeCodeCall<
  "MultiplyFakeQuantValue($_builder, $_loc, $0...)">;

// Convert AddV2Op following an AffineOp to BiasAddOp.
// For Conv3D, even though the Conv3D op has "NDHWC" data format, the BiasAdd
// will still has the data format of "NHWC".
def ConvertAddToBiasAdd : Pat<
  (TF_AddV2Op
    (SupportedAffineOpMatcher $conv_out, $input, $weight),
    (TF_ConstOp:$add_rhs IsFloatElementsAttr:$add_rhs_value)),
  (TF_BiasAddOp $conv_out, $add_rhs, (CreateStringAttr<"NHWC">)),
  [(HasRankOf<1> $add_rhs_value),
   (HasEqualElementSize<[-1], [0]> $conv_out, $add_rhs)]>;

// Fuse consecutive BiasAddOp and an AddV2Op.
def FuseBiasAndAddV2 : Pat<
  (TF_AddV2Op
    (TF_BiasAddOp:$bias_add
      $conv_out,
      (TF_ConstOp:$bias IsFloatElementsAttr:$bias_value), $data_format),
    (TF_ConstOp:$add_rhs IsFloatElementsAttr:$add_rhs_value)),
  (TF_BiasAddOp
    $conv_out, (TF_AddV2Op $bias, $add_rhs), $data_format),
  [(HasOneUse $bias_add),
   (HasEqualShape $bias_value, $add_rhs_value)]>;

// Fuse AffineOp followed by an MulOp patterns.
def FuseAffineOpAndMul : Pat<
  (TF_MulOp
    (SupportedAffineOpMatcher $conv_out, $input, $weight),
    (TF_ConstOp:$mul_rhs IsFloatElementsAttr:$mul_rhs_value)),
  (CloneOpWithReplacedOperands
        (GetDefiningOp $conv_out),
        $input,
        (MultiplyFakeQuantValue $weight,
          (MakeOneDimValueBroadcastable $mul_rhs, $weight))),
  [(HasOneUse $conv_out),
   (HasRankOf<1> $mul_rhs_value),
   (HasStaticShapeConstraint $weight),
   (CanBeSymmetricallyQuantized $weight),
   (HasEqualElementSize<[-1], [0]> $conv_out, $mul_rhs)]>;

// Fuse AffineOp followed by an BiasAddOp and an MulOp patterns.
def FuseAffineOpWithBiasAddAndMul : Pat<
  (TF_MulOp
    (TF_BiasAddOp:$bias_add
      (SupportedAffineOpMatcher $conv_out, $input, $weight),
      $bias, $data_format),
    (TF_ConstOp:$mul_rhs IsFloatElementsAttr:$mul_rhs_value)),
  (TF_BiasAddOp
    (CloneOpWithReplacedOperands
      (GetDefiningOp $conv_out),
      $input,
      (MultiplyFakeQuantValue $weight,
        (MakeOneDimValueBroadcastable $mul_rhs, $weight))),
    (MultiplyFakeQuantValue $bias, $mul_rhs), $data_format),
  [(HasOneUse $conv_out),
   (HasOneUse $bias_add),
   (HasRankOf<1> $mul_rhs_value),
   (HasStaticShapeConstraint $weight),
   (CanBeSymmetricallyQuantized $weight),
   (CanBeSymmetricallyQuantized $bias),
   (HasEqualShape $bias, $mul_rhs_value)]>;
