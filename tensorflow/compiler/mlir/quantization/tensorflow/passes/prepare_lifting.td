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

class HasRank<int n> : Constraint<
  CPred<"$0.getType().cast<ShapedType>().hasRank() && "
        "$0.getType().cast<ShapedType>().getRank() == " # n>,
  "Checks if the value has rank of 'n'.">;

class HasEqualElementSize<list<int> shape_1, list<int> shape_2> : Constraint<
  CPred<"quant::HasEqualElementSize($0, $1,"
  "llvm::ArrayRef<unsigned>({" # !interleave(shape_1, ", ") # "}),"
  "llvm::ArrayRef<unsigned>({" # !interleave(shape_2, ", ") # "}))">,
  "Checks if the given dimensions contain the same number of elements.">;

def HasEqualShape : Constraint<CPred<
  "$0.getType().cast<ShapedType>().hasRank() && "
  "$1.getType().cast<ShapedType>().hasRank() && "
  "$0.getType().cast<ShapedType>().getShape() == $1.getType().cast<ShapedType>().getShape()">,
  "Checks if the shapes of tensors are same.">;

def Expand1DTo4DForConv2D : NativeCodeCall<
  "$0.cast<DenseElementsAttr>().reshape("
  "RankedTensorType::get({1,1,1,$0.getType().cast<ShapedType>().getNumElements()},"
  "getElementTypeOrSelf($0.getType())))">;

def Expand1DTo4DForDepthwiseConv2D : NativeCodeCall<
  "$0.cast<DenseElementsAttr>().reshape("
  "RankedTensorType::get({1,1,$1.getType().cast<ShapedType>().getDimSize(2),$1.getType().cast<ShapedType>().getDimSize(3)},"
  "getElementTypeOrSelf($0.getType())))">;

def CreateUnrankedTensorTypeWithElementType : NativeCodeCall<
    "UnrankedTensorType::get(getElementTypeOrSelf($0.getType()))">;

// Matching AffineOp followed by an AddOp patterns.
def MatchConv2dAndAddPattern : Pat<
  (TF_AddOp (TF_Conv2DOp:$conv_out $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
            (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_BiasAddOp (TF_Conv2DOp $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, $data_format, $dilations), (TF_ConstOp $value), $data_format),
  [(HasOneUse $conv_out), (HasRank<1> $value), (HasRank<4> $filter),
   (HasEqualElementSize<[3],[0]> $filter, $value)]>;

def MatchDepthwiseConv2dAndAddPattern : Pat<
  (TF_AddOp (TF_DepthwiseConv2dNativeOp:$conv_out $input, $filter, $strides, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
         (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_BiasAddOp (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
         $explicit_padding, $data_format, $dilations,
         (returnType (CreateUnrankedTensorTypeWithElementType $input))),
         (TF_ConstOp $value), $data_format),
  [(HasOneUse $conv_out), (HasRank<1> $value), (HasRank<4> $filter),
   (HasEqualElementSize<[2,3],[0]> $filter, $value)]>;

// Fusing AffineOp followed by an MulOp patterns.
def FuseConv2dAndMul : Pat<
  (TF_MulOp (TF_Conv2DOp:$conv_out $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
            (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_Conv2DOp $input, (TF_MulOp $filter,
        (TF_ConstOp (Expand1DTo4DForConv2D $value))), $strides, $use_cudnn, $padding,
         $explicit_padding, $data_format, $dilations),
  [(HasOneUse $conv_out), (HasRank<1> $value), (HasRank<4> $filter),
   (HasEqualElementSize<[3],[0]> $filter, $value)]>;

def FuseDepthwiseConv2dAndMul : Pat<
  (TF_MulOp (TF_DepthwiseConv2dNativeOp:$conv_out $input, $filter, $strides, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
            (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_DepthwiseConv2dNativeOp $input, (TF_MulOp $filter,
        (TF_ConstOp (Expand1DTo4DForDepthwiseConv2D $value, $filter))), $strides, $padding,
         $explicit_padding, $data_format, $dilations),
  [(HasOneUse $conv_out), (HasRank<1> $value), (HasRank<4> $filter),
   (HasEqualElementSize<[2,3],[0]> $filter, $value)]>;

// Fusing AffineOp followed by an BiasAddOp and an AddOp patterns.
def FuseConv2dWithBiasAndAdd : Pat<
  (TF_AddOp
   (TF_BiasAddOp:$biasadd_out
    (TF_Conv2DOp:$conv_out $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
    $bias, IsDataFormatNHWC:$bias_data_format),
    (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_BiasAddOp
   (TF_Conv2DOp $input, $filter, $strides, $use_cudnn, $padding,
    $explicit_padding, $data_format, $dilations),
   (TF_AddOp $bias, (TF_ConstOp $value)), $bias_data_format),
  [(HasOneUse $conv_out), (HasOneUse $biasadd_out), (HasRank<1> $value),
   (HasEqualShape $value, $bias)]>;

def FuseDepthwiseConv2dWithBiasAndAdd : Pat<
  (TF_AddOp
   (TF_BiasAddOp:$biasadd_out
    (TF_DepthwiseConv2dNativeOp:$conv_out $input, $filter, $strides, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
    $bias, IsDataFormatNHWC:$bias_data_format),
    (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_BiasAddOp
   (TF_DepthwiseConv2dNativeOp $input, $filter, $strides, $padding,
    $explicit_padding, $data_format, $dilations,
    (returnType (CreateUnrankedTensorTypeWithElementType $input))),
   (TF_AddOp $bias, (TF_ConstOp $value)), $bias_data_format),
  [(HasOneUse $conv_out), (HasOneUse $biasadd_out), (HasRank<1> $value),
   (HasEqualShape $value, $bias)]>;

// Fusing AffineOp followed by an BiasAddOp and an MulOp patterns.
def FuseConv2dWithBiasAndMul : Pat<
  (TF_MulOp
   (TF_BiasAddOp:$biasadd_out
    (TF_Conv2DOp:$conv_out $input, $filter, $strides, $use_cudnn, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
    $bias, IsDataFormatNHWC:$bias_data_format),
    (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_BiasAddOp (TF_Conv2DOp $input, (TF_MulOp $filter,
        (TF_ConstOp (Expand1DTo4DForConv2D $value))), $strides, $use_cudnn, $padding,
         $explicit_padding, $data_format, $dilations), (TF_MulOp $bias,
        (TF_ConstOp $value)), $bias_data_format),
  [(HasOneUse $conv_out), (HasOneUse $biasadd_out),
   (HasRank<1> $value), (HasRank<4> $filter),
   (HasEqualElementSize<[3],[0]> $filter, $value),
   (HasEqualShape $value, $bias)]>;

def FuseDepthwiseConv2dWithBiasAndMul : Pat<
  (TF_MulOp
   (TF_BiasAddOp:$biasadd_out
    (TF_DepthwiseConv2dNativeOp:$conv_out $input, $filter, $strides, $padding,
         $explicit_padding, IsDataFormatNHWC:$data_format, $dilations),
    $bias, IsDataFormatNHWC:$bias_data_format),
    (TF_ConstOp IsFloatElementsAttr:$value)),
  (TF_BiasAddOp (TF_DepthwiseConv2dNativeOp $input, (TF_MulOp $filter,
        (TF_ConstOp (Expand1DTo4DForDepthwiseConv2D $value, $filter))), $strides, $padding,
         $explicit_padding, $data_format, $dilations,
        (returnType (CreateUnrankedTensorTypeWithElementType $input))),
        (TF_MulOp $bias, (TF_ConstOp $value)), $bias_data_format),
  [(HasOneUse $conv_out), (HasOneUse $biasadd_out),
   (HasRank<1> $value), (HasRank<4> $filter),
   (HasEqualElementSize<[2,3],[0]> $filter, $value),
   (HasEqualShape $value, $bias)]>;
