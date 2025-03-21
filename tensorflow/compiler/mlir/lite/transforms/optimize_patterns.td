/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// This is the optimization pattern definition file for TensorFlow Lite.

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_enums.td"
include "mlir/IR/CommonAttrConstraints.td"

// Checks if the param passed is a F32 ElementsAttr.
def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<ElementsAttr>() && $_self.cast<ElementsAttr>().getShapedType().getElementType().isF32()">,
        "32 bit float constant tensor">;

// Checks if the param passed is a float ElementsAttr.
def FloatElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<ElementsAttr>() && $_self.cast<ElementsAttr>().getShapedType().getElementType().isa<FloatType>()">,
        "float constant tensor">;

def ExtractSingleElementAsFloat : NativeCodeCall<
    "ExtractSingleElementAsFloat($_self.cast<ElementsAttr>())">;

// Checks if the value has rank 'n'.
class HasRank<int n> : Constraint<
    CPred<"$0.getType().cast<ShapedType>().hasRank() && "
          "$0.getType().cast<ShapedType>().getRank() == " # n>>;

class FloatValueEquals<string val> : Constraint<CPred<
  "FloatValueEquals($0, " # val # ")">>;

class IsBoolAttrEqual<string true_or_false> : Constraint<CPred<
  "$0.getValue() == "#true_or_false#"">>;

// Flattens a constant tensor to 1D.
def FlattenTo1D : NativeCodeCall<"FlattenTo1D($0)">;

def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

def IsPermutationNCHW : Constraint<CPred<"IsPermutationNCHW($0)">>;

def IsBiasShape : Constraint<
    CPred<"$0.getType().cast<ShapedType>().getRank() == 4 && "
          "$0.getType().cast<ShapedType>().getShape()[2] == 1 && "
          "$0.getType().cast<ShapedType>().getShape()[3] == 1">,
    "has shape consistent with a bias">;

def ReshapeNCHWBiasToNHWC : NativeCodeCall<"ReshapeNCHWBiasToNHWC($0, $1)">;

def HasSameType : Constraint<CPred<[{$0.getType() == $1.getType()}]>>;

//===----------------------------------------------------------------------===//
// Ternary ops patterns.
//===----------------------------------------------------------------------===//
// Multi-pattern consisting of matching stand-alone convolution op followed by
// activation op.
multiclass FuseActFnIntoConvOpPat<Op ActFnOp, ConstantStrAttr ActFnAttr> {
  def FuseActivationFuncWithConv#ActFnOp#ActFnAttr : Pat<
    (ActFnOp (TFL_Conv2DOp:$conv_out $input, $filter, $bias, $h_factor,
                 $w_factor, TFL_AF_None, $padding, $stride_h, $stride_w)),
    (TFL_Conv2DOp $input, $filter, $bias, $h_factor, $w_factor, ActFnAttr,
        $padding, $stride_h, $stride_w),
    [(HasOneUse $conv_out)]>;
  def FuseActivationFuncWithDepthwiseConv#ActFnOp#ActFnAttr : Pat<
    (ActFnOp (TFL_DepthwiseConv2DOp:$conv_out $input, $filter, $bias, $h_factor,
                $w_factor, TFL_AF_None, $padding, $stride_h, $stride_w,
                $multiplier)),
    (TFL_DepthwiseConv2DOp $input, $filter, $bias, $h_factor, $w_factor,
        ActFnAttr, $padding, $stride_h, $stride_w, $multiplier),
    [(HasOneUse $conv_out)]>;
}

multiclass FuseActFnIntoPoolOpPat<Op ActFnOp, ConstantStrAttr ActFnAttr> {
  def FuseActivationFuncWithAvgPool#ActFnOp#ActFnAttr : Pat<
    (ActFnOp (TFL_AveragePool2DOp:$pool_out $input, $filter_height,
                  $filter_width, $padding, $stride_h, $stride_w, TFL_AF_None)),
    (TFL_AveragePool2DOp $input, $filter_height, $filter_width, $padding,
        $stride_h, $stride_w, ActFnAttr),
    [(HasOneUse $pool_out)]>;
  def FuseActivationFuncWithMaxPool#ActFnOp#ActFnAttr : Pat<
    (ActFnOp (TFL_MaxPool2DOp:$pool_out $input, $padding, $stride_w, $stride_h,
                  $filter_width, $filter_height, TFL_AF_None)),
    (TFL_MaxPool2DOp $input, $padding, $stride_w, $stride_h,
        $filter_width, $filter_height, ActFnAttr),
    [(HasOneUse $pool_out)]>;
}

// TODO(hinsu): Also fuse ops corresponding to SIGN_BIT fused
// activation functions.
// Currently we're not fusing tanh, sigmoid, hard_swish and other activations
// those cannot be simply translated into clamping.
foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                     [TFL_Relu6Op, TFL_AF_Relu6],
                     [TFL_Relu1Op, TFL_AF_Relu1]] in {
  defm : FuseActFnIntoConvOpPat<!cast<Op>(actFnPair[0]), !cast<ConstantStrAttr>(actFnPair[1])>;
  defm : FuseActFnIntoPoolOpPat<!cast<Op>(actFnPair[0]), !cast<ConstantStrAttr>(actFnPair[1])>;
}

def GetBiasMultiplier:
  NativeCodeCall<"GetBiasMultiplier($_builder, $0, $1.cast<DenseFPElementsAttr>())">;

class CanFuseConvOrDepthwiseConv<string is_depthwise> : Constraint<
  CPred<"TFL::CanFuseConvOrDepthwiseConv($0, $1, " # is_depthwise # ")">>;

// If we see a binary op (add, sub) op adding a constant value to a convolution
// op with constant bias, we can fuse the binary op into the convolution op by
// constant folding the bias and the binary op's constant operand. The following
// pattern restricts to float constant values for now.
multiclass FuseBinaryOpToPrecedingAffine<Op binaryOp> {
  def FuseBinaryOpWithConv#binaryOp : Pat<
    (binaryOp (TFL_Conv2DOp:$output $input, $filter,
                (Arith_ConstantOp FloatElementsAttr:$bias), $h_factor, $w_factor,
                TFL_AF_None, $padding, $stride_h, $stride_w),
              (Arith_ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_Conv2DOp $input, $filter,
      (binaryOp (Arith_ConstantOp $bias),
                (Arith_ConstantOp (FlattenTo1D $value)), TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
     (HasOneUse $output)]>;
  def FuseBinaryOpWithDepthwiseConv#binaryOp : Pat<
    (binaryOp (TFL_DepthwiseConv2DOp:$output $input, $filter,
                (Arith_ConstantOp FloatElementsAttr:$bias),
                $h_factor, $w_factor, TFL_AF_None, $padding, $stride_h,
                $stride_w, $multiplier),
              (Arith_ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_DepthwiseConv2DOp $input, $filter,
      (binaryOp (Arith_ConstantOp $bias),
                (Arith_ConstantOp (FlattenTo1D $value)), TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h, $stride_w,
      $multiplier),
    [(CanFuseConvOrDepthwiseConv<"true"> $filter, $value),
     (HasOneUse $output)]>;
   def FuseBinaryOpWithTransposeConv#binaryOp : Pat<
    (binaryOp (TFL_TransposeConvOp:$output $output_shape, $weights, $input,
                (Arith_ConstantOp FloatElementsAttr:$bias), $padding,
                $stride_h, $stride_w, TFL_AF_None),
              (Arith_ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_TransposeConvOp $output_shape, $weights, $input,
      (binaryOp (Arith_ConstantOp $bias),
         (Arith_ConstantOp $value), TFL_AF_None),
      $padding, $stride_h, $stride_w, $act_fn),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (HasOneUse $output)]>;
  // Fuse for TransposeConv with no bias
  def FuseBinaryOpWithTransposeConvNoneBias#binaryOp : Pat<
    (binaryOp:$root (TFL_TransposeConvOp:$output $output_shape, $weights, $input,
                $bias, $padding,
                $stride_h, $stride_w, TFL_AF_None),
              (Arith_ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_TransposeConvOp $output_shape, $weights, $input,
      (TFL_MulOp  (Arith_ConstantOp $value),
                  (GetBiasMultiplier $root, $value),
                  TFL_AF_None
      ),
      $padding, $stride_h, $stride_w, $act_fn),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (IsNoneType $bias),
     (HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp]<Op> in
  defm : FuseBinaryOpToPrecedingAffine<binaryOp>;

def ExpandTo4DForConv: NativeCodeCall<"ExpandTo4DForConv($0)">;

def ExpandTo4DForDepthwiseConv: NativeCodeCall<
  "ExpandTo4DForDepthwiseConv($0)">;

def IsScaleOfSum:
   Constraint<CPred<"IsScaleOfSum($0, $1, $2)">>;

// If we see a (div or Mul) op (dividing/multiplying) a constant value
// to a convolution op with constant filter and bias, we can fuse the div/mul
// into the convolution op by constant folding
// the filter/bias and the div/mul op's constant operand.
// The following pattern restricts to float constant values for now.

multiclass FuseMulOrDivWithConv2dOrDepthwiseConv2d<Op BinaryOp> {
  def FuseMulOrDivWithDepthwiseConv#BinaryOp : Pat<
    (BinaryOp (TFL_DepthwiseConv2DOp:$output $input,
                (Arith_ConstantOp FloatElementsAttr:$filter),
                (Arith_ConstantOp FloatElementsAttr:$bias),
                $h_factor, $w_factor, TFL_AF_None, $padding, $stride_h,
                $stride_w, $multiplier),
              (Arith_ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_DepthwiseConv2DOp $input,
      (BinaryOp
        (Arith_ConstantOp $filter),
        (Arith_ConstantOp (ExpandTo4DForDepthwiseConv $value)),
        TFL_AF_None),
      (BinaryOp
        (Arith_ConstantOp $bias),
        (Arith_ConstantOp (FlattenTo1D $value)),
        TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h,
      $stride_w, $multiplier),
    [(CanFuseConvOrDepthwiseConv<"true"> $filter, $value),
     (HasOneUse $output)]>;
  def FuseMulOrDivWithConv#BinaryOp : Pat<
    (BinaryOp (TFL_Conv2DOp:$conv_output $input,
                (Arith_ConstantOp FloatElementsAttr:$filter),
                (Arith_ConstantOp FloatElementsAttr:$bias),
                $h_factor, $w_factor, TFL_AF_None,
                $padding, $stride_h, $stride_w),
              (Arith_ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_Conv2DOp $input,
      (BinaryOp (Arith_ConstantOp $filter),
        (Arith_ConstantOp (ExpandTo4DForConv $value)),
        TFL_AF_None),
      (BinaryOp (Arith_ConstantOp $bias),
        (Arith_ConstantOp (FlattenTo1D $value)),
        TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
     (HasOneUse $conv_output)]>;
  def FuseMulOrDivWithTransposeConv#BinaryOp : Pat<
    (BinaryOp (TFL_TransposeConvOp:$output $output_shape,
                (Arith_ConstantOp FloatElementsAttr:$weights), $input,
                (Arith_ConstantOp FloatElementsAttr:$bias),
                $padding, $stride_h, $stride_w, TFL_AF_None),
              (Arith_ConstantOp $value), $act_fn),
    (TFL_TransposeConvOp $output_shape,
      (BinaryOp (Arith_ConstantOp $weights),
        (Arith_ConstantOp (ExpandTo4DForConv $value)),
        TFL_AF_None),
      $input,
      (BinaryOp (Arith_ConstantOp $bias),
        (Arith_ConstantOp $value),
        TFL_AF_None),
      $padding, $stride_h, $stride_w, $act_fn),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (HasOneUse $output)]>;
  def FuseMulOrDivWithTransposeConvWithNoneBias#BinaryOp : Pat<
    (BinaryOp (TFL_TransposeConvOp:$output $output_shape,
                (Arith_ConstantOp FloatElementsAttr:$weights), $input,
                $bias,
                $padding, $stride_h, $stride_w, TFL_AF_None),
              (Arith_ConstantOp $value), $act_fn),
    (TFL_TransposeConvOp $output_shape,
      (BinaryOp (Arith_ConstantOp $weights),
        (Arith_ConstantOp (ExpandTo4DForConv $value)),
        TFL_AF_None),
      $input,
      $bias,
      $padding, $stride_h, $stride_w, $act_fn),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (IsNoneType $bias),
     (HasOneUse $output)]>;
}

foreach BinaryOp = [TFL_DivOp, TFL_MulOp]<Op> in
  defm : FuseMulOrDivWithConv2dOrDepthwiseConv2d<BinaryOp>;

// Squash tfl.dequantize and tfl.quantize pairs.
def : Pat<(TFL_QuantizeOp:$out (TFL_DequantizeOp $in), $qt),
          (replaceWithValue $in),
          [(HasSameType $in, $out)]>;

// Matching HardSwish
def MatchHardSwishPattern1 : Pat<
  (TFL_MulOp
    (TFL_MulOp
     $x, (TFL_AddOp
          $x,
          (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
          TFL_AF_Relu6),
     TFL_AF_None),
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
  (TFL_HardSwishOp $x)>;

def MatchHardSwishPattern2 : Pat<
  (TFL_MulOp
    $x,
    (TFL_MulOp
     (TFL_AddOp
      $x,
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_Relu6),
     (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
     TFL_AF_None),
  (TFL_HardSwishOp $x)>;

def MatchHardSwishPattern3 : Pat<
  (TFL_MulOp
    (TFL_MulOp
     $x,
     (TFL_AddOp
      $x,
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_Relu6),
     TFL_AF_None),
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
    TFL_AF_None),
  (TFL_HardSwishOp $x)>;

def MatchHardSwishPattern4 : Pat<
  (TFL_MulOp
    (TFL_MulOp
     (TFL_AddOp
      $x,
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_Relu6),
     (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
    $x,
    TFL_AF_None),
  (TFL_HardSwishOp $x)>;

// Matching HardSwish with extra FakeQuant. These FakeQuant ops were due to
// incorrect placement in the quantization aware training.
def MatchHardSwishQuantized : Pat<
  (TFL_MulOp (TFL_DequantizeOp (TFL_QuantizeOp
    (TFL_MulOp
     $x, (TFL_DequantizeOp (TFL_QuantizeOp (TFL_AddOp
          $x,
          (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
          TFL_AF_Relu6), $qattr2)),
     TFL_AF_None), $qattr1)),
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
  (TFL_HardSwishOp $x)>;

def MatchHardSwishPattern5 : Pat<
  (TFL_MulOp
     $arg,
     (TFL_DivOp
      (TFL_AddOp
        $arg,
        (Arith_ConstantOp F32ElementsAttr:$cst_3),
        TFL_AF_Relu6),
      (Arith_ConstantOp F32ElementsAttr:$cst_6),
      TFL_AF_None),
     TFL_AF_None),
  (TFL_HardSwishOp $arg),
  [
    (FloatValueEquals<"3"> $cst_3),
    (FloatValueEquals<"6"> $cst_6),
  ]>;

def MatchHardSwishPattern6 : Pat<
  (TFL_MulOp
     $arg,
     (TFL_MulOp
      (TFL_AddOp
        $arg,
        (Arith_ConstantOp F32ElementsAttr:$cst_3),
        TFL_AF_Relu6),
      (Arith_ConstantOp F32ElementsAttr:$cst_one_sixth),
      TFL_AF_None),
     TFL_AF_None),
  (TFL_HardSwishOp $arg),
  [
    (FloatValueEquals<"3"> $cst_3),
    (FloatValueEquals<"0.166666672"> $cst_one_sixth),
  ]>;

// Constraint that the attribute value is less than 'n'
class ConstDoubleValueLessThan<string n> : Constraint<
  CPred<"$0.isa<DenseElementsAttr>() && "
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
  "std::abs(*$0.cast<DenseElementsAttr>().getValues<float>().begin()) < "
  # n>>;

// Constraint that the attribute value is negative infinity or negative largest.
// We use both -inf & flt_min due to the forward compatibility.
def ConstAPFloatNegLargestOrNegInfinity : Constraint<CPred<
  "$0.isa<DenseElementsAttr>() && "
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
  "(($0.cast<DenseElementsAttr>().getValues<APFloat>()[0].isLargest() && "
  "$0.cast<DenseElementsAttr>().getValues<APFloat>()[0].isNegative()) || "
  "$0.cast<DenseElementsAttr>().getValues<APFloat>()[0].isNegInfinity())">>;

def L2NormValidReduceIndex : Constraint<CPred<
  "L2NormalizeReduceAxis($0, $1.cast<DenseElementsAttr>())">>;

// Currently L2Normalization doesn't support activation function
// in TFLite.
// TODO(karimnosseir): Add constraints that the kernel code assumes.
// constraint on axis and depth.
multiclass L2NormalizePatterns<Op FirstOp, Op SecondOp> {
  // This pattern constructs L2NormalizationOp from
  // Mul->Rsqrt->Sum->Square Or
  // Div->sqrt->Sum->Square
  def L2NormalizePattern1#FirstOp#SecondOp : Pat<
                  (FirstOp $x,
                     (SecondOp
                        (TFL_SumOp
                           (TFL_SquareOp:$sq_op $x),
                           (Arith_ConstantOp I32ElementsAttr:$axis),
                           $keep_dims)),
                     TFL_AF_None),
           (TFL_L2NormalizationOp $x, TFL_AF_None),
           [(L2NormValidReduceIndex $sq_op, $axis)]>;

  // Below patterns for L2Normalize when there is an Add or Maximum
  // adding or clamping to a small constant scalar.
  def L2NormalizePattern2#FirstOp#SecondOp : Pat<
                    (FirstOp $x,
                     (SecondOp
                      (TFL_AddOp
                       (TFL_SumOp
                        (TFL_SquareOp:$sq_op $x),
                        (Arith_ConstantOp I32ElementsAttr:$axis),
                        $keep_dims),
                       (Arith_ConstantOp $epsilon), TFL_AF_None)),
           TFL_AF_None),
           (TFL_L2NormalizationOp $x, TFL_AF_None),
           [(L2NormValidReduceIndex $sq_op, $axis),
            (ConstDoubleValueLessThan<"1e-3"> $epsilon)]>;

  def L2NormalizePattern3#FirstOp#SecondOp : Pat<
                    (FirstOp $x,
                     (SecondOp
                      (TFL_MaximumOp
                       (TFL_SumOp
                        (TFL_SquareOp:$sq_op $x),
                        (Arith_ConstantOp I32ElementsAttr:$axis),
                        $keep_dims),
                       (Arith_ConstantOp $epsilon))),
           TFL_AF_None),
           (TFL_L2NormalizationOp $x, TFL_AF_None),
           [(L2NormValidReduceIndex $sq_op, $axis),
            (ConstDoubleValueLessThan<"1e-3"> $epsilon)]>;

}

foreach L2NormalizePairs = [[TFL_MulOp, TFL_RsqrtOp], [TFL_DivOp, TFL_SqrtOp]]
  in defm : L2NormalizePatterns<L2NormalizePairs[0], L2NormalizePairs[1]>;

//===----------------------------------------------------------------------===//
// Binary ops patterns.
//===----------------------------------------------------------------------===//
def AreBroadcastableTypes : Constraint<CPred<
  "TFL::IsBroadcastableElementsAttrAndType($0.getType(), $1.getType())">>;

def IsTailOfShape : Constraint<CPred<
  "TFL::IsTailOfShape($0.getType(), $1.getType())">>;

def IsReducedTailOfShape : Constraint<CPred<
  "TFL::IsReducedTailOfShape($0.getType(), $1.getType())">>;

def Flatten : NativeCodeCall<
  "$0.cast<DenseElementsAttr>()"
    ".reshape(RankedTensorType::get({$0.getType().cast<ShapedType>().getNumElements()}, "
                                   "$0.getType().cast<ShapedType>().getElementType()))">;

def IsLastDimEqualToNumElements : Constraint<CPred<
  "TFL::IsLastDimEqualToNumElements($0.getType(), $1.getType())">>;

def IsDefinedByFullyConnectedOp : Constraint<CPred<
  "$0.getDefiningOp<TFL::FullyConnectedOp>() != nullptr">>;

def IsDefinedByConv2DOp : Constraint<CPred<
  "$0.getDefiningOp<TFL::Conv2DOp>() != nullptr">>;

// Returns true if the supplied value-
// 1) Has only one use or
// 2) Is only used by binary op like AddOp, SubOp, MulOp or DivOp.
def HasOneUseOrUsedByOnlyBinaryOps : Constraint<CPred<
  "TFL::HasOneUseOrUsedByOnlyBinaryOps($0)">>;

// Pattern for skipping Tile if it is mainly for broadcasting and the
// Op is already supporting broadcasting.
multiclass FuseTileBroadcastIntoFollowingBinary<Op BinaryOp> {
  def FuseTileBroadcastToBinaryOp1#BinaryOp : Pat<
    (BinaryOp:$result (TFL_TileOp $input, (Arith_ConstantOp $tile)),
     $operand, $act_func),
    (BinaryOp $input, $operand, $act_func),
  [(OperandsBroadcastToOutputType $input, $operand, $result),
   (HasRankAtMost<4> $input),
   (HasRankAtMost<4> $operand)]>;

  def FuseTileBroadcastToBinaryOp2#BinaryOp : Pat<
    (BinaryOp:$result $operand,
      (TFL_TileOp $input, (Arith_ConstantOp $tile)), $act_func),
    (BinaryOp $operand, $input, $act_func),
  [(OperandsBroadcastToOutputType $operand, $input, $result),
   (HasRankAtMost<4> $operand),
   (HasRankAtMost<4> $input)]>;
}

// Multi-pattern consisting of matching stand-alone op or op followed by relu.
multiclass FusedBinaryActivationFuncOpPat<Op BinaryOp> {
  foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                       [TFL_Relu6Op, TFL_AF_Relu6],
                       [TFL_Relu1Op, TFL_AF_Relu1]] in {
    def FuseBinaryWithActivation#BinaryOp#actFnPair[0] : Pat<
      (actFnPair[0] (BinaryOp:$binary_out $lhs, $rhs, TFL_AF_None)),
      (BinaryOp $lhs, $rhs, actFnPair[1]),
    [(HasOneUse $binary_out)]>;
  }
}

// Returns true if the supplied values both have one use each
// or are reused by the same user.
def HasOneUseOrReusedBySameOp : Constraint<CPred<
  "((std::distance($0.getUses().begin(), $0.getUses().end()) == 2) && ($0 == $1))"
    "|| ($0.hasOneUse() && $1.hasOneUse())">>;

foreach BinaryOp = [TFL_AddOp, TFL_SubOp, TFL_DivOp, TFL_MulOp] in {
  defm : FuseTileBroadcastIntoFollowingBinary<BinaryOp>;

  // Instantiated FusedBinary patterns for the from-to pairs of ops.
  defm : FusedBinaryActivationFuncOpPat<BinaryOp>;

  // Move binary op before reshape: reshape -> binary => binary -> reshape.
  // This is valid only when the binary operand is constant and the shape is the
  // tail of the other operand and the intermediate result isn't used by other
  // ops.
  // $rhs is required to be the tail shape of $lhs, so after transformation the
  // shape of the binary op result is valid. For example, assume the shapes of
  // $input, $lhs and $rhs are [1600], [1,40,40] and [40x1]. After the
  // transformation, the shape of the binary op result is [40x1600], which
  // couldn't be reshaped to [1,40,40]. `IsTailOfShape` constraint is added to
  // make sure $rhs is the tail shape of $lhs.
  def MoveBinaryOpConstBeforeReshape#BinaryOp : Pat<
    (BinaryOp (TFL_ReshapeOp:$lhs $input, (Arith_ConstantOp:$shape $s)),
      (Arith_ConstantOp:$rhs $a), $act_fn),
    (TFL_ReshapeOp (BinaryOp $input, $rhs, $act_fn), $shape),
    // The broadcasting of "BinaryOp" only happens in the lower
    // dimensions, and the higher dimensions are same, so we know the
    // result and input of the "BinaryOp" in the source pattern have
    // the same shape, which is defined by `shape`.
    [(IsTailOfShape $rhs, $lhs),
     (HasOneUse $lhs),
     // The result of the new "BinaryOp" will have the same shape as
     // `input`. In other words, the shape of the `Reshape` op are not
     // changed after the transformation.
     (IsTailOfShape $rhs, $input),
     (HasRankAtMost<4> $input),
     (HasRankAtMost<4> $lhs),
     (HasRankAtMost<4> $rhs),
     (SameElementType $input, $rhs)]>;

  // Move binary op before reshape:
  // binary(reshape(lhs), reshape(rhs)) => reshape(binary(lhs, rhs))
  // This is valid only when both side of the binary operand is reshaped, and
  // the sizes are the same both before and after the reshape.
  def MoveBinaryOpBeforeReshape#BinaryOp : Pat<
    (BinaryOp (TFL_ReshapeOp:$lhs $input1, (Arith_ConstantOp:$shape1 $s1)),
              (TFL_ReshapeOp:$rhs $input2, (Arith_ConstantOp:$shape2 $s2)),
              $act_fn),
    (TFL_ReshapeOp (BinaryOp $input1, $input2, $act_fn), $shape1),
    [(IsTailOfShape $rhs, $lhs),
     (IsTailOfShape $lhs, $rhs),
     (IsTailOfShape $input1, $input2),
     (IsTailOfShape $input2, $input1),
     (SameElementType $input1, $input2),
     (HasOneUseOrReusedBySameOp $lhs, $rhs)]>;

  // Move binary op batched RHS before reshape:
  // binary(reshape(lhs), rhs) => reshape(binary(lhs, flatten(rhs)))
  // Pattern targetted here is as follows-
  // [input, lhr, rhs] == [<1x1024x128>, <1x1024x8x16>, <1x1x8x16xf32>]
  // This is valid only when the-
  // 1.last dimension of lhs is equal to the number of elements in constant rhs.
  // 2.Reduded shape of rhs, here <8x16> is equal to last dimensions of lhs.
  // Therefore, after transformation broadcast of binary op is always
  // applied to the last dimension of $input.
  def MoveBinaryOpFlattenConstBeforeReshape#BinaryOp : Pat<
    (BinaryOp (TFL_ReshapeOp:$lhs $input, (Arith_ConstantOp:$shape $s)),
              (Arith_ConstantOp:$rhs ElementsAttr:$rhs_attr), $act_fn),
    (TFL_ReshapeOp (BinaryOp $input, (Arith_ConstantOp (Flatten $rhs_attr)),
                              $act_fn),
                    $shape),
    [(AnyStaticShapeTensor $input),
      (IsReducedTailOfShape $rhs, $lhs),
      (IsLastDimEqualToNumElements $input, $rhs),
      (HasOneUse $lhs),
      // Restrict operands to have at most rank 4 because TFLite binary
      // kernel supports up to 4D broadcast.
      (HasRankAtMost<4> $input),
      (HasRankAtMost<4> $lhs),
      (HasRankAtMost<4> $rhs),
      (IsDefinedByFullyConnectedOp $input)]>;

  // Pattern to remove redundant reshape op used as LHS to binary ops
  // Binary(Reshape(input, shape), rhs) -> Binary(input, rhs)
  // This pattern is valid only if-
  // 1. The shape is only adding broadcasting that can otherwise be implicitly
  // handled by the binary op. Ex- shape == [1, 1, 1, 128]
  // 2. The rank of the input to reshape is <= reshape output.
  // 3. The rank of the output to reshape is <= binary rhs.
  // The conditions 2 and 3 will make sure any required increase in
  // dimentionality dure to reshape op is not lost.
  def RemoveRedundantReshapeUsedAsLhsTo#BinaryOp : Pat<
    (BinaryOp (TFL_ReshapeOp:$lhs $input, (Arith_ConstantOp:$shape $s)),
              $rhs, $act_fn),
    (BinaryOp $input, $rhs, $act_fn),
    [(AnyStaticShapeTensor $input),
     (AnyStaticShapeTensor $rhs),
     (IsRankLessThanEqualTo $input, $lhs),
     (IsRankLessThanEqualTo $lhs, $rhs),
     (IsReducedTailOfShape $lhs, $input),
     (HasOneUseOrUsedByOnlyBinaryOps $lhs),
     // Restrict operands to have at most rank 4 because TFLite binary
     // kernel supports up to 4D broadcast.
     (HasRankAtMost<4> $input),
     (HasRankAtMost<4> $rhs)]>;

  // Pattern to remove redundant reshape op used as RHS to binary ops
  // Binary(lhs, Reshape(input, shape)) -> Binary(lhs, input)
  // This pattern is valid only if-
  // 1. The shape is only adding broadcasting that can otherwise be implicitly
  // handled by the binary op. Ex- shape == [1, 1, 1, 128]
  // 2. The rank of the input to reshape is <= reshape output.
  // 3. The rank of the output to reshape is <= binary lhs.
  // The conditions 2 and 3 will make sure any required increase in
  // dimentionality dure to reshape op is not lost.
  def RemoveRedundantReshapeUsedAsRhsTo#BinaryOp : Pat<
    (BinaryOp $lhs, (TFL_ReshapeOp:$rhs $input, (Arith_ConstantOp:$shape $s)),
              $act_fn),
    (BinaryOp $lhs, $input, $act_fn),
    [(AnyStaticShapeTensor $input),
     (AnyStaticShapeTensor $lhs),
     (IsRankLessThanEqualTo $input, $rhs),
     (IsRankLessThanEqualTo $rhs, $lhs),
     (IsReducedTailOfShape $rhs, $input),
     (HasOneUseOrUsedByOnlyBinaryOps $rhs),
     // Restrict operands to have at most rank 4 because TFLite binary
     // kernel supports up to 4D broadcast.
     (HasRankAtMost<4> $input),
     (HasRankAtMost<4> $lhs)]>;
}

foreach BinaryOp = [TFL_FloorDivOp, TFL_FloorModOp, TFL_MinimumOp,
                    TFL_MaximumOp, TFL_LessOp, TFL_LessEqualOp, TFL_GreaterOp,
                    TFL_GreaterEqualOp] in {
  // Move binary op before reshape: reshape -> binary => binary -> reshape.
  // This is valid only when the binary operand is constant and the shape is the
  // tail of the other operand and the intermediate result isn't used by other
  // ops.
  // $rhs is required to be the tail shape of $lhs, so after transformation the
  // shape of the binary op result is valid. For example, assume the shapes of
  // $input, $lhs and $rhs are [1600], [1,40,40] and [40x1]. After the
  // transformation, the shape of the binary op result is [40x1600], which
  // couldn't be reshaped to [1,40,40]. `IsTailOfShape` constraint is added to
  // make sure $rhs is the tail shape of $lhs.
  def MoveBinaryOpConstBeforeReshape#BinaryOp : Pat<
    (BinaryOp (TFL_ReshapeOp:$lhs $input, (Arith_ConstantOp:$shape $s)),
      (Arith_ConstantOp:$rhs $a)),
    (TFL_ReshapeOp (BinaryOp $input, $rhs), $shape),
    // The broadcasting of "BinaryOp" only happens in the lower
    // dimensions, and the higher dimensions are same, so we know the
    // result and input of the "BinaryOp" in the source pattern have
    // the same shape, which is defined by `shape`.
    [(IsTailOfShape $rhs, $lhs),
     (HasOneUse $lhs),
     // The result of the new "BinaryOp" will have the same shape as
     // `input`. In other words, the shape of the `Reshape` op are not
     // changed after the transformation.
     (IsTailOfShape $rhs, $input),
     (HasRankAtMost<4> $input),
     (HasRankAtMost<4> $lhs),
     (HasRankAtMost<4> $rhs),
     (SameElementType $input, $rhs)]>;

    // Move binary op before reshape:
    // binary(reshape(lhs), reshape(rhs)) => reshape(binary(lhs, rhs))
    // This is valid only when both side of the binary operand is reshaped, and
    // the sizes are the same both before and after the reshape.
    def MoveBinaryOpBeforeReshape#BinaryOp : Pat<
      (BinaryOp (TFL_ReshapeOp:$lhs $input1, (Arith_ConstantOp:$shape1 $s1)),
                (TFL_ReshapeOp:$rhs $input2, (Arith_ConstantOp:$shape2 $s2))),
      (TFL_ReshapeOp (BinaryOp $input1, $input2), $shape1),
      [(IsTailOfShape $rhs, $lhs),
       (IsTailOfShape $lhs, $rhs),
       (IsTailOfShape $input1, $input2),
       (IsTailOfShape $input2, $input1),
       (SameElementType $input1, $input2)]>;

    // Move binary op before reshape:
    // binary(reshape(lhs), rhs) => reshape(binary(lhs, flatten(rhs)))
    // This is valid only when the last dimension of lhs is equal to the
    // number of elements in constant rhs.
    // Therefore, after transformation broadcast of binary op is always
    // applied to the last dimension of $input.
    def MoveBinaryOpFlattenConstBeforeReshape#BinaryOp : Pat<
      (BinaryOp (TFL_ReshapeOp:$lhs $input, (Arith_ConstantOp:$shape $s)),
                (Arith_ConstantOp:$rhs ElementsAttr:$rhs_attr)),
      (TFL_ReshapeOp (BinaryOp $input, (Arith_ConstantOp (Flatten $rhs_attr))),
                     $shape),
      [(AnyStaticShapeTensor $input),
       (IsTailOfShape $rhs, $lhs),
       (IsLastDimEqualToNumElements $input, $rhs),
       (HasOneUse $lhs),
       // Restrict operands to have at most rank 4 because TFLite binary
       // kernel supports up to 4D broadcast.
       (HasRankAtMost<4> $input),
       (HasRankAtMost<4> $lhs),
       (HasRankAtMost<4> $rhs),
       (IsDefinedByFullyConnectedOp $input)]>;
}

// Reorder the element-wise value operations and the element move operations,
// such that the value operation happens before move operation.
foreach ValueOp = [TFL_CeilOp, TFL_ExpOp, TFL_FloorOp, TFL_NegOp,
                   TFL_ReluOp, TFL_Relu1Op, TFL_Relu6Op, TFL_RoundOp,
                   TFL_TanhOp, TFL_SqrtOp, TFL_SquareOp, TFL_LogisticOp] in {
  foreach MoveOp = [TFL_DepthToSpaceOp, TFL_ExpandDimsOp, TFL_SqueezeOp,
                   TFL_ReshapeOp, TFL_TransposeOp] in {
    def ReorderElementwiseAndMoveOperations#ValueOp#MoveOp : Pat<
      (ValueOp:$value (MoveOp:$move $input, $move_def)),
      (MoveOp (ValueOp $input), $move_def),
      [(SameElementType $input, $value), (HasOneUse $move)]>;
  }
}

// Returns truncated shape of a ranked-tensor.
// Prefix-Truncated, here, means eliminating any contiguous 1s' in the lower
// dimentions of the tensor
def GetPrefixTruncatedShape: NativeCodeCall<"GetShape($0, true)">;

// Returns True if the operand type is RankedTensorType and valid.
def HasValidRankedTensor : Constraint<CPred<
  "$0.getType().isa<RankedTensorType>() && "
  "$0.getType().cast<RankedTensorType>().getNumDynamicDims() <= 1">>;

// Check if the truncated shape of the lhs is equal to the shape of rhs
def IsPrefixTruncatedShapeEqualTo : Constraint<CPred<
  "GetShape($0, true) == GetShape($1)">>;

def ConvertSqueezeToReshape : Pat<
  (TFL_SqueezeOp:$squeeze_op $input, $squeeze_dims),
  (TFL_ReshapeOp $input, (Arith_ConstantOp (GetShape $squeeze_op))),
  [(HasValidRankedTensor $squeeze_op)]>;

// Pattern to perform the following optimization
// transpose [1xAxB] -> [1xBxA]
//    |
// reshape [1xBxA] -> [BxA]
//    |
// transpose [BxA] -> [AxB]
//    ||
// reshape [1xAxB] -> [AxB]
def ConvertTrasposeReshapeTransposeToReshape : Pat<
  (TFL_TransposeOp:$second_transpose
    (TFL_ReshapeOp:$middle_reshape
      (TFL_TransposeOp:$first_transpose $input, $permutation2),
        $shape),
      $permutation1),
  (TFL_ReshapeOp $input, (Arith_ConstantOp (GetPrefixTruncatedShape $input))),
  [(IsPrefixTruncatedShapeEqualTo $first_transpose, $middle_reshape),
   (IsPrefixTruncatedShapeEqualTo $input, $second_transpose)]>;

class AllUsersAreAddOpsAndCanFuseWith<string op_to_fuse_with> : Constraint<CPred<
  "llvm::all_of($0.getUsers(), [&](Operation *user) {"
  "  if(!llvm::isa<TFL::AddOp>(user)) return false;"
  "  if(!user->getOperand(0).getDefiningOp<"#op_to_fuse_with#">()) return false;"
  "  return true;"
  "})">, "all users are AddOp and can fuse with "#op_to_fuse_with#"">;

// TODO(b/294385379): This pattern only appears when we convert
// from shlo due to differences in broadcasting behavior
def UndoBroadcastFullyConnectedBiasAdd : Pat<
  (TFL_AddOp $lhs, (Arith_ConstantOp:$const_value $bias), TFL_AF_None),
  (TFL_AddOp $lhs, (Arith_ConstantOp (FlattenTo1D $bias)), TFL_AF_None),
  [(AnyStaticShapeTensor $lhs),
   (IsLastDimEqualToNumElements $bias, $bias),
   (HasRankAtMost<4> $bias),
   (HasRankAtLeast<2> $bias),
   (IsDefinedByFullyConnectedOp $lhs),
   (AllUsersAreAddOpsAndCanFuseWith<"TFL::FullyConnectedOp"> $const_value)]>;

// TODO(b/294385379): This pattern only appears when we convert
// from shlo due to differences in broadcasting behavior.
def UndoBroadcastConvBiasAdd : Pat<
  (TFL_AddOp $lhs, (Arith_ConstantOp:$const_op $bias), TFL_AF_None),
  (TFL_AddOp $lhs, (Arith_ConstantOp (FlattenTo1D $bias)), TFL_AF_None),
  [(AnyStaticShapeTensor $lhs),
   (IsLastDimEqualToNumElements $bias, $bias),
   (HasOneUse $const_op),
   (HasRankAtMost<4> $bias),
   (HasRankAtLeast<2> $bias),
   (IsDefinedByConv2DOp $lhs)]>;

// Pattern to convert a trivial transpose op to a reshape op.
def ConvertTrivialTransposeOpToReshapeOp : Pat<
  (TFL_TransposeOp:$transpose_op $input, (Arith_ConstantOp:$permutation $p1)),
  (TFL_ReshapeOp $input, (Arith_ConstantOp (GetShape $transpose_op))),
  [(IsTransposeTrivial $input, $permutation),
   (AnyStaticShapeTensor $input),
   (AnyStaticShapeTensor $transpose_op)]>;

// Pattern to fuse redundant tanspose op
def FoldDoubleTranspose : Pat<
  (TFL_TransposeOp
    (TFL_TransposeOp:$transpose_out1 $input, (Arith_ConstantOp:$permutation1 $p1)),
    (Arith_ConstantOp:$permutation2 $p2)),
  (TFL_TransposeOp $input,
    (Arith_ConstantOp (RemapPermutation $permutation1, $permutation2))),
  [(HasOneUse $transpose_out1)]>;

// Convert expand_dims to reshape if possible.
def ConvertExpandDimsToReshape : Pat<
  (TFL_ExpandDimsOp:$expand_dims_op $input, $dim),
  (TFL_ReshapeOp $input, (Arith_ConstantOp (GetShape $expand_dims_op))),
  [(AnyStaticShapeTensor $expand_dims_op)]>;

// Here, the element type can be any integer or float type.
class IsConstantValueOf<int value> : Constraint<CPred<
  "TFL::IsConstantValueOf($0," # value # ")">>;

// ReLU patterns
def MatchReluPattern : Pat<
  (TFL_MaximumOp $input, (Arith_ConstantOp $Zero)),
  (TFL_ReluOp $input),
  [(FloatValueEquals<"0"> $Zero)]>;

// Optimize Minimum of tf.Relu and constant six to tf.Relu6
def MinimumOfReluAnd6ToRelu6 :
  Pat<(TFL_MinimumOp (TFL_ReluOp $x), (Arith_ConstantOp $y)),
      (TFL_Relu6Op $x),
      [(IsConstantValueOf<6> $y)]>;

def MinimumOfReluAnd6ToRelu62 :
  Pat<(TFL_ReluOp
        (TFL_MinimumOp $x,
          (Arith_ConstantOp $y))),
      (TFL_Relu6Op $x),
      [(IsConstantValueOf<6> $y)]>;

// For both relu1 and relu_0_to_1, the min/max operators commute,
// so there are two possible orderings we need to rewrite.
// Concretely, `m < n -> max(m, min(n, x)) = min(m, max(m, x))`.
// Proof:
// case (x <= m)
//   max(m, min(n, x)) = max(m, m) = m and
//   min(n, max(m, x)) = min(n, m) = m
// case (m < x < n)
//   max(m, min(n, x)) = max(m, x) = x and
//   min(n, max(m, x)) = min(n, x) = x
// case (n <= x)
//   max(m, min(n, x)) = max(m, n) = n and
//   min(n, max(m, x)) = min(n, x) = n
def MatchRelu1Pattern1 : Pat<
  (TFL_MinimumOp (TFL_MaximumOp $input, (Arith_ConstantOp $NegOne)),
    (Arith_ConstantOp $One)),
  (TFL_Relu1Op $input),
  [(FloatValueEquals<"-1"> $NegOne), (FloatValueEquals<"1"> $One)]>;

def MatchRelu1Pattern2 : Pat<
  (TFL_MaximumOp (TFL_MinimumOp $input, (Arith_ConstantOp $One)),
    (Arith_ConstantOp $NegOne)),
  (TFL_Relu1Op $input),
  [(FloatValueEquals<"-1"> $NegOne), (FloatValueEquals<"1"> $One)]>;

def MatchRelu0To1Pattern1: Pat<
  (TFL_MinimumOp (TFL_MaximumOp $x, (Arith_ConstantOp $max_cst)),
    (Arith_ConstantOp $min_cst)),
    (TFL_Relu0To1Op $x),
  [(FloatValueEquals<"0"> $max_cst), (FloatValueEquals<"1"> $min_cst)]>;

def MatchRelu0To1Pattern2: Pat<
  (TFL_MaximumOp (TFL_MinimumOp $x, (Arith_ConstantOp $min_cst)),
    (Arith_ConstantOp $max_cst)),
    (TFL_Relu0To1Op $x),
  [(FloatValueEquals<"0"> $max_cst), (FloatValueEquals<"1"> $min_cst)]>;

def MatchLeakyRelu : Pat<
  (TFL_MaximumOp
    (TFL_MulOp:$mul_out $x,
     (Arith_ConstantOp F32ElementsAttr:$alpha), TFL_AF_None),
    $x),
  (TFL_LeakyReluOp $x, ExtractSingleElementAsFloat:$alpha),
  [(ConstDoubleValueLessThan<"1"> $alpha),
   (HasOneUse $mul_out)]>;

// Returns True if all users of this operation are in TF/TFL and don't need
// shape exact matching. This prevents from removing cast on return values which
// can break the verifier on function type mismatch.
def AllUsersInTF : Constraint<CPred<[{
  llvm::all_of($0.getUsers(), [&](Operation *user) {
    auto name = user->getName().getDialectNamespace();
    return name == "tf" || name == "tfl";
  })
  }]>, "all users are TF/TFL operations.">;

def RemoveShapeOnlyCast : Pat<(TFL_CastOp:$output $input),
                            (replaceWithValue $input),
                            [(SameElementType $input, $output),
                             (AllUsersInTF $output)]>;


// Checks if the operand0's rank is one less than operand1's rank.
def PReluAlphaRankCheck : Constraint<
  CPred<"$0.getType().cast<ShapedType>().getRank() == "
  "$1.getType().cast<ShapedType>().getRank() - 1">>;

// PReLU pattern from Keras:
// f(x) = Relu(x) + (-alpha * Relu(-x))
def MatchPRelu : Pat<
  (TFL_AddOp
   (TFL_ReluOp:$relu_out $x),
   (TFL_MulOp:$mul_out
    (TFL_ReluOp (TFL_NegOp:$input_neg_out $x)),
    $neg_alpha,
    TFL_AF_None),
   TFL_AF_None),
  (TFL_PReluOp $x, (TFL_NegOp $neg_alpha)),
  [(PReluAlphaRankCheck $neg_alpha, $x),
   (HasOneUse $relu_out),
   (HasOneUse $mul_out),
   (HasOneUse $input_neg_out)]>;

// The constant folding in this pass might produce constant in the tf dialect.
// This rule is to legalize these constant to the tfl dialect.
def LegalizeConstOp : Pat<
  (TF_ConstOp ElementsAttr:$value), (TFL_ConstOp $value)>;

// Reorders adds to allow constant folding.
// Add --> Add $input, $constantA
//    \--> $constantB
// To
// Add --> $input
//    \--> Add ($constantA, $constantB)
foreach ActFun = [TFL_AF_Relu, TFL_AF_Relu6, TFL_AF_Relu1, TFL_AF_None] in {
  def ReorderAddToAllowConstFold_ActFunc_#ActFun : Pat<
    (TFL_AddOp
     (TFL_AddOp:$first_output $input, (Arith_ConstantOp $a), TFL_AF_None),
     (Arith_ConstantOp $b), ActFun),
    (TFL_AddOp $input,
     (TFL_AddOp (Arith_ConstantOp $a), (Arith_ConstantOp $b), TFL_AF_None),
     ActFun),
    [(HasOneUse $first_output),
     (HasRankAtMost<4> $input),
     (HasRankAtMost<4> $a),
     (HasRankAtMost<4> $b)]>;
}

// We can eliminate Relu from Relu(SquaredDifference(x, y)),
// since the result of SquaredDifference is always non-negative.
// TFLite interpreter doesn't support Relu+int32 for now. So the test cases
// are failing without the following pattern to optimize Relu away fixes
// the problem.
def OptimizeReluSquaredDifference : Pat<
  (TFL_ReluOp (TFL_SquaredDifferenceOp $l, $r)),
  (TFL_SquaredDifferenceOp $l, $r)>;

// Optimize X^1 o X
def OptimizePow1ToIdentity : Pat<
  (TFL_PowOp $input,
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">)),
  (replaceWithValue $input)>;

// Optimize X^2 to X*X
def OptimizePow2ToSquare : Pat<
  (TFL_PowOp $input,
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "2.0f">)),
  (TFL_MulOp $input, $input, TFL_AF_None)>;

// Optimize X^(1/2) to √X
def OptimizePow2ToSqrt : Pat<
  (TFL_PowOp $input,
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.5f">)),
  (TFL_SqrtOp $input)>;

// Optimize X^(-1/2) to 1/√X == rsqrt(x)
def OptimizePow2ToRsqrt : Pat<
  (TFL_PowOp $input,
    (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "-0.5f">)),
  (TFL_RsqrtOp $input)>;

def CanOptimizeIdentityGatherNdOrScatterNdOp : Constraint<CPred<
  "TFL::CanOptimizeIdentityGatherNdOrScatterNdOp("
  "$0, $1.cast<DenseIntElementsAttr>(), $2.getType())">>;

def OptimizeIdentityGatherNdOp : Pat<
  (TFL_GatherNdOp:$output $params, (Arith_ConstantOp I32ElementsAttr: $indices)),
  (replaceWithValue $params),
  [(CanOptimizeIdentityGatherNdOrScatterNdOp $params, $indices, $output)]>;

def OptimizeIdentityScatterNdOp : Pat<
  (TFL_ScatterNdOp:$output (Arith_ConstantOp I32ElementsAttr: $indices), $params, $ignored),
  (replaceWithValue $params),
  [(CanOptimizeIdentityGatherNdOrScatterNdOp $params, $indices, $output)]>;

def ShapeMatchesReduceWithKeepAxes : Constraint<CPred<
  "ShapeMatchesReduceWithKeepAxes($0, $1, $2)">>;

// Fold reshapes re-inserting reduced dimensions into the results of a reduction
// with `keep_dims=false` by changing it to one using `keep_dims=true`.
foreach ReduceOp = [TFL_MeanOp, TFL_ReduceMaxOp, TFL_ReduceMinOp,
                    TFL_ReduceProdOp, TFL_SumOp] in {
  def FoldReshapeTo#ReduceOp : Pat<
    (TFL_ReshapeOp
      (ReduceOp:$reduce $input, (Arith_ConstantOp I32ElementsAttr: $axes),
                        ConstBoolAttrFalse),
      (Arith_ConstantOp I32ElementsAttr: $shape)),
    (ReduceOp $input, (Arith_ConstantOp $axes), ConstBoolAttrTrue),
    [(ShapeMatchesReduceWithKeepAxes $input, $axes, $shape),
     (HasOneUse $reduce)]>;
}


def IsSame : Constraint<CPred<"$0 == $1">>;
def HasTwoUse : Constraint<CPred<
  "std::distance($0.use_begin(), $0.use_end()) == 2">>;
def AxesIsLastDimension : Constraint<CPred<
  "$0.cast<DenseIntElementsAttr>().getNumElements() == 1 && "
  "($0.cast<DenseIntElementsAttr>().getValues<APInt>()[0] == "
  "$1.getType().cast<ShapedType>().getRank() - 1 || $0.cast<DenseIntElementsAttr>().getValues<int32_t>()[0] == -1)">>;

// Convert exp(x)/sum(exp(x)) into softmax.
def OptimizeToSoftmax : Pat<
  (TFL_DivOp (TFL_ExpOp:$exp $input),
             (TFL_SumOp:$sum $sum_input, (Arith_ConstantOp I32ElementsAttr: $axes),
                             ConstBoolAttrTrue), TFL_AF_None),
  (TFL_SoftmaxOp $input, ConstF32Attr<"1.0">),
  [(IsSame $exp, $sum_input),
   (AxesIsLastDimension $axes, $sum_input),
   (HasTwoUse $exp),
   (HasOneUse $sum)]>;

// Convert softmax(x-max(x)) into softmax(x) as the softmax op already deals
// with the max normalization.
def FoldNormalizationIntoSoftmax : Pat<
  (TFL_SoftmaxOp
    (TFL_SubOp:$sub $input,
      (TFL_ReduceMaxOp:$max $max_input, (Arith_ConstantOp I32ElementsAttr: $axes),
                            ConstBoolAttrTrue),
    TFL_AF_None),
    $beta),
  (TFL_SoftmaxOp $input, $beta),
  [(IsSame $input, $max_input),
   (AxesIsLastDimension $axes, $max_input),
   (HasOneUse $sub),
   (HasOneUse $max)]>;

// Convert softmax(x-reshape(maximum(max(x), -inf))) into softmax(x) as the softmax op already deals
// with the max normalization. This comes from upstream Jax (https://github.com/google/jax/pull/15677)
def FoldNormalizationIntoSoftmaxJaxWithAxisMinus1 : Pat<
  (TFL_SoftmaxOp
    (TFL_SubOp:$sub $input,
    (TFL_ReshapeOp:$reshape
      (TFL_MaximumOp:$maximum
        (TFL_ReduceMaxOp:$max $max_input, (Arith_ConstantOp I32ElementsAttr: $axes),
                              ConstBoolAttrFalse),
        (Arith_ConstantOp F32ElementsAttr: $threshold)
      ),
      (Arith_ConstantOp I32ElementsAttr: $shape)
    ),
    TFL_AF_None),
    $beta),
  (TFL_SoftmaxOp $input, $beta),
  [(IsSame $input, $max_input),
   (AxesIsLastDimension $axes, $max_input),
   (ConstAPFloatNegLargestOrNegInfinity $threshold),
   (HasOneUse $maximum),
   (HasOneUse $reshape),
   (HasOneUse $sub),
   (HasOneUse $max)]>;

def HaveSameType : Constraint<CPred<"($0.getType() == $1.getType())">>;

class AllElementsAreF32<string val> : Constraint<CPred<
  "($0.isa<DenseElementsAttr>() && "
   "$0.cast<DenseElementsAttr>().getType().cast<ShapedType>().getElementType().isF32() && "
   "std::all_of($0.cast<DenseElementsAttr>().getValues<float>().begin(), "
               "$0.cast<DenseElementsAttr>().getValues<float>().end(), "
               "[](float v){ return v == " #val# ";}))">>;

// Optimize X*1 to X
def OptimizeMul1ToIdentity : Pat<
  (TFL_MulOp:$result $input,
             (Arith_ConstantOp $constant),
             TFL_AF_None),
  (replaceWithValue $input),
  [(HaveSameType $input, $result),
   (AllElementsAreF32<"1.0f"> $constant)]>;

class AllElementsAreBool<string val> : Constraint<CPred<
  "($0.isa<DenseElementsAttr>() && "
   "$0.cast<DenseElementsAttr>().getType().cast<ShapedType>().getElementType().isInteger(1) && "
   "std::all_of($0.cast<DenseElementsAttr>().getValues<bool>().begin(), "
               "$0.cast<DenseElementsAttr>().getValues<bool>().end(), "
               "[](bool v){ return v == " #val# ";}))">>;

// Remove select operators when the result is known in advance.
foreach SelectOp = [TFL_SelectOp, TFL_SelectV2Op] in {
  // select(true_tensor, A, B) -> A
  def Optimize#SelectOp#True : Pat<
    (SelectOp:$result (Arith_ConstantOp $constant),
                      $input1,
                      $input2),
    (replaceWithValue $input1),
    [(HaveSameType $input1, $result),
     (AllElementsAreBool<"true"> $constant)]>;
  // select(false_tensor, A, B) -> B
  def Optimize#SelectOp#False : Pat<
    (SelectOp:$result (Arith_ConstantOp $constant),
                      $input1,
                      $input2),
    (replaceWithValue $input2),
    [(HaveSameType $input2, $result),
     (AllElementsAreBool<"false"> $constant)]>;
  // select(logical_not(C), A, B) -> select(C, B, A)
  def Optimize#SelectOp#Not : Pat<
    (SelectOp (TFL_LogicalNotOp $condition), $input1, $input2),
    (SelectOp $condition, $input2, $input1)>;
  // select(C, true_tensor, false_tensor) -> C
  def Optimize#SelectOp#IsNoop : Pat<
    (SelectOp:$result $condition,
                       (Arith_ConstantOp $input1),
                       (Arith_ConstantOp $input2)),
    (replaceWithValue $condition),
    [(HaveSameType $condition, $result),
     (AllElementsAreBool<"true"> $input1),
     (AllElementsAreBool<"false"> $input2)]>;
  // select(C, false_tensor, true_tensor) -> logical_not(C)
  def Optimize#SelectOp#IsNegate : Pat<
    (SelectOp:$result $condition,
                       (Arith_ConstantOp $input1),
                       (Arith_ConstantOp $input2)),
    (TFL_LogicalNotOp $condition),
    [(HaveSameType $condition, $result),
     (AllElementsAreBool<"false"> $input1),
     (AllElementsAreBool<"true"> $input2)]>;
}

def EliminateLogicalAndTrue : Pat<
  (TFL_LogicalAndOp:$result $lhs, (Arith_ConstantOp:$rhs $constant)),
  (replaceWithValue $lhs),
  [(AllElementsAreBool<"true"> $constant), (HaveSameType $lhs, $result)]>;

def EliminateLogicalAndFalse : Pat<
  (TFL_LogicalAndOp:$result $lhs, (Arith_ConstantOp:$rhs $constant)),
  (replaceWithValue $rhs),
  [(AllElementsAreBool<"false"> $constant), (HaveSameType $rhs, $result)]>;

def EliminateLogicalOrTrue : Pat<
  (TFL_LogicalOrOp:$result $lhs, (Arith_ConstantOp:$rhs $constant)),
  (replaceWithValue $rhs),
  [(AllElementsAreBool<"true"> $constant), (HaveSameType $rhs, $result)]>;

def EliminateLogicalOrFalse : Pat<
  (TFL_LogicalOrOp:$result $lhs, (Arith_ConstantOp:$rhs $constant)),
  (replaceWithValue $lhs),
  [(AllElementsAreBool<"false"> $constant), (HaveSameType $lhs, $result)]>;

// Remove reductions that do nothing: input and output have the same size.
foreach ReduceOp = [TFL_ReduceAnyOp, TFL_ReduceAllOp,
                    TFL_ReduceMinOp, TFL_ReduceMaxOp,
                    TFL_MeanOp, TFL_SumOp, TFL_ReduceProdOp] in {
  def EliminateNoOpReductionOp#ReduceOp : Pat<
    (ReduceOp:$output $input, (Arith_ConstantOp $index), $keep_dims),
    (replaceWithValue $input),
    [(IsTailOfShape $input, $output),
     (IsTailOfShape $output, $input)]>;
}

// Remove (log-)softmax before arg-minmax as (log-)softmax is monotonic.
foreach ArgMinMaxOp = [TFL_ArgMinOp, TFL_ArgMaxOp] in {
  def RemoveSoftmaxOpBefore#ArgMinMaxOp : Pat<
    (ArgMinMaxOp (TFL_SoftmaxOp:$softmax $logits, TFL_FloatNonNegative:$beta),
                 (Arith_ConstantOp:$const_axes I32ElementsAttr:$axes)),
    (ArgMinMaxOp $logits, $const_axes),
    [(HasOneUse $softmax),
     (AxesIsLastDimension $axes, $logits)]>;

  def RemoveLogSoftmaxOpBefore#ArgMinMaxOp : Pat<
    (ArgMinMaxOp (TFL_LogSoftmaxOp:$log_softmax $logits),
                 (Arith_ConstantOp:$const_axes I32ElementsAttr:$axes)),
    (ArgMinMaxOp $logits, $const_axes),
    [(HasOneUse $log_softmax),
     (AxesIsLastDimension $axes, $logits)]>;
}

def CanOptimizeIdentitySliceOp : Constraint<CPred<
  "TFL::CanOptimizeIdentitySliceOp($0, $1, $2)">>;

// Remove Slice ops slicing the whole input tensor, effectively no-op.
def OptimizeSliceOp : Pat<
  (TFL_SliceOp:$output $input, (Arith_ConstantOp $begin), (Arith_ConstantOp $size)),
  (replaceWithValue $input),
  [(CanOptimizeIdentitySliceOp $input, $begin, $size)]>;

// Convert the StridedSliceOp to a SliceOp when possible. This will enable other
// optimizations on SliceOp to run.
def OptimizeStridedSlice : Pat<
  (TFL_StridedSliceOp $input,
    (Arith_ConstantOp $begin),
    (Arith_ConstantOp $end),
    (Arith_ConstantOp $stride),
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ZeroIntAttr:$_,
    ConstBoolAttrFalse),
  (TFL_SliceOp $input,
    (Arith_ConstantOp $begin),
    (Arith_ConstantOp (GetOffSet $begin, $end))),
  [(IsAllOnesConstant $stride),
   (HasNonNegativeValues $begin),
   (HasNonNegativeOffset $begin, $end)]>;

def GetNumElementsOrOne: NativeCodeCall<"GetNumElementsOrOne($0.getType())">;

def ReshapeValueDroppingLastDim : NativeCodeCall<
  "ReshapeValueDroppingLastDim($_builder, $0)">;

def IsOneHotIndexAttribute : Constraint<CPred<
  "TFL::IsOneHotIndexAttribute($0)">>;

// Checks if the shape has static shape with last dimension equals 1.
def IsLastDimensionEqualOne : Constraint<CPred<"IsLastDimensionEqualOne($0)">>;

// As above but if shape is not static and rank 2 with last dim 1.
def IsLastDimensionEqualOneOrDynamicBatchDimRank2 : Constraint<
  CPred<"IsLastDimensionEqualOne($0) || "
        "(!$0.getType().cast<ShapedType>().hasStaticShape() && "
        "  $0.getType().cast<ShapedType>().hasRank() && "
        "  $0.getType().cast<ShapedType>().getRank() == 2 && "
        "  !$0.getType().cast<ShapedType>().getShape().empty() && "
        "  $0.getType().cast<ShapedType>().getShape()[1] == 1)">>;

// Replace
//   Equal(X, indices)
// With
//   OneHot(Reshape(X), N, true, false, -1)
// where
//  - last dimension of the LHS of the equal is 1, and the rank is at least 2.
//  - indices is a incrementing series from 0 to N-1. (N elements total.)
def ReshapeEqualOpToOneHotOp : Pat<
  (TFL_EqualOp $x, (Arith_ConstantOp $series)),
  (TFL_OneHotOp (ReshapeValueDroppingLastDim $x),
                (Arith_ConstantOp (GetNumElementsOrOne $series)),
                (Arith_ConstantOp ConstantAttr<RankedSignlessIntElementsAttr<1, []>, "true">),
                (Arith_ConstantOp ConstantAttr<RankedSignlessIntElementsAttr<1, []>, "false">),
                ConstantAttr<I32Attr, "-1">),
  [(IsLastDimensionEqualOneOrDynamicBatchDimRank2 $x),
   (HasRankAtLeast<2> $x),
   (IsOneHotIndexAttribute $series)]>;

def F32ElementsVal : Constraint<CPred<
  "$0.getType().cast<TensorType>().getElementType().isF32()">,
  "32 bit float tensor">;
def I32ElementsVal : Constraint<CPred<
  "$0.getType().cast<TensorType>().getElementType().isInteger(32)">,
  "32 bit integer tensor">;

def ConvertSingleElementAttrToFloatAttr :
  NativeCodeCall<"ConvertSingleElementAttrToFloatAttr($0)">;

// Replace
//   (float)OneHot(index, depth, on_val, off_val, axis)
// With
//   OneHot(index, depth, (float)on_val, (float)off_val, axis)
def FuseOneHotAndCastToFloat : Pat<
  (TFL_CastOp:$output (TFL_OneHotOp $indices,
                                    $depth,
                                    (Arith_ConstantOp $on_val),
                                    (Arith_ConstantOp $off_val),
                                    $axis)),
  (TFL_OneHotOp $indices,
                $depth,
                (Arith_ConstantOp (ConvertSingleElementAttrToFloatAttr $on_val)),
                (Arith_ConstantOp (ConvertSingleElementAttrToFloatAttr $off_val)),
                $axis),
  [(F32ElementsVal $output)]>;

def Get1DShapeValue: NativeCodeCall<"Get1DShapeValue($_builder, $0)">;

class GetIthValue<int index> : NativeCodeCall<"$0[" # index # "]">;

def GetEmbeddingLookupShape: NativeCodeCall<"GetEmbeddingLookupShape($0, $1)">;

// Replace
//   OneHot(index, depth, on=1.0f, off=0.0f, axis=-1) * filter
// With
//   EmbeddingLookup(index, Transpose(filter))
//
// OneHot with on=1 off=0 axis=-1, where `index` is a single element tensor,
// creates a tensor of size depth, and all values are 0, except for the element
// at `index`, which is 1. Multiplying such a tensor with a 2D filter esentially
// returns the single column in filter as a 1D tensor. If the input has multiple
// elements, repeat this for every entry, forming the higher dimensions in the
// result tensor. For instance, if:
//   input = [1, 2]
//   depth = 4
//   filter = [[5, 7, 11, 13], [17, 19, 23, 29]]
// then:
//   onehot = [[0, 1, 0, 0], [0, 0, 1, 0]]
//   result = [[ 7, 19],   # == 1st column in filter
//             [11, 23]]   # == 2nd column in filter
// This is exactly what the EmbeddedLookup operator is doing, on the transposed
// matrix, without doing any arithmetic but only memcpy.
def ReplaceOneHotFullyConnectedWithLookup : Pat<
  (TFL_FullyConnectedOp:$outputs
    (TFL_OneHotOp
      AnyStaticShapeTensor:$indices,
      (Arith_ConstantOp $depth),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.0f">),
      ConstantAttr<I32Attr, "-1">),
    StaticShapeTensorOf<[F32, I8, UI8]>:$filter,
    $bias,
    TFL_AF_None,
    TFL_FCWO_Default,
    $keep_num_dims,
    $asymmetric_quantize_inputs),
  (TFL_ReshapeOp
  (TFL_EmbeddingLookupOp
      (TFL_ReshapeOp $indices, (Get1DShapeValue $indices)),
    (TFL_TransposeOp
      $filter,
        (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[2]>, "{1,0}">)),
      (returnType (GetEmbeddingLookupShape $indices, $filter))
    ),
    (Arith_ConstantOp (GetShape (GetIthValue<0> $outputs)))),
  [(I32ElementsVal $indices),     // lookup is not implemented for i64
   (IsNoneType $bias)]>;          // Maybe folded into the lookup matrix later

// Same as above, but if indices are already 1D, then no reshape is needed.
def ReplaceOneHotDynamicShapeFullyConnectedWithLookup : Pat<
  (TFL_FullyConnectedOp:$outputs
    (TFL_OneHotOp
      1DTensorOf<[I32]>:$indices,
      (Arith_ConstantOp $depth),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">),
      (Arith_ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.0f">),
      ConstantAttr<I32Attr, "-1">),
    StaticShapeTensorOf<[F32, I8, UI8]>:$filter,
    $bias,
    TFL_AF_None,
    TFL_FCWO_Default,
    $keep_num_dims,
    $asymmetric_quantize_inputs),
  (TFL_EmbeddingLookupOp $indices,
    (TFL_TransposeOp
      $filter,
        (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[2]>, "{1,0}">))),
  [(I32ElementsVal $indices),
   (IsNoneType $bias)]>;

def AreInputDimensionsOneInAxes : Constraint<CPred<
  "AreInputDimensionsOneInAxes($0, $1)">>;

// Eliminate cumulative summations if the input's dimension in axis is 1.
def EliminateCumSumInclusive : Pat<
  (TFL_CumsumOp
     $input,
     (Arith_ConstantOp I32ElementsAttr:$axis),
     ConstBoolAttrFalse,
     $reverse),
  (replaceWithValue $input),
  [(AreInputDimensionsOneInAxes $input, $axis)]>;

// Fusing raw computation of GELU op into one native tfl_gelu op.
//
// Requires constants to be exact match and only one use of all of the
// intermediate results.
//
// For GeluApproximate, replaces
//   0.5 * x * ( 1 + tanh( sqrt_2dPi  * ( x + 0.044715 * pow( x, 3 ) ) ) )
def MatchGeluApproximate : Pat<
  (TFL_MulOp
   (TFL_MulOp:$mul_out $arg0, (Arith_ConstantOp F32ElementsAttr:$Cst_1_2), TFL_AF_None),
   (TFL_AddOp:$add_out
    (TFL_TanhOp:$tanh_out
     (TFL_MulOp:$mul_out1
      (TFL_AddOp:$add_out1 $arg0,
       (TFL_MulOp:$mul_out2
        (TFL_PowOp:$pow_out $arg0, (Arith_ConstantOp F32ElementsAttr:$Cst_3)),
        (Arith_ConstantOp F32ElementsAttr:$Coeff), TFL_AF_None), TFL_AF_None),
      (Arith_ConstantOp F32ElementsAttr:$Cst_sqrt_2dPi), TFL_AF_None)),
    (Arith_ConstantOp F32ElementsAttr:$Cst_1), TFL_AF_None), TFL_AF_None),
  (TFL_GeluOp $arg0, ConstBoolAttrTrue),
  [(FloatValueEquals<"0.5"> $Cst_1_2),
   (FloatValueEquals<"1"> $Cst_1),
   (FloatValueEquals<"3"> $Cst_3),
   (FloatValueEquals<"0.797884583"> $Cst_sqrt_2dPi),
   (FloatValueEquals<"0.044715"> $Coeff),
   (HasOneUse $mul_out),
   (HasOneUse $add_out),
   (HasOneUse $tanh_out),
   (HasOneUse $mul_out1),
   (HasOneUse $add_out1),
   (HasOneUse $mul_out2),
   (HasOneUse $pow_out),
  ]>;

// Alternate pattern for GeluApproximate (see different order for mul), replaces
//   x * ( 0.5 * ( 1 + tanh( sqrt_2dPi  * ( x + 0.044715 * pow( x, 3 ) ) ) ) )
def MatchGeluApproximate1 : Pat<
  (TFL_MulOp $arg0,
   (TFL_MulOp:$mul_out
    (TFL_AddOp:$add_out
     (TFL_TanhOp:$tanh_out
      (TFL_MulOp:$mul_out1
       (TFL_AddOp:$add_out1 $arg0,
        (TFL_MulOp:$mul_out2
         (TFL_PowOp:$pow_out $arg0, (Arith_ConstantOp F32ElementsAttr:$Cst_3)),
         (Arith_ConstantOp F32ElementsAttr:$Coeff), TFL_AF_None), TFL_AF_None),
       (Arith_ConstantOp F32ElementsAttr:$Cst_sqrt_2dPi), TFL_AF_None)),
     (Arith_ConstantOp F32ElementsAttr:$Cst_1), TFL_AF_None), (Arith_ConstantOp F32ElementsAttr:$Cst_1_2), TFL_AF_None), TFL_AF_None),
  (TFL_GeluOp $arg0, ConstBoolAttrTrue),
  [(FloatValueEquals<"0.5"> $Cst_1_2),
   (FloatValueEquals<"1"> $Cst_1),
   (FloatValueEquals<"3"> $Cst_3),
   (FloatValueEquals<"0.797884583"> $Cst_sqrt_2dPi),
   (FloatValueEquals<"0.044715"> $Coeff),
   (HasOneUse $mul_out),
   (HasOneUse $add_out),
   (HasOneUse $tanh_out),
   (HasOneUse $mul_out1),
   (HasOneUse $add_out1),
   (HasOneUse $mul_out2),
   (HasOneUse $pow_out),
  ]>;

// For Gelu, replaces
//   0.5 * x * ( 1 + erf( x * sqrt_1_2 ) )
def MatchGelu : Pat<
  (TFL_MulOp
   (TFL_MulOp:$mul_out $arg0, (Arith_ConstantOp F32ElementsAttr:$Cst_1_2), TFL_AF_None),
   (TFL_AddOp:$add_out
    (TF_ErfOp:$erf_out
     (TFL_MulOp:$mul_out1 $arg0, (Arith_ConstantOp F32ElementsAttr:$Cst_sqrt_1_2), TFL_AF_None)),
    (Arith_ConstantOp  F32ElementsAttr:$Cst_1), TFL_AF_None), TFL_AF_None),
  (TFL_GeluOp $arg0, ConstBoolAttrFalse),
  [(FloatValueEquals<"0.5"> $Cst_1_2),
   (FloatValueEquals<"1"> $Cst_1),
   (FloatValueEquals<"0.707106769"> $Cst_sqrt_1_2),
   (HasOneUse $mul_out),
   (HasOneUse $add_out),
   (HasOneUse $erf_out),
   (HasOneUse $mul_out1),
  ]>;

// For Gelu, replaces
//   0.5 * x * ( erfc( -x * sqrt_1_2 ) )
def MatchGeluWithErfc : Pat<
  (TFL_MulOp
   (TFL_MulOp:$mul_out $arg0, (Arith_ConstantOp F32ElementsAttr:$Cst_1_2), TFL_AF_None),
   (TF_ErfcOp:$erfc_out
    (TFL_MulOp:$mul_out1
     (TFL_NegOp $arg0),
     (Arith_ConstantOp F32ElementsAttr:$Cst_sqrt_1_2), TFL_AF_None)), TFL_AF_None),
  (TFL_GeluOp $arg0, ConstBoolAttrFalse),
  [(FloatValueEquals<"0.5"> $Cst_1_2),
   (FloatValueEquals<"0.707106769"> $Cst_sqrt_1_2),
   (HasOneUse $mul_out),
   (HasOneUse $erfc_out),
   (HasOneUse $mul_out1),
  ]>;

// Fetches the output of FC op, from the provided arguments.
def GetFcOutput : NativeCodeCall<
  "GetFcOutput(&$_builder, $0, $1, $2, $3, $4, $5, $6, $7)">;

// Verifies all values in the provided argument are zero.
def AllValuesAreZero :  Constraint<CPred<"AllValuesAreZero($0)">>;

def SimplifyDoubleSelectFCZerosLHS : Pat<
  (TFL_SelectV2Op $condition, $zeros_2,
   (TFL_FullyConnectedOp:$results
    (TFL_SelectV2Op $condition, $zeros_1, $input),
    $filter, $bias, $fused_activation_function, $weights_format,
    ConstBoolAttrTrue, $asymmetric_quantize_inputs)),
  (TFL_SelectV2Op $condition, $zeros_2,
   (GetFcOutput $results, $input, $filter, $bias, $fused_activation_function,
    $weights_format, ConstBoolAttrTrue, $asymmetric_quantize_inputs)),
  [(IsLastDimensionEqualOne $condition),
   (AllValuesAreZero $zeros_1),
   (AllValuesAreZero $zeros_2)
  ]>;

def SimplifyDoubleSelectFCZerosRHS : Pat<
  (TFL_SelectV2Op $condition,
   (TFL_FullyConnectedOp:$results
    (TFL_SelectV2Op $condition, $input, $zeros_1),
    $filter, $bias, $fused_activation_function, $weights_format,
    ConstBoolAttrTrue, $asymmetric_quantize_inputs),
   $zeros_2),
  (TFL_SelectV2Op $condition,
   (GetFcOutput $results, $input, $filter, $bias, $fused_activation_function,
    $weights_format, ConstBoolAttrTrue, $asymmetric_quantize_inputs),
   $zeros_2),
  [(IsLastDimensionEqualOne $condition),
   (AllValuesAreZero $zeros_1),
   (AllValuesAreZero $zeros_2)
  ]>;

def FuseSigmoid : Pat<
  (TFL_DivOp
   (Arith_ConstantOp F32ElementsAttr:$ones),
   (TFL_AddOp:$add_out
    (TFL_ExpOp:$exp_out
     (TFL_NegOp:$neg_out $arg)),
    (Arith_ConstantOp F32ElementsAttr:$ones_1), TFL_AF_None), TFL_AF_None),
  (TFL_LogisticOp $arg),
  [(FloatValueEquals<"1"> $ones_1),
   (FloatValueEquals<"1"> $ones),
   (HasOneUse $neg_out),
   (HasOneUse $add_out),
   (HasOneUse $exp_out),
  ]>;

def FuseEluConst : Pat<
  (TFL_SelectOp
   (TFL_GreaterOp $arg, (Arith_ConstantOp F32ElementsAttr:$zero)),
   $arg,
   (TFL_SubOp:$sub_out
    (TFL_ExpOp:$exp_out
     (TFL_SelectOp:$select_out
      (TFL_GreaterOp $arg, (Arith_ConstantOp F32ElementsAttr:$zero_1)),
      (Arith_ConstantOp F32ElementsAttr:$zero_2),
      $arg)),
    (Arith_ConstantOp F32ElementsAttr:$one),
    TFL_AF_None)),
  (TFL_EluOp $arg),
  [
    (FloatValueEquals<"0"> $zero_2),
    (FloatValueEquals<"0"> $zero_1),
    (FloatValueEquals<"0"> $zero),
    (FloatValueEquals<"1"> $one),
    (HasOneUse $select_out),
    (HasOneUse $exp_out),
    (HasOneUse $sub_out),
  ]>;


def isF32Splat : Constraint<
  CPred<"IsF32Splat($0)">>;

def ExtractF32AtIndex0: NativeCodeCall<
    "$_builder.getF32FloatAttr($_self.cast<DenseElementsAttr>().getValues<float>()[0])">;

def FuseLeakyReluConst : Pat<
  (TFL_SelectOp
   (TFL_GreaterEqualOp:$geq_out $arg, (Arith_ConstantOp F32ElementsAttr:$threshold)),
   $arg,
   (TFL_MulOp:$mul_out $arg, (Arith_ConstantOp F32ElementsAttr:$alpha), TFL_AF_None)),
  (TFL_LeakyReluOp $arg, ExtractF32AtIndex0:$alpha),
  [
    (FloatValueEquals<"0"> $threshold),
    (isF32Splat $alpha),
    (HasOneUse $geq_out),
    (HasOneUse $mul_out),
  ]>;

// Return true if the product of dimension values of a subsection of the tensor
// is equal to the non-contracting dimension after a reshape
class NonBroadcastingNonContractingLhsDimsProductEqual<int agg_start_idx, int agg_end_idx=0> : Constraint<CPred<
  "TFL::NonBroadcastingNonContractingDimsProductEqual($0, $1, true, "# agg_start_idx #","# agg_end_idx #")">>;

class NonBroadcastingNonContractingRhsDimsProductEqual<int agg_start_idx, int agg_end_idx=0> : Constraint<CPred<
  "TFL::NonBroadcastingNonContractingDimsProductEqual($0, $1, false, "# agg_start_idx #","# agg_end_idx #")">>;

// Predicate to check if the last dimensions of two values is equal.
def TrailingDimValuesEqual : Constraint<
  CPred<"mlir::cast<ShapedType>($0.getType()).getShape().back()"
    "== mlir::cast<ShapedType>($1.getType()).getShape().back()">>;

// Predicate to check if the product of last few dimensions in LHS is equal to
// the last dimension in RHS.
class ContractingDimsProductEqual<int agg_start_idx> : Constraint<CPred<
  "TFL::ContractingDimsProductEqual($0, $1, "# agg_start_idx #")">>;

// Returns true if the dimensions of a subsection of two tensors is equal
class AreTensorSubSectionShapesEqual<int skip_first, int skip_last> : Constraint<CPred<
  "($0.getType().dyn_cast<ShapedType>().getShape()"
    ".drop_back("#skip_last#").drop_front("#skip_first#") =="
  "$1.getType().dyn_cast<ShapedType>().getShape()"
    ".drop_back("#skip_last#").drop_front("#skip_first#"))">>;

// Returns true if the broadcast dimension of a tensor is [1]
// here- broadcast dimension is first prefix dimension
// excluding the last two dimensions
def IsBroadcastDimEqualToOne : Constraint<CPred<
  "$0.getType().dyn_cast<ShapedType>().getShape()[0] == 1">>;

// Pattern to fuse/fold the reshape ops around TFL_BatchMatMulOp
// This pattern is applied when the rank of rhs is 2
// which means it has empty broadcast dimensions
def FuseReshapesAroundBatchMatMulLHS: Pat<
  (TFL_ReshapeOp:$final_shape_change
    (TFL_BatchMatMulOp:$bmm_tmp_output
      (TFL_ReshapeOp:$initial_shape_change $input, (Arith_ConstantOp $s0)),
      $rhs, $adj_x, $adj_y, $bool_attr),
    (Arith_ConstantOp $s1)),
  (TFL_BatchMatMulOp $input, $rhs, $adj_x, $adj_y, $bool_attr),
  [(HasRankAtLeast<2> $input),
   (HasRankAtLeast<2> $final_shape_change),
   (HasRank<2> $rhs),
   (HasRank<2> $initial_shape_change),
   (AnyStaticShapeTensor $input),
   (AnyStaticShapeTensor $initial_shape_change),
   (AnyStaticShapeTensor $final_shape_change),
   (TrailingDimValuesEqual $input, $initial_shape_change),
   (AreTensorSubSectionShapesEqual<0, 1> $input, $final_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<0> $input, $initial_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<0> $final_shape_change, $bmm_tmp_output)]>;

// Pattern to fuse/fold the reshape of TFL_BatchMatMulOp output to expand the
// dimensions of the output to add a unity broadcast dimension.
// This pattern assumes that the input has more than one contracting dimensions.
//
// This pattern is applied when-
// 1. The rank of rhs is 2
// 2. The original input reshape has a) reduction in leading broadcast dim and
//    b) flattening of the contracting dims.
// 3. non-broadcasting, non-contracting dims of the rhs are not flattened.
def FuseOutputReshape_BatchMatMulWithFlattenedContractingDims: Pat<
  (TFL_ReshapeOp:$final_shape_change
    (TFL_BatchMatMulOp:$bmm_tmp_output
      (TFL_ReshapeOp:$initial_shape_change
        $input, (Arith_ConstantOp I32ElementsAttr:$s0)),
      $rhs, $adj_x, $adj_y, $bool_attr),
    (Arith_ConstantOp $s1)),
  (TFL_BatchMatMulOp
    (TFL_ReshapeOp $input,
      (Arith_ConstantOp (GetExpandedShapeAttr<1> $initial_shape_change))),
    $rhs, $adj_x, $adj_y, $bool_attr),
  [(HasRankAtLeast<4> $input),
   (HasRank<2> $rhs),
   (HasRank<2> $initial_shape_change),
   (AnyStaticShapeTensor $input),
   (AnyStaticShapeTensor $initial_shape_change),
   (AnyStaticShapeTensor $final_shape_change),
   (IsBroadcastDimEqualToOne $input),
   (IsBroadcastDimEqualToOne $final_shape_change),
   (TrailingDimValuesEqual $bmm_tmp_output, $final_shape_change),
   (ContractingDimsProductEqual<2> $input, $initial_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<0,1> $input, $initial_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<0,1> $final_shape_change, $bmm_tmp_output)]>;

// Pattern to fuse/fold the reshape of TFL_BatchMatMulOp input.
//
// This pattern is applied if-
// 1. LHS has only one non-broadcasting, non-contracting dimension
// 2. No batching dimensions on LHS and RHS
// 3. RHS has a flattened non-broadcasting, non-contracting dimension
def FuseInputReshape_BatchMatMulWithFlattenedRhsDims: Pat<
  (TFL_ReshapeOp:$final_shape_change
    (TFL_BatchMatMulOp:$bmm_tmp_output
      (TFL_ReshapeOp:$initial_shape_change
        $input, (Arith_ConstantOp I32ElementsAttr:$s0)),
      $rhs, $adj_x, $adj_y, $bool_attr),
    (Arith_ConstantOp I32ElementsAttr:$s1)),
  (TFL_ReshapeOp
    (TFL_BatchMatMulOp $input, $rhs, $adj_x, $adj_y, $bool_attr,
      (returnType (GetExpandedShapeType<1> $bmm_tmp_output))),
    (Arith_ConstantOp $s1)),
  [(HasRank<2> $rhs),
   (HasRank<3> $input),
   (HasRank<2> $initial_shape_change),
   (AnyStaticShapeTensor $input),
   (AnyStaticShapeTensor $initial_shape_change),
   (AnyStaticShapeTensor $final_shape_change),
   (IsBroadcastDimEqualToOne $input),
   (TrailingDimValuesEqual $input, $initial_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<0,1> $input, $initial_shape_change),
   (NonBroadcastingNonContractingRhsDimsProductEqual<2> $final_shape_change, $bmm_tmp_output)]>;

// Pattern to fuse/fold the reshape ops around TFL_BatchMatMulOp
// This pattern is applied when the rank of rhs is 3
// and the broadcast dimension is [1]
def FuseReshapesAroundBatchMatMulLHS1: Pat<
  (TFL_ReshapeOp:$final_shape_change
    (TFL_BatchMatMulOp:$bmm_tmp_output
      (TFL_ReshapeOp:$initial_shape_change $input, (Arith_ConstantOp $s0)),
      $rhs, $adj_x, $adj_y, $bool_attr),
    (Arith_ConstantOp $s1)),
  (TFL_BatchMatMulOp $input, $rhs, $adj_x, $adj_y, $bool_attr),
  [(HasRankAtLeast<3> $input),
   (HasRank<3> $rhs),
   (HasRank<3> $initial_shape_change),
   (AnyStaticShapeTensor $input),
   (AnyStaticShapeTensor $initial_shape_change),
   (AnyStaticShapeTensor $final_shape_change),
   (IsBroadcastDimEqualToOne $rhs),
   (IsBroadcastDimEqualToOne $input),
   (TrailingDimValuesEqual $input, $initial_shape_change),
   (AreTensorSubSectionShapesEqual<1, 1> $input, $final_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<1> $input, $initial_shape_change),
   (NonBroadcastingNonContractingLhsDimsProductEqual<1> $final_shape_change, $bmm_tmp_output)]>;



// Fuse redundant TFL_TransposeOp into TFL_BatchMatMulOp
def FuseTransposeIntoBatchMatMulLHS: Pat<
  (TFL_BatchMatMulOp
    (TFL_TransposeOp:$transposed_value $input, (Arith_ConstantOp:$perm_value $p0)),
    $rhs, $adj_x, $adj_y, $asymmetric_quantize_inputs),
  (TFL_BatchMatMulOp $input, $rhs, ConstBoolAttrTrue, $adj_y, $asymmetric_quantize_inputs),
  [(AreLastTwoDimsTransposed $perm_value),
   (IsBoolAttrEqual<"false"> $adj_x)]>;

// Fuses TFL_TransposeOp after TFL_BatchMatMulOp into the BMM.
def FuseTransposeAfterBatchMatmul : Pat<
  (TFL_TransposeOp
    (TFL_BatchMatMulOp $lhs, $rhs, ConstBoolAttrFalse, ConstBoolAttrTrue,
                       $asymmetric_quantize_inputs
    ),
    (Arith_ConstantOp:$perm_value $p0)
  ),
  (TFL_BatchMatMulOp $rhs, $lhs, ConstBoolAttrFalse, ConstBoolAttrTrue,
                     $asymmetric_quantize_inputs
  ),
  [(AreLastTwoDimsTransposed $perm_value)]>;

// Fuse redundant RHS TFL_TransposeOp into TFL_BatchMatMulOp if rhs is any
// tensor of rank-2.
def FuseTransposeIntoBatchMatMulRHS: Pat<
  (TFL_BatchMatMulOp $lhs,
    (TFL_TransposeOp $input, (Arith_ConstantOp:$perm_value $p0)),
    $adj_x, $adj_y, $asymmetric_quantize_inputs),
  (TFL_FullyConnectedOp
    $lhs,
    $input, (CreateNoneValue $lhs), TFL_AF_None, TFL_FCWO_Default,
    ConstBoolAttrTrue, $asymmetric_quantize_inputs),
  [(HasRank<2> $input),
   (AreLastTwoDimsTransposed $perm_value),
   (IsBoolAttrEqual<"false"> $adj_x),
   (IsBoolAttrEqual<"false"> $adj_y)]>;

def FuseTransposeIntoBatchMatMulRHSAdjY: Pat<
  (TFL_BatchMatMulOp
    $lhs,
    (TFL_TransposeOp:$transposed_value $input, (Arith_ConstantOp:$perm_value $p0)),
    $adj_x, $adj_y, $asymmetric_quantize_inputs),
  (TFL_BatchMatMulOp $lhs, $input, $adj_x, ConstBoolAttrTrue, $asymmetric_quantize_inputs),
  [(AreLastTwoDimsTransposed $perm_value),
   (HasOneUse $transposed_value),
   (IsBoolAttrEqual<"false"> $adj_y)]>;

// Fuse squeezing LHS reshape into tfl.fully_connected.
// This will work for any FullyConnected(Reshape(LHS), RHS), provided-
// 1.The reshape is only squeezing the broadcast dimensions of the LHS.
// 2. Rank after reshape is 2.
// 3. keep_num_dims=True
// Ex- [1,  128, 32] -> [128, 32] or [1, 1, 1, 128, 32] -> [128, 32]
//
// We should be able to set keep_num_dims=False and fold the reshape.
def FuseSqueezingLhsReshapeIntoFC_Output: Pat<
  (TFL_FullyConnectedOp:$results
    (TFL_ReshapeOp:$reshaped_input $input, (Arith_ConstantOp $s)),
    $filter, $bias, $fused_activation_function, $weights_format,
    $keep_num_dims, $asymmetric_quantize_inputs),
  (TFL_FullyConnectedOp
    $input, $filter, $bias, $fused_activation_function, $weights_format,
    ConstBoolAttrFalse, $asymmetric_quantize_inputs),
  [(IsBoolAttrEqual<"true"> $keep_num_dims),
   (AnyStaticShapeTensor $input),
   (AnyStaticShapeTensor $reshaped_input),
   (HasRankAtLeast<3> $input),
   (HasRank<2> $reshaped_input),
   (HasRank<2> $filter),
   (IsReducedTailOfShape $input, $reshaped_input)]>;

// Replace conv-->transpose-->add with conv-->add-->transpose
// The bias needs only reshape (i.e. ReshapeNCHWBiasToNHWC) and not transpose
// because the bias's shape simply changes from NxCx1x1 to Nx1x1xC.
def ReorderNCHWTransposeAdd : Pat <
  (TFL_AddOp:$add_op
    (TFL_TransposeOp:$transpose_op $input, $perm),
    (Arith_ConstantOp:$bias_op FloatElementsAttr:$bias),
    $act_fn),
  (TFL_TransposeOp
    (TFL_AddOp $input,
      (Arith_ConstantOp (ReshapeNCHWBiasToNHWC $bias_op, $bias)),
      $act_fn),
    $perm),
  [(IsPermutationNCHW $perm),
  (IsBiasShape $bias_op),
  (IsDefinedByConv2DOp $input),
  (HasOneUse $add_op),
  (HasOneUse $transpose_op)]>;


// TODO(weiyiw): Also support variant length arguments
def FuseSliceAndPack : Pat<(
  TFL_PackOp
    (variadic
      (TFL_ReshapeOp
        (TFL_SliceOp $input0,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"0">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_),
      (TFL_ReshapeOp
        (TFL_SliceOp $input1,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_),
      (TFL_ReshapeOp
        (TFL_SliceOp $input2,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"2">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_)
    ), $_, $_), (replaceWithValue $input0),
    [(IsSame $input0, $input1), (IsSame $input0, $input2)]>;

def FuseSliceAndPack4D : Pat<(
  TFL_PackOp
    (variadic
      (TFL_ReshapeOp
        (TFL_SliceOp $input0,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"0">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_),
      (TFL_ReshapeOp
        (TFL_SliceOp $input1,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_),
      (TFL_ReshapeOp
        (TFL_SliceOp $input2,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"2">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_),
      (TFL_ReshapeOp
        (TFL_SliceOp $input3,
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"3">),
          (Arith_ConstantOp ConstantAttr<RankedI32ElementsAttr<[1]>,"1">)), $_)
    ), $_, $_), (replaceWithValue $input0),
    [(IsSame $input0, $input1), (IsSame $input0, $input2), (IsSame $input0, $input3)]>;

// Given a value, checks if dim `d` is static.
class HasStaticDim<int d> : Constraint<CPred<
  "!$0.getType().cast<ShapedType>().isDynamicDim(" # d # ")">>;

class IsBalancedPaddingArray<int spatials_start, int spatials_end> :
  Constraint<CPred<
    "IsBalancedPaddingArray("# spatials_start #","# spatials_end #","
      "$0.cast<DenseElementsAttr>())">>;

// Given in_shape, out_shape, stride checks ceil(in_shape[d] / stride) == out_shape[d]
def IsSameStridedShape2D : Constraint<CPred<
  "HasSameStridedShape($0.getDefiningOp<TFL::Conv2DOp>(),"
    "$1.getType().cast<ShapedType>().getShape())">>;

def IsSameStridedShapeDepthwise : Constraint<CPred<
  "HasSameStridedShape($0.getDefiningOp<TFL::DepthwiseConv2DOp>(),"
    "$1.getType().cast<ShapedType>().getShape())">>;

def IsSameStridedShape3D : Constraint<CPred<
  "HasSameStridedShape($0.getDefiningOp<TFL::Conv3DOp>(),"
    "$1.getType().cast<ShapedType>().getShape())">>;

def IsValidPadding : Constraint<CPred<"$0.str() == \"VALID\"">>;

// Fuse explicit tfl.pad ops into standard convolutions when it implies "SAME"
// padding. "SAME" padding is defined to be any non-trivial padding where
// ceil(in_dim_i / stride_i) == out_dim_i
// and 0 <= (pad_i_hi - pad_i_lo) <= 1 for all spatial dims i.

def FuseSamePaddingConv2D : Pat<
  (TFL_Conv2DOp:$conv_out
    (TFL_PadOp $input, (Arith_ConstantOp $paddings)),
    $filter,
    $bias,
    $h_dilate,
    $w_dilate,
    $faf,
    $padding,
    $stride_h,
    $stride_w
  ), (TFL_Conv2DOp
    $input,
    $filter,
    $bias,
    $h_dilate,
    $w_dilate,
    $faf,
    TFL_PAD_Same,
    $stride_h,
    $stride_w
  ),
  [(HasStaticDim<1> $input),
    (HasStaticDim<2> $input),
    (IsBalancedPaddingArray<1, 3> $paddings),
    (IsValidPadding $padding),
    (IsSameStridedShape2D $conv_out, $input)]>;

def FuseSamePaddingDepthwiseConv : Pat<
  (TFL_DepthwiseConv2DOp:$conv_out
    (TFL_PadOp $input, (Arith_ConstantOp $paddings)),
    $filter,
    $bias,
    $h_dilate,
    $w_dilate,
    $faf,
    $padding,
    $stride_h,
    $stride_w,
    $depth
  ), (TFL_DepthwiseConv2DOp
    $input,
    $filter,
    $bias,
    $h_dilate,
    $w_dilate,
    $faf,
    TFL_PAD_Same,
    $stride_h,
    $stride_w,
    $depth
  ),
  [(HasStaticDim<1> $input),
    (HasStaticDim<2> $input),
    (IsBalancedPaddingArray<1, 3> $paddings),
    (IsValidPadding $padding),
    (IsSameStridedShapeDepthwise $conv_out, $input)]>;

def FuseSamePaddingConv3D : Pat<
  (TFL_Conv3DOp:$conv_out
    (TFL_PadOp $input, (Arith_ConstantOp $paddings)),
    $filter,
    $bias,
    $d_dilate,
    $h_dilate,
    $w_dilate,
    $faf,
    $padding,
    $stride_d,
    $stride_h,
    $stride_w
  ), (TFL_Conv3DOp
    $input,
    $filter,
    $bias,
    $d_dilate,
    $h_dilate,
    $w_dilate,
    $faf,
    TFL_PAD_Same,
    $stride_d,
    $stride_h,
    $stride_w
  ),
  [(HasStaticDim<1> $input),
    (HasStaticDim<2> $input),
    (HasStaticDim<3> $input),
    (IsBalancedPaddingArray<1, 4> $paddings),
    (IsValidPadding $padding),
    (IsSameStridedShape3D $conv_out, $input)]>;

// Replace
//   Gather(Cast(input), indices)
// With
//   Cast(Gather(input, indices))
// This reduces the number of tensor elements that need to be converted.
def ReorderGatherAndCast : Pat<
  (TFL_GatherOp (TFL_CastOp:$cast $params), $indices, $axis, $batch_dims),
  (TFL_CastOp (TFL_GatherOp $params, $indices, $axis, $batch_dims)),
  [(HasOneUse $cast)]>;

// Replace division by a constant with a multiplication by a reciprocal of that
// constant. Floating point division can be ~10x more expensive than a
// multiplication.
def RealDivWithF32ConstDivisor : Pat<
  (TFL_DivOp:$src $arg0, (Arith_ConstantOp FloatElementsAttr<32>:$value), $activation),
  (TFL_MulOp:$dest1 $arg0,
    (TFL_DivOp (Arith_ConstantOp
      (GetScalarOfType<1> (Arith_ConstantOp $value))),
      (Arith_ConstantOp $value),  TFL_AF_None),
    $activation)>;


// Fuse Sum -> Mul into Mean if the  RHS of Mul is a constant equals to scale 
// where scale = 1.0 / (product of the summed dimensions that are part of the sum op).
def FuseSumMulIntoMean: Pat<
  (TFL_MulOp:$src
    (TFL_SumOp:$sum $sum_input, (Arith_ConstantOp I32ElementsAttr: $axes),
                             $keep_dims),
    (Arith_ConstantOp FloatElementsAttr:$scale), TFL_AF_None),
  (TFL_MeanOp $sum_input, (Arith_ConstantOp $axes), $keep_dims),
  [(IsScaleOfSum $sum_input, $axes, $scale)]>;