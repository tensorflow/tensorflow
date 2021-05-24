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

// This is the optimization pattern definition file for TensorFlow Lite.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/lite/utils/utils.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Checks if the param passed is a F32 ElementsAttr.
def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<ElementsAttr>() && $_self.cast<ElementsAttr>().getType().getElementType().isF32()">,
        "32 bit float constant tensor">;

// Checks if the param passed is a float ElementsAttr.
def FloatElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<ElementsAttr>() && $_self.cast<ElementsAttr>().getType().getElementType().isa<FloatType>()">,
        "float constant tensor">;

// Checks if the param passed is of NoneType.
def IsNoneType : Constraint<CPred<"$0.getType().isa<NoneType>()">>;

def ExtractSingleElementAsFloat : NativeCodeCall<
    "ExtractSingleElementAsFloat($_self.cast<ElementsAttr>())">;

// Checks if the value has rank at most 'n'.
class HasRankAtMost<int n> : Constraint<
    CPred<"$0.getType().cast<ShapedType>().getRank() <= " # n>>;

//===----------------------------------------------------------------------===//
// Ternary ops patterns.
//===----------------------------------------------------------------------===//
// Multi-pattern consisting of matching stand-alone convolution op followed by
// activation op.
multiclass FuseActFnIntoConvOpPat<Op ActFnOp, Attr ActFnAttr> {
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

multiclass FuseActFnIntoPoolOpPat<Op ActFnOp, Attr ActFnAttr> {
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
  defm : FuseActFnIntoConvOpPat<!cast<Op>(actFnPair[0]), !cast<Attr>(actFnPair[1])>;
  defm : FuseActFnIntoPoolOpPat<!cast<Op>(actFnPair[0]), !cast<Attr>(actFnPair[1])>;
}

class CanFuseConvOrDepthwiseConv<string is_depthwise> : Constraint<
  CPred<"TFL::CanFuseConvOrDepthwiseConv($0, $1, " # is_depthwise # ")">>;

// If we see a binary op (add, sub) op adding a constant value to a convolution
// op with constant bias, we can fuse the binary op into the convolution op by
// constant folding the bias and the binary op's constant operand. The following
// pattern restricts to float constant values for now.
multiclass FuseBinaryOpToPrecedingAffine<Op binaryOp> {
  def FuseBinaryOpWithConv#binaryOp : Pat<
    (binaryOp (TFL_Conv2DOp:$output $input, $filter,
                (ConstantOp FloatElementsAttr:$bias), $h_factor, $w_factor,
                TFL_AF_None, $padding, $stride_h, $stride_w),
              (ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_Conv2DOp $input, $filter,
      (binaryOp (ConstantOp $bias),
         (ConstantOp $value), TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
     (HasOneUse $output)]>;
  def FuseBinaryOpWithDepthwiseConv#binaryOp : Pat<
    (binaryOp (TFL_DepthwiseConv2DOp:$output $input, $filter,
                (ConstantOp FloatElementsAttr:$bias),
                $h_factor, $w_factor, TFL_AF_None, $padding, $stride_h,
                $stride_w, $multiplier),
              (ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_DepthwiseConv2DOp $input, $filter,
      (binaryOp (ConstantOp $bias), (ConstantOp $value), TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h, $stride_w,
      $multiplier),
    [(CanFuseConvOrDepthwiseConv<"true"> $filter, $value),
     (HasOneUse $output)]>;
   def FuseBinaryOpWithTransposeConv#binaryOp : Pat<
    (binaryOp (TFL_TransposeConvOp:$output $output_shape, $weights, $inputs,
                (ConstantOp FloatElementsAttr:$bias), $padding,
                $stride_h, $stride_w),
              (ConstantOp FloatElementsAttr:$value), TFL_AF_None),
    (TFL_TransposeConvOp $output_shape, $weights, $inputs,
      (binaryOp (ConstantOp $bias),
         (ConstantOp $value), TFL_AF_None),
      $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (HasOneUse $output)]>;
  // Fuse for TransposeConv with no bias
  def FuseBinaryOpWithTransposeConvNoneBias#binaryOp : Pat<
    (binaryOp (TFL_TransposeConvOp:$output $output_shape, $weights, $inputs,
                (ConstantOp $bias), $padding,
                $stride_h, $stride_w),
              (ConstantOp FloatElementsAttr:$value), TFL_AF_None),
    (TFL_TransposeConvOp $output_shape, $weights, $inputs,
      (ConstantOp $value),
      $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (IsNoneType $bias),
     (HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp]<Op> in
  defm : FuseBinaryOpToPrecedingAffine<binaryOp>;

def ExpandTo4DForConv: NativeCodeCall<"ExpandTo4DForConv($0)">;

def ExpandTo4DForDepthwiseConv: NativeCodeCall<
  "ExpandTo4DForDepthwiseConv($0)">;

// If we see a (div or Mul) op (dividing/multiplying) a constant value
// to a convolution op with constant filter and bias, we can fuse the div/mul
// into the convolution op by constant folding
// the filter/bias and the div/mul op's constant operand.
// The following pattern restricts to float constant values for now.

multiclass FuseMulOrDivWithConv2dOrDepthwiseConv2d<Op BinaryOp> {
  def FuseMulOrDivWithDepthwiseConv#BinaryOp : Pat<
    (BinaryOp (TFL_DepthwiseConv2DOp:$output $input,
                (ConstantOp FloatElementsAttr:$filter),
                (ConstantOp FloatElementsAttr:$bias),
                $h_factor, $w_factor, TFL_AF_None, $padding, $stride_h,
                $stride_w, $multiplier),
              (ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_DepthwiseConv2DOp $input,
      (BinaryOp
        (ConstantOp $filter),
        (ConstantOp (ExpandTo4DForDepthwiseConv $value)),
        TFL_AF_None),
      (BinaryOp
        (ConstantOp $bias),
        (ConstantOp $value),
        TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h,
      $stride_w, $multiplier),
    [(CanFuseConvOrDepthwiseConv<"true"> $filter, $value),
     (HasOneUse $output)]>;
  def FuseMulOrDivWithConv#BinaryOp : Pat<
    (BinaryOp (TFL_Conv2DOp:$conv_output $input,
                (ConstantOp FloatElementsAttr:$filter),
                (ConstantOp FloatElementsAttr:$bias),
                $h_factor, $w_factor, TFL_AF_None,
                $padding, $stride_h, $stride_w),
              (ConstantOp FloatElementsAttr:$value), $act_fn),
    (TFL_Conv2DOp $input,
      (BinaryOp (ConstantOp $filter),
        (ConstantOp (ExpandTo4DForConv $value)),
        TFL_AF_None),
      (BinaryOp (ConstantOp $bias),
        (ConstantOp $value),
        TFL_AF_None),
      $h_factor, $w_factor, $act_fn, $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
     (HasOneUse $conv_output)]>;
  def FuseMulOrDivWithTransposeConv#BinaryOp : Pat<
    (BinaryOp (TFL_TransposeConvOp:$output $output_shape,
                (ConstantOp FloatElementsAttr:$weights), $input,
                (ConstantOp FloatElementsAttr:$bias),
                $padding, $stride_h, $stride_w),
              (ConstantOp $value), TFL_AF_None),
    (TFL_TransposeConvOp $output_shape,
      (BinaryOp (ConstantOp $weights),
        (ConstantOp (ExpandTo4DForConv $value)),
        TFL_AF_None),
      $input,
      (BinaryOp (ConstantOp $bias),
        (ConstantOp $value),
        TFL_AF_None),
      $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (HasOneUse $output)]>;
  def FuseMulOrDivWithTransposeConvWithNoneBias#BinaryOp : Pat<
    (BinaryOp (TFL_TransposeConvOp:$output $output_shape,
                (ConstantOp FloatElementsAttr:$weights), $input,
                (ConstantOp $bias),
                $padding, $stride_h, $stride_w),
              (ConstantOp $value), TFL_AF_None),
    (TFL_TransposeConvOp $output_shape,
      (BinaryOp (ConstantOp $weights),
        (ConstantOp (ExpandTo4DForConv $value)),
        TFL_AF_None),
      $input,
      (ConstantOp $bias),
      $padding, $stride_h, $stride_w),
    [(CanFuseConvOrDepthwiseConv<"false"> $weights, $value),
     (IsNoneType $bias),
     (HasOneUse $output)]>;
}

foreach BinaryOp = [TFL_DivOp, TFL_MulOp]<Op> in
  defm : FuseMulOrDivWithConv2dOrDepthwiseConv2d<BinaryOp>;


// This pattern applies when the same quantize/dequantize have been used twice
// with the same scale. We want to remove the redundancy.
// TODO(fengliuai): move this to the sanity check of pre-quantize pass.
def eliminate_dq_q_pairs : Pat<
  (TFL_QuantizeOp (TFL_DequantizeOp $in), $qt),
  (replaceWithValue $in),
  [(NotFromQuantOpOrSameQuantType $in, $qt)]>;




// Checks if the operand has rank == n
class OperandHasRank<int n> : Constraint<
  CPred<"$0.getType().cast<ShapedType>().getRank() == " # n>>;

// Matching HardSwish
def MatchHardSwishPattern1 : Pat<
  (TFL_MulOp
    (TFL_MulOp
     $x, (TFL_AddOp
          $x,
          (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
          TFL_AF_Relu6),
     TFL_AF_None),
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
  (TFL_HardSwishOp $x)>;

def MatchHardSwishPattern2 : Pat<
  (TFL_MulOp
    $x,
    (TFL_MulOp
     (TFL_AddOp
      $x,
      (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_Relu6),
     (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
     TFL_AF_None),
  (TFL_HardSwishOp $x)>;

def MatchHardSwishPattern3 : Pat<
  (TFL_MulOp
    (TFL_MulOp
     $x,
     (TFL_AddOp
      $x,
      (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_Relu6),
     TFL_AF_None),
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
    TFL_AF_None),
  (TFL_HardSwishOp $x)>;

// Matching HardSwish with extra FakeQuant. These FakeQuant ops were due to
// incorrect placement in the quantization aware training.
def MatchHardSwishQuantized : Pat<
  (TFL_MulOp (TFL_DequantizeOp (TFL_QuantizeOp
    (TFL_MulOp
     $x, (TFL_DequantizeOp (TFL_QuantizeOp (TFL_AddOp
          $x,
          (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
          TFL_AF_Relu6), $qattr2)),
     TFL_AF_None), $qattr1)),
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
  (TFL_HardSwishOp $x)>;

// Constraint that the attribute value is less than 'n'
class ConstDoubleValueLessThan<string n> : Constraint<
  CPred<"$0.isa<DenseElementsAttr>() && "
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 && "
  "std::abs(*$0.cast<DenseElementsAttr>().getValues<float>().begin()) < "
  # n>>;

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
                           (ConstantOp I32ElementsAttr:$axis),
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
                        (ConstantOp I32ElementsAttr:$axis),
                        $keep_dims),
                       (ConstantOp $epsilon), TFL_AF_None)),
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
                        (ConstantOp I32ElementsAttr:$axis),
                        $keep_dims),
                       (ConstantOp $epsilon))),
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

def OperandsBroadcastToOutputType : Constraint<CPred<
  "TFL::OperandsBroadcastToOutputType($0.getType(), $1.getType(), "
                                     "$2.getType())">>;

def IsTailOfShape : Constraint<CPred<
  "TFL::IsTailOfShape($0.getType(), $1.getType())">>;

def Flatten : NativeCodeCall<
  "$0.cast<DenseElementsAttr>()"
    ".reshape(RankedTensorType::get({$0.getType().cast<ShapedType>().getNumElements()}, "
                                   "$0.getType().cast<ShapedType>().getElementType()))">;

def IsLastDimEqualToNumElements : Constraint<CPred<
  "$0.getType().cast<ShapedType>().getRank() >= 1 && "
  "$0.getType().cast<ShapedType>().getDimSize($0.getType().cast<ShapedType>().getRank() - 1) == "
  "$1.getType().cast<ShapedType>().getNumElements()">>;

def IsDefinedByFullyConnectedOp : Constraint<CPred<
  "$0.getDefiningOp<TFL::FullyConnectedOp>() != nullptr">>;

// Pattern for skipping Tile if it is mainly for broadcasting and the
// Op is already supporting broadcasting.
multiclass FuseTileBroadcastIntoFollowingBinary<Op BinaryOp> {
  def FuseTileBroadcastToBinaryOp1#BinaryOp : Pat<
    (BinaryOp:$result (TFL_TileOp $input, (ConstantOp $tile)),
     $operand, $act_func),
    (BinaryOp $input, $operand, $act_func),
  [(OperandsBroadcastToOutputType $input, $operand, $result),
   (HasRankAtMost<4> $input),
   (HasRankAtMost<4> $operand)]>;

  def FuseTileBroadcastToBinaryOp2#BinaryOp : Pat<
    (BinaryOp:$result $operand,
      (TFL_TileOp $input, (ConstantOp $tile)), $act_func),
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
    (BinaryOp (TFL_ReshapeOp:$lhs $input, (ConstantOp:$shape $s)),
      (ConstantOp:$rhs $a), $act_fn),
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
     (HasRankAtMost<4> $rhs)]>;

    // Move binary op before reshape:
    // binary(reshape(lhs), reshape(rhs)) => reshape(binary(lhs, rhs))
    // This is valid only when both side of the binary operand is reshaped, and
    // the sizes are the same both before and after the reshape.
    def MoveBinaryOpBeforeReshape#BinaryOp : Pat<
      (BinaryOp (TFL_ReshapeOp:$lhs $input1, (ConstantOp:$shape1 $s1)),
                (TFL_ReshapeOp:$rhs $input2, (ConstantOp:$shape2 $s2)),
                $act_fn),
      (TFL_ReshapeOp (BinaryOp $input1, $input2, $act_fn), $shape1),
      [(IsTailOfShape $rhs, $lhs),
       (IsTailOfShape $lhs, $rhs),
       (IsTailOfShape $input1, $input2),
       (IsTailOfShape $input2, $input1)]>;

    // Move binary op before reshape:
    // binary(reshape(lhs), rhs) => reshape(binary(lhs, flatten(rhs)))
    // This is valid only when the last dimension of lhs is equal to the
    // number of elements in constant rhs.
    // Therefore, after transformation broadcast of binary op is always
    // applied to the last dimension of $input.
    def MoveBinaryOpFlattenConstBeforeReshape#BinaryOp : Pat<
      (BinaryOp (TFL_ReshapeOp:$lhs $input, (ConstantOp:$shape $s)),
                (ConstantOp:$rhs ElementsAttr:$rhs_attr), $act_fn),
      (TFL_ReshapeOp (BinaryOp $input, (ConstantOp (Flatten $rhs_attr)),
                               $act_fn),
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
    (BinaryOp (TFL_ReshapeOp:$lhs $input, (ConstantOp:$shape $s)),
      (ConstantOp:$rhs $a)),
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
     (HasRankAtMost<4> $rhs)]>;

    // Move binary op before reshape:
    // binary(reshape(lhs), reshape(rhs)) => reshape(binary(lhs, rhs))
    // This is valid only when both side of the binary operand is reshaped, and
    // the sizes are the same both before and after the reshape.
    def MoveBinaryOpBeforeReshape#BinaryOp : Pat<
      (BinaryOp (TFL_ReshapeOp:$lhs $input1, (ConstantOp:$shape1 $s1)),
                (TFL_ReshapeOp:$rhs $input2, (ConstantOp:$shape2 $s2))),
      (TFL_ReshapeOp (BinaryOp $input1, $input2), $shape1),
      [(IsTailOfShape $rhs, $lhs),
       (IsTailOfShape $lhs, $rhs),
       (IsTailOfShape $input1, $input2),
       (IsTailOfShape $input2, $input1)]>;

    // Move binary op before reshape:
    // binary(reshape(lhs), rhs) => reshape(binary(lhs, flatten(rhs)))
    // This is valid only when the last dimension of lhs is equal to the
    // number of elements in constant rhs.
    // Therefore, after transformation broadcast of binary op is always
    // applied to the last dimension of $input.
    def MoveBinaryOpFlattenConstBeforeReshape#BinaryOp : Pat<
      (BinaryOp (TFL_ReshapeOp:$lhs $input, (ConstantOp:$shape $s)),
                (ConstantOp:$rhs ElementsAttr:$rhs_attr)),
      (TFL_ReshapeOp (BinaryOp $input, (ConstantOp (Flatten $rhs_attr))),
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
      [(HasOneUse $move)]>;
  }
}

// Returns shape of a ranked tensor.
// if called without a ranked tensor it will fail.
def GetShape: NativeCodeCall<"GetShape($0)">;

// Returns True if the operand type is RankedTensorType and valid.
def HasValidRankedTensor : Constraint<CPred<
  "$0.getType().isa<RankedTensorType>() && "
  "$0.getType().cast<RankedTensorType>().getNumDynamicDims() <= 1">>;

def ConvertSqueezeToReshape : Pat<
  (TFL_SqueezeOp:$squeeze_op $input, $squeeze_dims),
  (TFL_ReshapeOp $input, (ConstantOp (GetShape $squeeze_op))),
  [(HasValidRankedTensor $squeeze_op)]>;

// Convert expand_dims to reshape if possible.
def ConvertExpandDimsToReshape : Pat<
  (TFL_ExpandDimsOp:$expand_dims_op $input, $dim),
  (TFL_ReshapeOp $input, (ConstantOp (GetShape $expand_dims_op))),
  [(AnyStaticShapeTensor $expand_dims_op)]>;

class FloatValueEquals<string val> : Constraint<CPred<
  "FloatValueEquals($0, " # val # ")">>;

// ReLU patterns
def MatchReluPattern : Pat<
  (TFL_MaximumOp $input, (ConstantOp $Zero)),
  (TFL_ReluOp $input),
  [(FloatValueEquals<"0"> $Zero)]>;

def MatchRelu1Pattern1 : Pat<
  (TFL_MinimumOp (TFL_MaximumOp $input, (ConstantOp $NegOne)),
    (ConstantOp $One)),
  (TFL_Relu1Op $input),
  [(FloatValueEquals<"-1"> $NegOne), (FloatValueEquals<"1"> $One)]>;

def MatchRelu1Pattern2 : Pat<
  (TFL_MaximumOp (TFL_MinimumOp $input, (ConstantOp $One)),
    (ConstantOp $NegOne)),
  (TFL_Relu1Op $input),
  [(FloatValueEquals<"-1"> $NegOne), (FloatValueEquals<"1"> $One)]>;

def MatchLeakyRelu : Pat<
  (TFL_MaximumOp
    (TFL_MulOp:$mul_out $x,
     (ConstantOp F32ElementsAttr:$alpha), TFL_AF_None),
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
     (TFL_AddOp:$first_output $input, (ConstantOp $a), TFL_AF_None),
     (ConstantOp $b), ActFun),
    (TFL_AddOp $input,
     (TFL_AddOp (ConstantOp $a), (ConstantOp $b), TFL_AF_None),
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
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">)),
  (replaceWithValue $input)>;

// Optimize X^2 to X*X
def OptimizePow2ToSquare : Pat<
  (TFL_PowOp $input,
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "2.0f">)),
  (TFL_MulOp $input, $input, TFL_AF_None)>;

// Optimize X^(1/2) to √X
def OptimizePow2ToSqrt : Pat<
  (TFL_PowOp $input,
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.5f">)),
  (TFL_SqrtOp $input)>;

// Optimize X^(-1/2) to 1/√X == rsqrt(x)
def OptimizePow2ToRsqrt : Pat<
  (TFL_PowOp $input,
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "-0.5f">)),
  (TFL_RsqrtOp $input)>;

def CanOptimizeIdentityGatherNdOrScatterNdOp : Constraint<CPred<
  "TFL::CanOptimizeIdentityGatherNdOrScatterNdOp("
  "$0, $1.cast<DenseIntElementsAttr>())">>;

def OptimizeIdentityGatherNdOp : Pat<
  (TFL_GatherNdOp $params, (ConstantOp I32ElementsAttr: $indices)),
  (replaceWithValue $params),
  [(CanOptimizeIdentityGatherNdOrScatterNdOp $params, $indices)]>;

def OptimizeIdentityScatterNdOp : Pat<
  (TFL_ScatterNdOp (ConstantOp I32ElementsAttr: $indices), $params, $ignored),
  (replaceWithValue $params),
  [(CanOptimizeIdentityGatherNdOrScatterNdOp $params, $indices)]>;

def ShapeMatchesReduceWithKeepAxes : Constraint<CPred<
  "ShapeMatchesReduceWithKeepAxes($0, $1, $2)">>;

// Fold reshapes re-inserting reduced dimensions into the results of a reduction
// with `keep_dims=false` by changing it to one using `keep_dims=true`.
foreach ReduceOp = [TFL_ReduceMaxOp, TFL_ReduceMinOp, TFL_ReduceProdOp,
                    TFL_SumOp] in {
  def FoldReshapeTo#ReduceOp : Pat<
    (TFL_ReshapeOp
      (ReduceOp:$reduce $input, (ConstantOp I32ElementsAttr: $axes),
                        ConstBoolAttrFalse),
      (ConstantOp I32ElementsAttr: $shape)),
    (ReduceOp $input, (ConstantOp $axes), ConstBoolAttrTrue),
    [(ShapeMatchesReduceWithKeepAxes $input, $axes, $shape),
     (HasOneUse $reduce)]>;
}


def IsSame : Constraint<CPred<"$0 == $1">>;
def HasTwoUse : Constraint<CPred<
  "std::distance($0.use_begin(), $0.use_end()) == 2">>;
def AxesIsLastDimension : Constraint<CPred<
  "$0.cast<DenseIntElementsAttr>().getNumElements() == 1 && "
  "($0.cast<DenseIntElementsAttr>().getValue<APInt>({0}) == "
  "$1.getType().cast<ShapedType>().getRank() - 1 || $0.cast<DenseIntElementsAttr>().getValue<int32_t>({0}) == -1)">>;

// Convert exp(x)/sum(exp(x)) into softmax.
def OptimizeToSoftmax : Pat<
  (TFL_DivOp (TFL_ExpOp:$exp $input),
             (TFL_SumOp:$sum $sum_input, (ConstantOp I32ElementsAttr: $axes),
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
      (TFL_ReduceMaxOp:$max $max_input, (ConstantOp I32ElementsAttr: $axes),
                            ConstBoolAttrTrue),
    TFL_AF_None),
    $beta),
  (TFL_SoftmaxOp $input, $beta),
  [(IsSame $input, $max_input),
   (AxesIsLastDimension $axes, $max_input),
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
             (ConstantOp $constant),
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
    (SelectOp (ConstantOp $constant),
               $input1,
               $input2),
    (replaceWithValue $input1),
    [(HaveSameType $input1, $input2),
     (IsTailOfShape $input1, $constant),
     (IsTailOfShape $constant, $input1),
     (AllElementsAreBool<"true"> $constant)]>;
  // select(false_tensor, A, B) -> B
  def Optimize#SelectOp#False : Pat<
    (SelectOp (ConstantOp $constant),
               $input1,
               $input2),
    (replaceWithValue $input2),
    [(HaveSameType $input1, $input2),
     (IsTailOfShape $input1, $constant),
     (IsTailOfShape $constant, $input1),
     (AllElementsAreBool<"false"> $constant)]>;
}

// Remove (log-)softmax before arg-minmax as (log-)softmax is monotonic.
foreach ArgMinMaxOp = [TFL_ArgMinOp, TFL_ArgMaxOp] in {
  def RemoveSoftmaxOpBefore#ArgMinMaxOp : Pat<
    (ArgMinMaxOp (TFL_SoftmaxOp:$softmax $logits, TFL_FloatNonNegative:$beta),
                 (ConstantOp:$const_axes I32ElementsAttr:$axes)),
    (ArgMinMaxOp $logits, $const_axes),
    [(HasOneUse $softmax),
     (AxesIsLastDimension $axes, $logits)]>;

  def RemoveLogSoftmaxOpBefore#ArgMinMaxOp : Pat<
    (ArgMinMaxOp (TFL_LogSoftmaxOp:$log_softmax $logits),
                 (ConstantOp:$const_axes I32ElementsAttr:$axes)),
    (ArgMinMaxOp $logits, $const_axes),
    [(HasOneUse $log_softmax),
     (AxesIsLastDimension $axes, $logits)]>;
}

def CanOptimizeIdentitySliceOp : Constraint<CPred<
  "TFL::CanOptimizeIdentitySliceOp($0, $1, $2)">>;

// Remove Slice ops slicing the whole input tensor, effectively no-op.
def OptimizeSliceOp : Pat<
  (TFL_SliceOp:$output $input, (ConstantOp $begin), (ConstantOp $size)),
  (replaceWithValue $input),
  [(CanOptimizeIdentitySliceOp $input, $begin, $size)]>;
