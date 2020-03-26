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
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

def ExtractSingleElementAsFloat : NativeCodeCall<
    "ExtractSingleElementAsFloat($_self.cast<ElementsAttr>())">;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

//===----------------------------------------------------------------------===//
// Ternary ops patterns.
//===----------------------------------------------------------------------===//
// Multi-pattern consisting of matching stand-alone convolution op followed by
// activation op.
multiclass FuseActFnIntoConvOpPat<dag ActFnOp, dag ActFnAttr> {
  def : Pat<(ActFnOp (TFL_Conv2DOp:$conv_out $input, $filter, $bias,
                                   $h_factor, $w_factor, TFL_AF_None,
                                   $padding, $stride_h, $stride_w)),
            (TFL_Conv2DOp $input, $filter, $bias,
                          $h_factor, $w_factor, ActFnAttr,
                          $padding, $stride_h, $stride_w),
            [(HasOneUse $conv_out)]>;
  def : Pat<(ActFnOp (TFL_DepthwiseConv2DOp:$conv_out $input, $filter, $bias,
                                   $h_factor, $w_factor, TFL_AF_None,
                                   $padding, $stride_h, $stride_w,
                                   $multiplier)),
            (TFL_DepthwiseConv2DOp $input, $filter, $bias,
                                   $h_factor, $w_factor, ActFnAttr,
                                   $padding, $stride_h, $stride_w,
                                   $multiplier),
            [(HasOneUse $conv_out)]>;
}

// TODO(hinsu): Also fuse ops corresponding to SIGN_BIT fused
// activation functions.
// Currently we're not fusing tanh, sigmoid, hard_swish and other activations
// those cannot be simply translated into clamping.
foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                     [TFL_Relu6Op, TFL_AF_Relu6],
                     [TFL_Relu1Op, TFL_AF_Relu1]] in
  defm : FuseActFnIntoConvOpPat<actFnPair[0], actFnPair[1]>;


class CanFuseConvOrDepthwiseConv<string is_depthwise> : Constraint<
  CPred<"TFL::CanFuseConvOrDepthwiseConv($0, $1, " # is_depthwise # ")">>;

// If we see a binary op (add, sub) op adding a constant value to a convolution
// op with constant bias, we can fuse the binary op into the convolution op by
// constant folding the bias and the binary op's constant operand. The following
// pattern restricts to float constant values for now.
multiclass FuseBinaryOpToPrecedingAffine<dag binaryOp> {
  def : Pat<(binaryOp (TFL_Conv2DOp:$output $input, $filter,
                          (ConstantOp F32ElementsAttr:$bias),
                          $h_factor, $w_factor, TFL_AF_None,
                          $padding, $stride_h, $stride_w),
                     (ConstantOp F32ElementsAttr:$value), $act_fn),
          (TFL_Conv2DOp $input, $filter,
                        (binaryOp (ConstantOp $bias),
                                   (ConstantOp $value), TFL_AF_None),
                        $h_factor, $w_factor, $act_fn,
                        $padding, $stride_h, $stride_w),
          [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
           (HasOneUse $output)]>;
  def : Pat<(binaryOp (TFL_DepthwiseConv2DOp:$output $input, $filter,
                          (ConstantOp F32ElementsAttr:$bias),
                          $h_factor, $w_factor, TFL_AF_None,
                          $padding, $stride_h, $stride_w,
                          $multiplier),
                     (ConstantOp F32ElementsAttr:$value), $act_fn),
          (TFL_DepthwiseConv2DOp $input, $filter,
                          (binaryOp (ConstantOp $bias),
                                     (ConstantOp $value),
                                     TFL_AF_None),
                          $h_factor, $w_factor, $act_fn,
                          $padding, $stride_h, $stride_w,
                          $multiplier),
          [(CanFuseConvOrDepthwiseConv<"true"> $filter, $value),
           (HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp] in
  defm : FuseBinaryOpToPrecedingAffine<binaryOp>;

def ExpandTo4DForConv: NativeCodeCall<"ExpandTo4DForConv($0)">;

def ExpandTo4DForDepthwiseConv: NativeCodeCall<
  "ExpandTo4DForDepthwiseConv($0)">;

// If we see a (div or Mul) op (dividing/multiplying) a constant value
// to a convolution op with constant filter and bias, we can fuse the div/mul
// into the convolution op by constant folding
// the filter/bias and the div/mul op's constant operand.
// The following pattern restricts to float constant values for now.

multiclass FuseMulOrDivWithConv2dOrDepthwiseConv2d<dag BinaryOp> {
  def : Pat<(BinaryOp (TFL_DepthwiseConv2DOp:$output $input,
                          (ConstantOp F32ElementsAttr:$filter),
                          (ConstantOp F32ElementsAttr:$bias),
                          $h_factor, $w_factor, TFL_AF_None,
                          $padding, $stride_h, $stride_w,
                          $multiplier),
                     (ConstantOp F32ElementsAttr:$value), $act_fn),
          (TFL_DepthwiseConv2DOp $input,
                          (BinaryOp (ConstantOp $filter),
                                     (ConstantOp
                                       (ExpandTo4DForDepthwiseConv $value)),
                                     TFL_AF_None),
                          (BinaryOp (ConstantOp $bias),
                                     (ConstantOp $value),
                                     TFL_AF_None),
                          $h_factor, $w_factor, $act_fn,
                          $padding, $stride_h, $stride_w,
                          $multiplier),
          [(CanFuseConvOrDepthwiseConv<"true"> $filter, $value),
           (HasOneUse $output)]>;
  def : Pat<(BinaryOp (TFL_Conv2DOp:$conv_output $input,
                          (ConstantOp F32ElementsAttr:$filter),
                          (ConstantOp F32ElementsAttr:$bias),
                          $h_factor, $w_factor, TFL_AF_None,
                          $padding, $stride_h, $stride_w),
                     (ConstantOp F32ElementsAttr:$value), $act_fn),
          (TFL_Conv2DOp $input,
                          (BinaryOp (ConstantOp $filter),
                                     (ConstantOp (ExpandTo4DForConv $value)),
                                     TFL_AF_None),
                          (BinaryOp (ConstantOp $bias),
                                     (ConstantOp $value),
                                     TFL_AF_None),
                          $h_factor, $w_factor, $act_fn,
                          $padding, $stride_h, $stride_w),
          [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
           (HasOneUse $conv_output)]>;
}

foreach BinaryOp = [TFL_DivOp, TFL_MulOp] in
  defm : FuseMulOrDivWithConv2dOrDepthwiseConv2d<BinaryOp>;


// This pattern applies when the same quantize/dequantize have been used twice
// with the same scale. We want to remove the redundancy.
// TODO(fengliuai): move this to the sanity check of pre-quantize pass.
def : Pat<(TFL_QuantizeOp (TFL_DequantizeOp $in), $qt), (replaceWithValue $in)>;


// Constraint that makes sure both operands are the same operands.
def EqualOperands : Constraint<CPred<"$0 == $1">>;


// Checks if the operand has rank == n
class OperandHasRank<int n> : Constraint<
  CPred<"$0.getType().cast<ShapedType>().getRank() == " # n>>;

// Matching HardSwish
def : Pat<
  (TFL_MulOp
    (TFL_MulOp
     $x, (TFL_AddOp
          $y,
          (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
          TFL_AF_Relu6),
     TFL_AF_None),
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
  (TFL_HardSwishOp $x),
  [(EqualOperands $x, $y)]>;

def : Pat<
  (TFL_MulOp
    $x,
    (TFL_MulOp
     (TFL_AddOp
      $y,
      (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
      TFL_AF_Relu6),
     (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
     TFL_AF_None),
  (TFL_HardSwishOp $x),
  [(EqualOperands $x, $y)]>;

// Matching HardSwish with extra FakeQuant. These FakeQuant ops were due to
// incorrect placement in the quantization aware training.
// TODO(b/149735743): We should make the placement automatically.
def : Pat<
  (TFL_MulOp (TFL_DequantizeOp (TFL_QuantizeOp
    (TFL_MulOp
     $x, (TFL_DequantizeOp (TFL_QuantizeOp (TFL_AddOp
          $y,
          (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "3.0f">),
          TFL_AF_Relu6), $qattr2)),
     TFL_AF_None), $qattr1)),
    (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "0.166666666f">),
     TFL_AF_None),
  (TFL_HardSwishOp $x),
  [(EqualOperands $x, $y)]>;

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
multiclass L2NormalizePatterns<dag FirstOp, dag SecondOp> {
  // This pattern constructs L2NormalizationOp from
  // Mul->Rsqrt->Sum->Square Or
  // Div->sqrt->Sum->Square
  def : Pat<(FirstOp $operand1,
                     (SecondOp
                        (TFL_SumOp
                           (TFL_SquareOp:$sq_op $square_operand),
                           (ConstantOp I32ElementsAttr:$axis),
                           $keep_dims)),
                     TFL_AF_None),
           (TFL_L2NormalizationOp $operand1, TFL_AF_None),
           [(EqualOperands $operand1, $square_operand),
            (L2NormValidReduceIndex $sq_op, $axis)]>;

  // Below patterns for L2Normalize when there is an Add or Maximum
  // adding or clamping to a small constant scalar.
  def : Pat<(FirstOp $operand1,
                     (SecondOp
                      (TFL_AddOp
                       (TFL_SumOp
                        (TFL_SquareOp:$sq_op $square_operand),
                        (ConstantOp I32ElementsAttr:$axis),
                        $keep_dims),
                       (ConstantOp $epsilon), TFL_AF_None)),
           TFL_AF_None),
           (TFL_L2NormalizationOp $operand1, TFL_AF_None),
           [(EqualOperands $operand1, $square_operand),
            (L2NormValidReduceIndex $sq_op, $axis),
            (ConstDoubleValueLessThan<"1e-3"> $epsilon)]>;

  def : Pat<(FirstOp $operand1,
                     (SecondOp
                      (TFL_MaximumOp
                       (TFL_SumOp
                        (TFL_SquareOp:$sq_op $square_operand),
                        (ConstantOp I32ElementsAttr:$axis),
                        $keep_dims),
                       (ConstantOp $epsilon))),
           TFL_AF_None),
           (TFL_L2NormalizationOp $operand1, TFL_AF_None),
           [(EqualOperands $operand1, $square_operand),
            (L2NormValidReduceIndex $sq_op, $axis),
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

def HaveSameType : Constraint<CPred<"$0.getType(), $1.getType()">>;

// Pattern for skipping Tile if it is mainly for broadcasting and the
// Op is already supporting broadcasting.
multiclass FuseTileBroadcastIntoFollowingBinary<dag BinaryOp> {
  def : Pat<(BinaryOp:$result (TFL_TileOp $input, (ConstantOp $tile)),
             $operand, $act_func),
  (BinaryOp $input, $operand, $act_func),
  [(OperandsBroadcastToOutputType $input, $operand, $result)]>;

  def : Pat<(BinaryOp:$result $operand,
             (TFL_TileOp $input, (ConstantOp $tile)), $act_func),
  (BinaryOp $operand, $input, $act_func),
  [(OperandsBroadcastToOutputType $operand, $input, $result)]>;
}

// Multi-pattern consisting of matching stand-alone op or op followed by relu.
multiclass FusedBinaryActivationFuncOpPat<dag BinaryOp> {
  foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                       [TFL_Relu6Op, TFL_AF_Relu6],
                       [TFL_Relu1Op, TFL_AF_Relu1]] in {
    def : Pat<(actFnPair[0] (BinaryOp:$binary_out $lhs, $rhs, TFL_AF_None)),
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
  def : Pat<(BinaryOp (TFL_ReshapeOp:$lhs $input, (ConstantOp:$shape $s)),
                      (ConstantOp:$rhs $a), TFL_AF_None),
            (TFL_ReshapeOp (BinaryOp $input, $rhs, TFL_AF_None), $shape),
             // The broadcasting of "BinaryOp" only happens in the lower
             // dimensions, and the higher dimensions are same.
            [(IsTailOfShape $rhs, $lhs),
             (HasOneUse $lhs),
             // the two operands of the binary op is broadcastable
             (AreBroadcastableTypes $rhs, $input)]>;
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
  def : Pat<(BinaryOp (TFL_ReshapeOp:$lhs $input, (ConstantOp:$shape $s)),
                      (ConstantOp:$rhs $a)),
            (TFL_ReshapeOp (BinaryOp $input, $rhs), $shape),
             // The broadcasting of "BinaryOp" only happens in the lower
             // dimensions, and the higher dimensions are same.
            [(IsTailOfShape $rhs, $lhs),
             (HasOneUse $lhs),
             // the two operands of the binary op is broadcastable
             (AreBroadcastableTypes $rhs, $input)]>;
}

// Returns shape of a ranked tensor.
// if called without a ranked tensor it will fail.
def GetShape: NativeCodeCall<"GetShape($0)">;

// Convert squeeze to reshape if possible.
def : Pat<(TFL_SqueezeOp:$squeeze_op $input, $squeeze_dims),
          (TFL_ReshapeOp $input,
           (ConstantOp (GetShape $squeeze_op))),
          [(AnyStaticShapeTensor $squeeze_op)]>;

// Convert expand_dims to reshape if possible.
def : Pat<(TFL_ExpandDimsOp:$expand_dims_op $input, $dim),
          (TFL_ReshapeOp $input,
           (ConstantOp (GetShape $expand_dims_op))),
          [(AnyStaticShapeTensor $expand_dims_op)]>;

class ValueEquals<string val> : Constraint<CPred<
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 &&"
  "*$0.cast<DenseElementsAttr>().getValues<float>().begin() == " # val>>;

// ReLU patterns
def : Pat<(TFL_MinimumOp (TFL_MaximumOp $input,
                          (ConstantOp $NegOne)),
           (ConstantOp $One)),
          (TFL_Relu1Op $input),
          [(ValueEquals<"-1"> $NegOne), (ValueEquals<"1"> $One)]>;

def : Pat<(TFL_MaximumOp (TFL_MinimumOp $input,
                          (ConstantOp $One)),
           (ConstantOp $NegOne)),
          (TFL_Relu1Op $input),
          [(ValueEquals<"-1"> $NegOne), (ValueEquals<"1"> $One)]>;

def : Pat<(TFL_MaximumOp (TFL_MulOp:$mul_out $input1,
                          (ConstantOp F32ElementsAttr:$alpha), TFL_AF_None),
           $input2),
          (TFL_LeakyReluOp $input1, ExtractSingleElementAsFloat:$alpha),
          [(ConstDoubleValueLessThan<"1"> $alpha),
           (EqualOperands $input1, $input2),
           (HasOneUse $mul_out)]>;

// Checks if the operand0's rank is one less than operand1's rank.
def PReluAlphaRankCheck : Constraint<
  CPred<"$0.getType().cast<ShapedType>().getRank() == "
  "$1.getType().cast<ShapedType>().getRank() - 1">>;

// PReLU pattern from Keras:
// f(x) = Relu(x) + (-alpha * Relu(-x))
def : Pat<(TFL_AddOp
           (TFL_ReluOp:$relu_out $input1),
           (TFL_MulOp:$mul_out
            (TFL_ReluOp (TFL_NegOp:$input_neg_out $input2)),
            $neg_alpha,
            TFL_AF_None),
           TFL_AF_None),
          (TFL_PReluOp $input1, (TFL_NegOp $neg_alpha)),
          [(EqualOperands $input1, $input2),
           (PReluAlphaRankCheck $neg_alpha, $input1),
           (HasOneUse $relu_out),
           (HasOneUse $mul_out),
           (HasOneUse $input_neg_out)]>;

// The constant folding in this pass might produce constant in the tf dialect.
// This rule is to legalize these constant to the tfl dialect.
def : Pat<(TF_ConstOp ElementsAttr:$value), (TFL_ConstOp $value)>;
