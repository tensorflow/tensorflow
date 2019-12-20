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
include "mlir/Dialect/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

//===----------------------------------------------------------------------===//
// Ternary ops patterns.
//===----------------------------------------------------------------------===//
// Multi-pattern consisting of matching stand-alone convolution op followed by
// activation op.
multiclass FuseActFnIntoConvOpPat<dag ActFnOp, dag ActFnAttr> {
  def : Pat<(ActFnOp (TFL_Conv2DOp $input, $filter, $bias,
                                   $h_factor, $w_factor, TFL_AF_None,
                                   $padding, $stride_h, $stride_w)),
            (TFL_Conv2DOp $input, $filter, $bias,
                          $h_factor, $w_factor, ActFnAttr,
                          $padding, $stride_h, $stride_w)>;
  def : Pat<(ActFnOp (TFL_DepthwiseConv2DOp $input, $filter, $bias,
                                   $h_factor, $w_factor, TFL_AF_None,
                                   $padding, $stride_h, $stride_w,
                                   $multiplier)),
            (TFL_DepthwiseConv2DOp $input, $filter, $bias,
                                   $h_factor, $w_factor, ActFnAttr,
                                   $padding, $stride_h, $stride_w,
                                   $multiplier)>;
}

// TODO(hinsu): Also fuse ops corresponding to SIGN_BIT fused
// activation functions.
// Currently we're not fusing tanh, sigmoid, hard_swish and other activations
// those cannot be simply translated into clamping.
foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                     [TFL_Relu6Op, TFL_AF_Relu6],
                     [TFL_Relu1Op, TFL_AF_Relu1]] in
  defm : FuseActFnIntoConvOpPat<actFnPair[0], actFnPair[1]>;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0->hasOneUse()">>;

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
          [(HasOneUse $output)]>;
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
          [(HasOneUse $output)]>;
}
foreach binaryOp = [TFL_AddOp, TFL_SubOp] in
  defm : FuseBinaryOpToPrecedingAffine<binaryOp>;

class CanFuseConvOrDepthwiseConv<string is_depthwise> : Constraint<
  CPred<"TFL::CanFuseConvOrDepthwiseConv($0, $1, " # is_depthwise # ")">>;

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
  CPred<"$0->getType().cast<ShapedType>().getRank() == " # n>>;

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

def AreBroadcastableTypes : Constraint<CPred<
  "TFL::IsBroadcastableElementsAttrAndType($0->getType(), $1->getType())">>;

// Pattern for skipping Tile if it is mainly for broadcasting and the
// Op is already supporting broadcasting.
multiclass FuseTileBroadcastIntoFollowingBinary<dag BinaryOp> {
  def : Pat<(BinaryOp (TFL_TileOp $input, (ConstantOp $tile)),
             $operand, $act_func),
  (BinaryOp $input, $operand, $act_func),
  [(AreBroadcastableTypes $input, $operand)]>;

  def : Pat<(BinaryOp $operand,
             (TFL_TileOp $input, (ConstantOp $tile)), $act_func),
  (BinaryOp $operand, $input, $act_func),
  [(AreBroadcastableTypes $operand, $input)]>;
}

foreach BroadcastingOp = [TFL_AddOp, TFL_SubOp, TFL_DivOp, TFL_MulOp]
  in defm : FuseTileBroadcastIntoFollowingBinary<BroadcastingOp>;

// Returns shape of a ranked tensor.
// if called without a ranked tensor it will fail.
def GetShape: NativeCodeCall<"GetShape($0)">;

def : Pat<(TFL_SqueezeOp:$squeeze_op $input, $squeeze_dims),
          (TFL_ReshapeOp $input,
           (ConstantOp (GetShape $squeeze_op))),
          [(AnyStaticShapeTensor $squeeze_op)]>;

class ValueEquals<string val> : Constraint<CPred<
  "$0.cast<DenseElementsAttr>().getNumElements() == 1 &&"
  "*$0.cast<DenseElementsAttr>().getValues<float>().begin() == " # val>>;

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

// Multi-pattern consisting of matching stand-alone op or op followed by relu.
multiclass FusedBinaryActivationFuncOpPat<dag BinaryOp> {
  foreach actFnPair = [[TFL_ReluOp, TFL_AF_Relu],
                       [TFL_Relu6Op, TFL_AF_Relu6],
                       [TFL_Relu1Op, TFL_AF_Relu1]] in {
    def : Pat<(actFnPair[0] (BinaryOp $lhs, $rhs, TFL_AF_None)),
              (BinaryOp $lhs, $rhs, actFnPair[1])>;
  }
}

// Instantiated FusedBinary patterns for the from-to pairs of ops.
foreach BinaryOps = [TFL_AddOp, TFL_DivOp,
                     TFL_MulOp, TFL_SubOp] in
  defm : FusedBinaryActivationFuncOpPat<BinaryOps>;

// The constant folding in this pass might produce constant in the tf dialect.
// This rule is to legalize these constant to the tfl dialect.
def : Pat<(TF_ConstOp ElementsAttr:$value), (TFL_ConstOp $value)>;
