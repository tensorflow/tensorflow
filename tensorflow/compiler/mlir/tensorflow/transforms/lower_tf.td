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
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Here, the element type can be any integer or float type. But, note that only
// 32 bit integers are supported for the values.
class GetScalarOfType<int value> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

class GetScalarOfFloatType<string value> : NativeCodeCall<
  "GetScalarOfFloatType(getElementTypeOrSelf($0)," # value # ")">;

def GetScalarInfOfType : NativeCodeCall<
  "GetScalarOfFloatType(getElementTypeOrSelf($0), "
  "std::numeric_limits<double>::infinity())">;

def GetScalarNanOfType : NativeCodeCall<
  "GetScalarOfFloatType(getElementTypeOrSelf($0), "
  "std::numeric_limits<double>::quiet_NaN())">;

class GetI64ScalarElementsAttr<int value> :
  NativeCodeCall<"GetI64ElementsAttr({" # value # "}, &$_builder)">;

def TrueBoolAttr : AttrConstraint<CPred<"$_self.getValue()">>;

def CreateTFShapeOp : NativeCodeCall<
    "$_builder.create<TF::ShapeOp>($0.getLoc(), $1, $2)">;

def IsI32 : NativeCodeCall<
    "$_builder.getBoolAttr(getElementTypeOrSelf($0.getType()).isInteger(32))">;

//===----------------------------------------------------------------------===//
// BiasAddGrad op patterns.
//===----------------------------------------------------------------------===//

def GetBiasAddGradReductionIndices : NativeCodeCall<
  "GetBiasAddGradReductionIndices("
  "$0.getType().cast<RankedTensorType>().getRank(), $1, &$_builder)">;

def LowerBiasAddGradOp :
  Pat<(TF_BiasAddGradOp AnyRankedTensor:$out_backprop, $data_format),
      (TF_SumOp $out_backprop,
                (TF_ConstOp (GetBiasAddGradReductionIndices $out_backprop,
                                                            $data_format)),
                /*keep_dims=*/ConstBoolAttrFalse)>;

// Lowers SoftmaxCrossEntropyWithLogitsOp using simpler TensorFlow ops. The op
// computes loss and backprop of the loss with respect to 'features'.
//
// Softmax cross entropy loss is defined as follows:
//
//  loss = Sum(-labels * Log(Exp(features) / Sum(Exp(features)))
//  loss = Sum(-labels * LogSoftmax(features))
//
// Computing gradient of the loss with respect to features gives us,
//
//  backprop = (Exp(features) / Sum(Exp(features))) - labels
//  backprop = Softmax(features) - labels
//
// Computation of the reduction axis for the Sum op depends on whether the
// input is a scalar or not. Restrict pattern to ranked inputs so that input to
// the Sum op is also ranked.

// TODO(hinsu): Support scalar inputs by introducing reshape to 1D.
def NonScalarType : Type<Neg<HasAnyRankOfPred<[0]>>, "Non scalar type">;

def LowerSoftmaxCrossEntropyWithLogitsOp : Pattern<
  (TF_SoftmaxCrossEntropyWithLogitsOp AnyRankedTensor:$features,
                                      AnyRankedTensor:$labels),
  [(TF_SumOp (TF_MulNoNanOp:$sum_input
                     (TF_LogSoftmaxOp $features),
                     (TF_NegOp $labels)),
             (TF_ConstOp (GetI64ScalarElementsAttr<-1>)),
             /*keep_dims=*/ConstBoolAttrFalse),
   (TF_SubOp (TF_SoftmaxOp $features), $labels)],
  [(NonScalarType $features), (NonScalarType $labels)]>;

// Returns size of the specified dimension as scalar elements attribute of type
// $1.
// Requires $0 to be of RankedTensorType with rank greater than `dim` and the
// dimension should be known.
class GetDimSizeOfType<int dim> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($1), "
  "$0.getType().cast<RankedTensorType>().getDimSize(" # dim # "))">;

// Same as the above with i32 element type.
class GetDimSizeAsI32<int dim> : NativeCodeCall<
  "GetScalarOfType($_builder.getIntegerType(32), "
  "$0.getType().cast<RankedTensorType>().getDimSize(" # dim # "))">;

// Sparse version of SoftmaxCrossEntropyWithLogits is lowered to dense by
// expanding the sparse labels using:
//
// labels = OneHotOp(sparse_labels, depth, 1.0, 0.0)
//
// If any of the indices are out of range, we must populate the labels with
// NaNs to follow the semantics of the op.
def LowerSparseSoftmaxCrossEntropyWithLogitsOp : Pattern<
  (TF_SparseSoftmaxCrossEntropyWithLogitsOp:$src_op
    AnyStaticShapeTensor:$features, $sparse_labels),
  [(TF_OneHotOp:$labels $sparse_labels,
     (TF_ConstOp (GetDimSizeAsI32<1> $features, $src_op__0)),
     (TF_ConstOp (GetScalarOfType<1> $features)),
     (TF_ConstOp (GetScalarOfType<0> $features)),
     ConstantAttr<I64Attr, "1">),
   (TF_SelectV2Op:$zero_or_nan
     (TF_LogicalAndOp
       (TF_LessEqualOp
         (TF_ConstOp (GetScalarOfType<0> $sparse_labels)), $sparse_labels),
       (TF_LessOp $sparse_labels,
         (TF_ConstOp (GetDimSizeOfType<1> $features, $sparse_labels)))),
     (TF_ConstOp (GetScalarOfType<0> $features)),
     (TF_ConstOp (GetScalarNanOfType $labels))),
   (TF_AddV2Op:$adjusted_labels $labels,
     (TF_ExpandDimsOp $zero_or_nan,
       (TF_ConstOp (GetI64ScalarElementsAttr<-1>)))),
  (TF_SoftmaxCrossEntropyWithLogitsOp $features, $adjusted_labels)]>;

//===----------------------------------------------------------------------===//
// Difference op patterns.
//===----------------------------------------------------------------------===//

def ComplexTensor   : TensorOf<[AnyComplex]>;
def RealTensor   : TensorOf<[AnySignlessInteger, AnyFloat]>;

def : Pat<(TF_SquareOp $val), (TF_MulOp $val, $val)>;

def : Pat<(TF_SquaredDifferenceOp RealTensor: $lhs, RealTensor:$rhs),
          (TF_SquareOp (TF_SubOp $lhs, $rhs))>;

def : Pat<(TF_SquaredDifferenceOp ComplexTensor: $lhs, ComplexTensor:$rhs),
          (TF_MulOp (TF_SubOp:$diff $lhs, $rhs), (TF_ConjOp $diff))>;

//===----------------------------------------------------------------------===//
// DivNoNan and MulNonNan op patterns.
//===----------------------------------------------------------------------===//

class BinaryNoNanPat<Op FromOp, Op ToOp>
  : Pat<(FromOp $l, $r),
        (TF_SelectV2Op (TF_EqualOp $r,
                                   (TF_ConstOp:$zero (GetScalarOfType<0> $r)),
                       /*incompatible_shape_error*/ConstBoolAttrTrue),
           $zero, (ToOp $l, $r))>;

foreach fromToBinPair = [[TF_DivNoNanOp, TF_DivOp],
                         [TF_MulNoNanOp, TF_MulOp]] in
  def : BinaryNoNanPat<fromToBinPair[0], fromToBinPair[1]>;

//===----------------------------------------------------------------------===//
// Expm1 op patterns.
//===----------------------------------------------------------------------===//

def LowerExpm1Op : Pat<(TF_Expm1Op $x),
                       (TF_SubOp (TF_ExpOp $x),
                        (TF_ConstOp (GetScalarOfType<1> $x)))>;

//===----------------------------------------------------------------------===//
// Fill op patterns.
//===----------------------------------------------------------------------===//

def LowerFillOp : Pat<(TF_FillOp $dims, $value),
                      (TF_BroadcastToOp $value, $dims)>;

//===----------------------------------------------------------------------===//
// Empty op patterns.
//===----------------------------------------------------------------------===//

def LowerEmptyOp : Pat<(TF_EmptyOp:$result $dims, TrueBoolAttr:$init),
                      (TF_BroadcastToOp
                       (TF_ConstOp (GetScalarOfType<0> $result)), $dims),
                      [(TF_SintOrFpTensor $result)]>;

//===----------------------------------------------------------------------===//
// Inf op patterns.
//===----------------------------------------------------------------------===//

def LowerIsInfOp : Pat<(TF_IsInfOp $x),
                       (TF_EqualOp (TF_AbsOp:$abs $x),
                        (TF_ConstOp:$inf (GetScalarInfOfType $x)),
                        /*incompatible_shape_error*/ConstBoolAttrTrue)>;

//===----------------------------------------------------------------------===//
// NaN op patterns.
//===----------------------------------------------------------------------===//

def LowerIsNanOp : Pat<(TF_IsNanOp $x),
                       (TF_NotEqualOp $x, $x,
                        /*incompatible_shape_error*/ConstBoolAttrTrue)>;

//===----------------------------------------------------------------------===//
// L2Loss op patterns.
//===----------------------------------------------------------------------===//

def GetAllAxes : NativeCodeCall<
  "GetI64ElementsAttrForSeq("
  "0, $0.getType().cast<RankedTensorType>().getRank(), &$_builder)">;

// L2Loss is lowered using the formula,
// L2Loss(input) = Sum(input * input) / 2

// TODO(hinsu): Support unranked tensors once b/144593778 is resolved to not
// cause result type mismatch.
def LowerL2LossOp :
  Pat<(TF_L2LossOp AnyRankedTensor:$input),
      (TF_DivOp
        (TF_SumOp (TF_MulOp $input, $input),
                  (TF_ConstOp (GetAllAxes $input)),
                  /*keep_dims=*/ConstBoolAttrFalse),
        (TF_ConstOp (GetScalarOfType<2> $input)))>;

//===----------------------------------------------------------------------===//
// Pad op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_PadOp TensorOf<[AnySignlessInteger, AnyFloat]>:$input, $paddings),
          (TF_PadV2Op $input, $paddings,
             (TF_ConstOp (GetScalarOfType<0> $input)))>;

//===----------------------------------------------------------------------===//
// Reciprocal op patterns.
//===----------------------------------------------------------------------===//

def LowerReciprocal : Pat<(TF_ReciprocalOp $x),
                          (TF_DivOp (TF_ConstOp (GetScalarOfType<1> $x)), $x)>;

//===----------------------------------------------------------------------===//
// Round op patterns.
//===----------------------------------------------------------------------===//


// Rounds on integers should just be bypassed.
def : Pat<(TF_RoundOp:$res TF_IntTensor:$input), (TF_IdentityOp $input)>;

// Implements TF Round on floats using basic operations.
def : Pat<(TF_RoundOp:$res TF_FloatTensor:$input),
          (TF_SelectOp
           (TF_LessOp
            (TF_SubOp $input, (TF_FloorOp:$floor $input)),
            (TF_ConstOp (GetScalarOfFloatType<"0.5"> $input))),
           $floor,
           (TF_AddV2Op
            (TF_ConstOp (GetScalarOfType<1> $input)), $floor))>;


//===----------------------------------------------------------------------===//
// Rsqrt op patterns.
//===----------------------------------------------------------------------===//

// RsqrtGrad(lhs, rhs) = (lhs * lhs * lhs) * (rhs / -2)
def : Pat<(TF_RsqrtGradOp $lhs, $rhs),
          (TF_MulOp (TF_MulOp (TF_MulOp $lhs, $lhs), $lhs),
                    (TF_DivOp $rhs,
                              (TF_ConstOp (GetScalarOfType<-2> $rhs))))>;


//===----------------------------------------------------------------------===//
// Size op patterns.
//===----------------------------------------------------------------------===//

// Size(x) = Prod(Shape(x), reduction_indices=0, keep_dims=false)
def : Pat<(TF_SizeOp:$res $arg),
          (TF_ProdOp
            (CreateTFShapeOp $res, $arg, (IsI32 $res)),
            /*reduction_indices=*/(TF_ConstOp (GetScalarOfType<0> $res)),
            /*keep_dims=*/ConstBoolAttrFalse)>;


//===----------------------------------------------------------------------===//
// TanhGrad op patterns.
//===----------------------------------------------------------------------===//

// grad = dy * (1 - y**2)

def LowerTanhGradOp :
  Pat<(TF_TanhGradOp $y, $dy),
      (TF_MulOp $dy,
                (TF_SubOp (TF_ConstOp (GetScalarOfType<1> $y)),
                          (TF_SquareOp $y)))>;


//===----------------------------------------------------------------------===//
// LowerFakeQuantWithMinMaxArgs op patterns.
//===----------------------------------------------------------------------===//

def LowerFakeQuantWithMinMaxArgs :
  Pat<(TF_FakeQuantWithMinMaxArgsOp TF_FloatTensor: $input,
       $min, $max, $bits, $narrow_range),
      (TF_FakeQuantWithMinMaxVarsOp $input,
         (TF_ConstOp $min), (TF_ConstOp $max), $bits, $narrow_range)>;

// ZerosLike op patterns.
//===----------------------------------------------------------------------===//


class LowerInitializationOp<Op FromOp, int initial_val>
  : Pat<(FromOp:$src_op
         TensorOf<[AnyInteger, AnyFloat, AnyComplex]>:$input),
        (TF_BroadcastToOp (TF_ConstOp (GetScalarOfType<initial_val> $input)),
                          (CreateTFShapeOp $src_op, $input,
                                           /*use 32bit*/ConstBoolAttrFalse))>;

def LowerZerosLikeOp : LowerInitializationOp<TF_ZerosLikeOp, 0>;
def LowerOnesLikeOp : LowerInitializationOp<TF_OnesLikeOp, 1>;

def LowerScatterNdOp :
  Pat<(TF_ScatterNdOp $indices,
       TensorOf<[AnyInteger, AnyFloat, AnyComplex]>:$updates, $shape),
      (TF_TensorScatterAddOp
       (TF_FillOp $shape, (TF_ConstOp (GetScalarOfType<0> $updates))),
       $indices, $updates)>;

//===----------------------------------------------------------------------===//
// Xdivy, Xlog1p and Xlogy op patterns.
//===----------------------------------------------------------------------===//

class BinaryXopyPat<dag From, dag To>
  : Pat<From,
        (TF_SelectV2Op (TF_EqualOp $x,
                                   (TF_ConstOp:$zero (GetScalarOfType<0> $x)),
                       /*incompatible_shape_error*/ConstBoolAttrTrue),
           $zero, To)>;

foreach fromToPair = [[(TF_XdivyOp $x, $y), (TF_DivOp $x, $y)],
                      [(TF_Xlog1pyOp $x, $y), (TF_MulOp $x, (TF_Log1pOp $y))],
                      [(TF_XlogyOp $x, $y), (TF_MulOp $x, (TF_LogOp $y))]] in
  def : BinaryXopyPat<fromToPair[0], fromToPair[1]>;
