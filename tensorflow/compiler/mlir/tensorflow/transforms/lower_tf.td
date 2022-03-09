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
include "mlir/Dialect/Func/IR/FuncOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.td"

class GetScalarOfFloatType<string value> : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0)," # value # ")">;

def GetScalarInfOfType : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0), "
  "std::numeric_limits<double>::infinity())">;

def GetScalarNanOfType : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0), "
  "std::numeric_limits<double>::quiet_NaN())">;

class GetI64ScalarElementsAttr<int value> :
  NativeCodeCall<"GetI64ElementsAttr({" # value # "}, &$_builder)">;

class GetF32Scalar<int value> :
  NativeCodeCall<"GetF32Scalar(&$_builder, " # value # ")">;

def TrueBoolAttr : AttrConstraint<CPred<"$_self.cast<::mlir::BoolAttr>().getValue()">>;

def CreateTFShapeOp : NativeCodeCall<
    "$_builder.create<TF::ShapeOp>($0.getLoc(), $1, $2)">;

def IsI32 : NativeCodeCall<
    "$_builder.getBoolAttr(getElementTypeOrSelf($0.getType()).isInteger(32))">;

def CreateTFCastOpF32 : NativeCodeCall<
    "CreateTFCastOpF32(&$_builder, $0.getLoc(), $1, $2)">;

def CreateTFCastOpI32 : NativeCodeCall<
    "CreateTFCastOpI32(&$_builder, $0.getLoc(), $1, $2)">;

def CreateTensorScatterNdOp : NativeCodeCall<
    "$_builder.create<TF::ScatterNdOp>($0.getLoc(), $0.getType(), $1, $2, $3)">;

//===----------------------------------------------------------------------===//
// Add op patterns.
//===----------------------------------------------------------------------===//

// We can lower AddOp to AddV2Op, except for string type. AddOp is defined for
// TF_NumberNotQuantizedOrStrTensor tensor types. TF_NumberNotQuantizedTensor
// is the same, except not containing TF_Str.

def LowerAddOp : Pat<(TF_AddOp TF_NumberNotQuantizedTensor:$x,
                               TF_NumberNotQuantizedTensor:$y),
                     (TF_AddV2Op $x, $y)>;

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
// Dequantize op patterns.
//===----------------------------------------------------------------------===//

def DequantizeHalfRange : NativeCodeCall<
  "DequantizeHalfRange(&$_builder, $0)">;

// TODO(b/188530181): Generalize to more quantized input types,
//   allow num_slices > 1, and allow non default arguments for $mode,
//   $narrow_range, and $axis.
def LowerDequantizeOp : Pat<
  (TF_DequantizeOp:$result
    $input, $min_range, $max_range,
    TF_AnyStrAttrOf<["MIN_COMBINED"]>:$mode,
    ConstBoolAttrFalse:$narrow_range,
    ConstantAttr<I64Attr, "-1">:$axis),
  (TF_CastOp
    (TF_AddV2Op
      (TF_MulOp
        (TF_AddV2Op
          (CreateTFCastOpF32 $result, $input, /*truncate=*/ConstBoolAttrFalse),
          (TF_ConstOp (DequantizeHalfRange $input))),
        (TF_DivOp
          (TF_SubOp $max_range, $min_range),
          (TF_ConstOp (GetF32Scalar<255>)))),
      $min_range),
    /*truncate=*/ConstBoolAttrFalse),
  [(TensorOf<[TF_Qint8, TF_Quint8]> $input),
   (TF_Float32Tensor $min_range),
   (TF_Float32Tensor $max_range),
   (TensorOf<[TF_Float32, TF_Bfloat16]> $result)]>;

//===----------------------------------------------------------------------===//
// Difference op patterns.
//===----------------------------------------------------------------------===//

def ComplexTensor : TensorOf<[AnyComplex]>;
def RealTensor : TensorOf<[AnySignlessInteger, AnyFloat]>;

def LowerSquareOp : Pat<(TF_SquareOp $val), (TF_MulOp $val, $val)>;

def LowerSquaredDifferenceOpOnRealTensors : Pat<
  (TF_SquaredDifferenceOp RealTensor: $lhs, RealTensor:$rhs),
  (TF_SquareOp (TF_SubOp $lhs, $rhs))>;

def LowerSquaredDifferenceOpOneComplexTensors : Pat<
  (TF_SquaredDifferenceOp ComplexTensor: $lhs, ComplexTensor:$rhs),
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

def LowerDivNoNanOp : BinaryNoNanPat<TF_DivNoNanOp, TF_DivOp>;

def LowerMulNoNanOp : BinaryNoNanPat<TF_MulNoNanOp, TF_MulOp>;

//===----------------------------------------------------------------------===//
// TruncateDiv op patterns.
//===----------------------------------------------------------------------===//

def LowerTruncateDivOp : Pat<
  (TF_TruncateDivOp $lhs, $rhs),
  (TF_DivOp $lhs, $rhs)>;

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
// Inv op patterns.
//===----------------------------------------------------------------------===//

def LowerInv : Pat<(TF_InvOp $x), (TF_ReciprocalOp $x)>;

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

def LowerPadOp : Pat<
  (TF_PadOp TensorOf<[AnySignlessInteger, AnyFloat]>:$input, $paddings),
  (TF_PadV2Op $input, $paddings,
    (TF_ConstOp
      (GetScalarOfType<0> $input)
    )
  )>;

//===----------------------------------------------------------------------===//
// Reciprocal op patterns.
//===----------------------------------------------------------------------===//

def LowerReciprocal : Pat<(TF_ReciprocalOp $x),
                          (TF_DivOp (TF_ConstOp (GetScalarOfType<1> $x)), $x)>;

//===----------------------------------------------------------------------===//
// Round op patterns.
//===----------------------------------------------------------------------===//

// Rint is specified as RoundHalfToEven, which happens to be the same behavior
// as TF_RoundOp, so lower to TF_RoundOp.
def LowerRintOp : Pat<(TF_RintOp:$res TF_FloatTensor:$input), (TF_RoundOp $input)>;

// Rounds on integers should just be bypassed.
def LowerRoundOpOnIntTensor : Pat<
  (TF_RoundOp:$res TF_IntTensor:$input),
  (TF_IdentityOp $input)>;

// Implements TF Round on floats using basic operations. TF Round is specified
// as RoundHalfToEven to be compatible with Numpy.
def LowerRoundOpOnFloatTensor : Pat<
  (TF_RoundOp:$res TF_FloatTensor:$input),
  (TF_SelectV2Op
    (TF_EqualOp
      (TF_ConstOp:$zero (GetScalarOfFloatType<"0.0"> $input)),
      (TF_SelectV2Op:$rounded
        (TF_LogicalOrOp
          (TF_GreaterOp
            (TF_SubOp:$fraction $input, (TF_FloorOp:$round_val $input)),
            (TF_ConstOp:$half (GetScalarOfFloatType<"0.5"> $input))
          ),
          (TF_LogicalAndOp
            // If the input is equal to 0.5 ...
            (TF_EqualOp
              $fraction,
              $half,
              /*incompatible_shape_error*/ConstBoolAttrTrue
            ),
            // ... check if the value is odd, so we can round up in that case.
            (TF_EqualOp
              (TF_SubOp
                $round_val,
                (TF_MulOp
                  (TF_ConstOp (GetScalarOfType<2> $input)),
                  (TF_FloorOp (TF_MulOp $half, $input))
                )
              ),
              (TF_ConstOp:$one (GetScalarOfType<1> $input)),
              /*incompatible_shape_error*/ConstBoolAttrTrue
            )
          )
        ),
        (TF_AddV2Op $round_val, $one),
        $round_val
      ),
      /*incompatible_shape_error*/ConstBoolAttrTrue
    ),
    $zero,
    $rounded
  )>;

//===----------------------------------------------------------------------===//
// Rsqrt op patterns.
//===----------------------------------------------------------------------===//

// RsqrtGrad(lhs, rhs) = (lhs * lhs * lhs) * (rhs / -2)
def LowerRsqrtGradOp : Pat<
  (TF_RsqrtGradOp $lhs, $rhs),
  (TF_MulOp
    (TF_MulOp
      (TF_MulOp $lhs, $lhs),
      $lhs
    ),
    (TF_DivOp
      $rhs,
      (TF_ConstOp
        (GetScalarOfType<-2> $rhs)
      )
    )
  )>;

//===----------------------------------------------------------------------===//
// Size op patterns.
//===----------------------------------------------------------------------===//

// Size(x) = Prod(Shape(x), reduction_indices=0, keep_dims=false)
def LowerSizeOp : Pat<
  (TF_SizeOp:$res $arg),
  (TF_ProdOp
    (CreateTFShapeOp
      $res,
      $arg,
      (IsI32 $res)
    ),
    /*reduction_indices=*/
    (TF_ConstOp
      (GetScalarOfType<0> $res)
    ),
    /*keep_dims=*/
    ConstBoolAttrFalse
  )>;

//===----------------------------------------------------------------------===//
// Sqrt op patterns.
//===----------------------------------------------------------------------===//

// SqrtGrad(y, dy) = dy * 0.5 / y
def LowerSqrtGradOp : Pat<
  (TF_SqrtGradOp $y, $dy),
  (TF_DivOp
    (TF_MulOp $dy, (TF_ConstOp (GetScalarOfFloatType<"0.5"> $dy))),
    $y
  )>;

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

class BinaryXopyPat<dag From, dag To> : Pat<
  From,
  (TF_SelectV2Op
    (TF_EqualOp
      $x,
      (TF_ConstOp
        (GetScalarOfType<0> $x)
      ),
      /*incompatible_shape_error*/ConstBoolAttrTrue
    ),
    $x,
    To
  )>;

def LowerXdivyOp : BinaryXopyPat<
  (TF_XdivyOp $x, $y),
  (TF_DivOp $x, $y)>;

def LowerXlog1pyOp : BinaryXopyPat<
  (TF_Xlog1pyOp $x, $y),
  (TF_MulOp $x, (TF_Log1pOp $y))>;

def LowerXlogyOp : BinaryXopyPat<
  (TF_XlogyOp $x, $y),
  (TF_MulOp $x, (TF_LogOp $y))>;

//===----------------------------------------------------------------------===//
// IsFinite op patterns.
//===----------------------------------------------------------------------===//

def LowerIsFiniteOp : Pat<(TF_IsFiniteOp $x),
                       (TF_EqualOp
                          (TF_SubOp $x, $x),
                          (TF_ConstOp (GetScalarOfType<0> $x)),
                          /*incompatible_shape_error*/ConstBoolAttrTrue)>;

//===----------------------------------------------------------------------===//
// TensorScatterUpdate op patterns.
//===----------------------------------------------------------------------===//

def LowerTensorScatterUpdate_1 : Pat<(TF_TensorScatterUpdateOp $input, $indices, $updates),
          (TF_CastOp
            (TF_TensorScatterUpdateOp
              (CreateTFCastOpI32
                $input,
                $input,
                /*truncate=*/ConstBoolAttrFalse),
              $indices,
              (CreateTFCastOpI32
                $updates,
                $updates,
                /*truncate=*/ConstBoolAttrFalse)),
             /*truncate=*/ConstBoolAttrFalse),
          [(TensorOf<[TF_Bool]> $input), (TensorOf<[TF_Bool]> $updates)] >;

def LowerTensorScatterUpdate_2 : Pat<(TF_TensorScatterUpdateOp $input, $indices, $updates),
          (TF_AddV2Op
             (TF_MulOp
                (TF_SubOp (TF_ConstOp (GetScalarOfType<1> $updates)),
                   (CreateTensorScatterNdOp $input, $indices,
                      (TF_OnesLikeOp $updates),
                   (CreateTFShapeOp $input, $input, ConstBoolAttrTrue))),
             $input),
           (CreateTensorScatterNdOp $input, $indices, $updates,
              (CreateTFShapeOp $input, $input, ConstBoolAttrTrue))),
          [(TensorOf<[TF_Int, TF_Float, TF_Complex]> $updates)]>;

//===----------------------------------------------------------------------===//
// Selu op patterns.
//===----------------------------------------------------------------------===//

def getScale : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0), 1.0507009873554804934193349852946)"
  >;

def getScaledAlpha : NativeCodeCall<
  "GetScalarOfType(getElementTypeOrSelf($0), 1.7580993408473768599402175208123)"
  >;

// Selu(x) = scale * x if x > 0 else scaledAlpha * (exp(x) - 1). Here
// scale is taken as a constant 1.0507009873554804934193349852946 and
// scaledAlpha  is taken as a constant 1.7580993408473768599402175208123.

def LowerSeluOp : Pat<(TF_SeluOp TF_FloatTensor:$features),
                  (TF_SelectV2Op
                    (TF_GreaterOp
                      $features,
                      (TF_ConstOp (GetScalarOfType<0> $features))),
                    (TF_MulOp
                      (TF_ConstOp (getScale $features)),
                      $features),
                    (TF_MulOp
                      (TF_ConstOp (getScaledAlpha $features)),
                      (TF_Expm1Op $features)))>;

// SeluGradOp(gradients, outputs) = gradients * scale if outputs > 0
// else gradients * (outputs + scaledAlpha). Here scale is taken as a
// constant 1.0507009873554804934193349852946 and scaledAlpha is taken
// as a constant 1.7580993408473768599402175208123.

def LowerSeluGradOp : Pat<(TF_SeluGradOp AnyStaticShapeTensor:$gradients,
                          TF_FloatTensor:$features),
                      (TF_SelectV2Op
                        (TF_GreaterOp
                          $features,
                          (TF_ConstOp (GetScalarOfType<0> $features))),
                        (TF_MulOp
                          $gradients,
                          (TF_ConstOp (getScale $features))),
                        (TF_MulOp
                          $gradients,
                          (TF_AddV2Op
                            $features,
                            (TF_ConstOp (getScaledAlpha $features)))))>;



//===----------------------------------------------------------------------===//
// Expm1 op patterns.
//===----------------------------------------------------------------------===//

// Expm1(x) = Exp(x) - 1
def LowerExp1mOp : Pat<
  (TF_Expm1Op $x),
  (TF_SubOp
    (TF_ExpOp $x),
    (TF_ConstOp (GetScalarOfType<1> $x))
  )>;
