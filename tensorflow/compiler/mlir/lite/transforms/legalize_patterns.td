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

// TFLite legalization patterns

include "mlir/IR/OpBase.td"
include "mlir/StandardOps/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def F32ElementsAttr : ElementsAttrBase<
  CPred<"$_self.cast<ElementsAttr>().getType().getElementType().isF32()">, "float constant tensor">;

// Extract the ith int element from an ArrayAttr $0 as an 32-bit IntegerAttr
// with builder.
class ExtractI32At<int i> : NativeCodeCall<
    "$_builder.getI32IntegerAttr($_self.cast<ArrayAttr>().getValue()[" # i #
    "].cast<IntegerAttr>().getInt())">;


// Merge the two Attributes to a ArrayAttr;
def Merge2AttrsToArray : NativeCodeCall<"$_builder.getArrayAttr({$0, $1})">;

// Use the tensor type information from $0 and convert min $1, max $2 and
// storage type $3 to a QuantizedType.
def ConvertToQuantType : NativeCodeCall<
    "GetQuantizedTypeAttr($_builder, $0->getType(), $1, $2, $3.getValue())">;

// Use the tensor type information from $0 and convert min $1, max $2 and
// numBits $3 and narrowRange $4 to a QuantizedType.
def ConvertToQuantTypeFromAttrs : NativeCodeCall<
    "GetQuantizedTypeAttr($_builder, $0->getType(), $1, $2, $3, $4)">;

// Predicate that holds if all the three attributes are set.
def HasAll3Attrs : Constraint<CPred<"HasAll3Attrs($0, $1, $2)">>;

// Converts an integer attribute $0 to 32-bit with builder.
def convertIntAttrTo32Bit : NativeCodeCall<
    "$_builder.getI32IntegerAttr($0.cast<IntegerAttr>().getInt())">;

// Extracts the single integer element from $_self.
def ExtractSingleElementAsInteger : NativeCodeCall<
    "ExtractSingleElementAsInteger($_self.cast<ElementsAttr>())">;

//===----------------------------------------------------------------------===//
// Nullary ops patterns.
//===----------------------------------------------------------------------===//
def : Pat<(TF_ConstOp ElementsAttr:$value), (TFL_ConstOp $value)>;

//===----------------------------------------------------------------------===//
// Unary ops patterns.
//===----------------------------------------------------------------------===//
def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "NHWC">;
def IsIntList1XY1 : AttrConstraint<CPred<"TFIntListIs1XY1($_self)">>;

def : Pat<(TF_AbsOp $arg), (TFL_AbsOp $arg)>;

def : Pat<(TF_AvgPoolOp $value,
              IsIntList1XY1:$ksize,
              IsIntList1XY1:$strides,
              $padding,
              IsDataFormatNHWC:$format),
          (TFL_AveragePool2DOp $value,
              /*filter_height=*/ExtractI32At<1>:$ksize,
              /*filter_width=*/ExtractI32At<2>:$ksize,
              /*padding=*/$padding,
              /*stride_h=*/ExtractI32At<1>:$strides,
              /*stride_w=*/ExtractI32At<2>:$strides,
              /*fused_activation_function=*/TFL_AF_None)>;

def : Pat<(TF_ArgMaxOp $input, $dim), (TFL_ArgMaxOp $input, $dim)>;
def : Pat<(TF_ArgMinOp $input, $dim), (TFL_ArgMinOp $input, $dim)>;

def : Pat<(TF_CeilOp $arg), (TFL_CeilOp $arg)>;

def : Pat<(TF_CosOp $arg), (TFL_CosOp $arg)>;

def : Pat<(TF_EluOp $arg), (TFL_EluOp $arg)>;

def : Pat<(TF_ExpandDimsOp $input, $dim), (TFL_ExpandDimsOp $input, $dim)>;

def : Pat<(TF_FakeQuantWithMinMaxArgsOp $inputs,
              $min, $max,
              $num_bits, $narrow_range),
          (TFL_DequantizeOp
              (TFL_QuantizeOp $inputs,
                 (ConvertToQuantTypeFromAttrs $inputs, $min, $max,
                     $num_bits, $narrow_range)))>;

def : Pat<(TF_FillOp $arg, $value), (TFL_FillOp $arg, $value)>;

def : Pat<(TF_FloorOp $arg), (TFL_FloorOp $arg)>;

def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>;
def : Pat<(TF_LogicalNotOp $arg), (TFL_LogicalNotOp $arg)>;
def : Pat<(TF_LogSoftmaxOp $arg), (TFL_LogSoftmaxOp $arg)>;

def : Pat<(TF_MaxPoolOp $value,
              IsIntList1XY1:$ksize,
              IsIntList1XY1:$strides,
              $padding,
              IsDataFormatNHWC:$format),
          (TFL_MaxPool2DOp $value,
              /*padding=*/$padding,
              /*stride_w=*/ExtractI32At<2>:$strides,
              /*stride_h=*/ExtractI32At<1>:$strides,
              /*filter_width=*/ExtractI32At<2>:$ksize,
              /*filter_height=*/ExtractI32At<1>:$ksize,
              /*fused_activation_function=*/TFL_AF_None)>;

def : Pat<(TF_MaximumOp $arg1, $arg2), (TFL_MaximumOp $arg1, $arg2)>;
def : Pat<(TF_MinimumOp $arg1, $arg2), (TFL_MinimumOp $arg1, $arg2)>;
def : Pat<(TF_OneHotOp $indices, $depth, $on_value, $off_value, $axis),
          (TFL_OneHotOp $indices, $depth, $on_value, $off_value,
           (convertIntAttrTo32Bit $axis))>;
def : Pat<(TF_RangeOp $start, $limit, $delta), (TFL_RangeOp $start, $limit, $delta)>;
def : Pat<(TF_Relu6Op $arg), (TFL_Relu6Op $arg)>;
def : Pat<(TF_ReluOp $arg), (TFL_ReluOp $arg)>;
// The second operand is captured in the type for this transform.
def : Pat<(TF_ReshapeOp:$res AnyStaticShapeTensor:$arg, $ignored),
          (TFL_ReshapeOp $arg), [(AnyStaticShapeTensor $res)]>;
def : Pat<(TF_ReverseSequenceOp $input, $seq_lengths, $seq_dim, $batch_dim),
          (TFL_ReverseSequenceOp $input, $seq_lengths,
           (convertIntAttrTo32Bit $seq_dim),
           (convertIntAttrTo32Bit $batch_dim))>;
def : Pat<(TF_RsqrtOp $arg), (TFL_RsqrtOp $arg)>;
// TODO(jpienaar): this is not true for all selects, TF's select supports rank 0
// condition
def : Pat<(TF_SelectOp $cond, $x, $y), (TFL_SelectOp $cond, $x, $y)>;
def : Pat<(TF_ShapeOp $arg), (TFL_ShapeOp $arg)>;
def : Pat<(TF_SigmoidOp $arg), (TFL_LogisticOp $arg)>;
def : Pat<(TF_SinOp F32Tensor:$arg), (TFL_SinOp $arg)>;
def : Pat<(TF_SliceOp $input, $begin, $size), (TFL_SliceOp $input, $begin, $size)>;
def : Pat<(TF_SoftmaxOp $arg), (TFL_SoftmaxOp $arg, ConstF32Attr<"1.0">)>;
def : Pat<(TF_SqueezeOp $arg, $squeeze_dims), (TFL_SqueezeOp $arg, $squeeze_dims)>;
def : Pat<(TF_TanhOp $arg), (TFL_TanhOp $arg)>;
def : Pat<(TF_TransposeOp $arg, $perm), (TFL_TransposeOp $arg, $perm)>;
def : Pat<(TF_ZerosLikeOp $arg), (TFL_ZerosLikeOp $arg)>;

// The following two rules can both match an tf.Placeholder.input node with
// min/max/type attributes, so we increase the benefit of the first rule by one
// so the tfl.quantize and tfl.dequantize ops will be inserted if it matches.
def : Pat<(TF_PlaceholderInputOp $inputs, $min, $max, $type),
          (TFL_DequantizeOp
              (TFL_QuantizeOp
                  (TFL_InputOp $inputs),
                  (ConvertToQuantType $inputs, $min, $max, $type))),
          [(HasAll3Attrs $min, $max, $type)], (addBenefit 1)>;
def : Pat<(TF_PlaceholderInputOp $inputs, $min, $max, $type),
          (TFL_InputOp $inputs)>;

//===----------------------------------------------------------------------===//
// Binary ops patterns.
//===----------------------------------------------------------------------===//
def : Pat<(TF_LessOp $l, $r), (TFL_LessOp $l, $r)>;
def : Pat<(TF_GreaterOp $l, $r), (TFL_GreaterOp $l, $r)>;

def : Pat<(TF_LessEqualOp $l, $r), (TFL_LessEqualOp $l, $r)>;
def : Pat<(TF_GreaterEqualOp $l, $r), (TFL_GreaterEqualOp $l, $r)>;

// Gather in TF -> Gather in TFL with axis=0
// The 'validate_indices' attribute is deprecated.
def : Pat<(TF_GatherOp $params, $indices, $ignored_validate_indices),
          (TFL_GatherOp $params, $indices, ConstantAttr<I32Attr, "0">)>;

def : Pat<(TF_GatherNdOp $params, $indices),
          (TFL_GatherNdOp $params, $indices)>;

def : Pat<(TF_GatherV2Op $params, $indices,
                         (ConstantOp ElementsAttr:$axis),
                         ConstantAttr<I64Attr, "0">:$batch_dims),
          (TFL_GatherOp $params, $indices,
                        ExtractSingleElementAsInteger:$axis)>;

def : Pat<(TF_FloorDivOp $l, $r), (TFL_FloorDivOp $l, $r)>;

def : Pat<(TF_NotEqualOp $l, $r), (TFL_NotEqualOp $l, $r)>;

def : Pat<(TF_LogicalAndOp $l, $r), (TFL_LogicalAndOp $l, $r)>;

def : Pat<(TF_LogicalOrOp $l, $r), (TFL_LogicalOrOp $l, $r)>;

// Multi-pattern consisting of matching stand-alone op or op followed by relu.
multiclass FusedBinaryActivationFuncOpPat<dag FromOp, dag ToOp> {
  def : Pat<(FromOp AnyTensor:$l, AnyTensor:$r),
            (ToOp $l, $r, TFL_AF_None)>;
  foreach actFnPair = [[TF_ReluOp, TFL_AF_Relu],
                       [TF_Relu6Op, TFL_AF_Relu6]] in {
    def : Pat<(actFnPair[0] (FromOp $lhs, $rhs)),
              (ToOp $lhs, $rhs, actFnPair[1])>;
    // TODO: Maybe move these below to general pass?
    def : Pat<(actFnPair[0] (ToOp $lhs, $rhs, TFL_AF_None)),
              (ToOp $lhs, $rhs, actFnPair[1])>;
  }
}

// Instantiated FusedBinary patterns for the from-to pairs of ops.
foreach fromToPair = [[TF_AddOp, TFL_AddOp],
                      [TF_AddV2Op, TFL_AddOp],
                      [TF_DivOp, TFL_DivOp],
                      [TF_MulOp, TFL_MulOp],
                      [TF_RealDivOp, TFL_DivOp],
                      [TF_SubOp, TFL_SubOp]] in
  defm : FusedBinaryActivationFuncOpPat<fromToPair[0], fromToPair[1]>;

def : Pat<(TF_BiasAddOp F32Tensor:$l, F32Tensor:$r,
                        IsDataFormatNHWC:$data_format),
          (TFL_AddOp $l, $r, TFL_AF_None)>;
// TODO(jpienaar): These should be handled by the pattern rewriter, find out
// why it isn't.
def : Pat<(TF_Relu6Op (TF_BiasAddOp F32Tensor:$l, F32Tensor:$r,
                                    IsDataFormatNHWC:$data_format)),
          (TFL_AddOp $l, $r, TFL_AF_Relu6)>;

def : Pat<(TF_FakeQuantWithMinMaxVarsOp $inputs,
              (ConstantOp F32ElementsAttr:$min),
              (ConstantOp F32ElementsAttr:$max),
              $num_bits, $narrow_range),
          (TFL_DequantizeOp
              (TFL_QuantizeOp $inputs,
                 (ConvertToQuantTypeFromAttrs $inputs, $min, $max,
                     $num_bits, $narrow_range)))>;

def : Pat<(TF_RankOp $input), (TFL_RankOp $input)>;

def : Pat<(TF_SquaredDifferenceOp $l, $r), (TFL_SquaredDifferenceOp $l, $r)>;

// Note(ycling): We can eliminate Relu from Relu(SquaredDifference(x, y)),
// since the result of SquaredDifference is always non-negative.
// TFLite interpreter doesn't support Relu+int32 for now. So the test cases
// are failing without the following pattern to optimize Relu away fixes
// the problem.
def : Pat<(TF_ReluOp (TF_SquaredDifferenceOp $l, $r)),
          (TFL_SquaredDifferenceOp $l, $r)>;

def : Pat<(TF_ReverseV2Op $arg0, $arg1), (TFL_ReverseV2Op $arg0, $arg1)>;

def : Pat<(TF_EqualOp $arg0, $arg1), (TFL_EqualOp $arg0, $arg1)>;

def : Pat<(TF_PadOp $arg0, $arg1), (TFL_PadOp $arg0, $arg1)>;

def : Pat<(TF_PadV2Op $arg0, $arg1, $cst), (TFL_PadV2Op $arg0, $arg1, $cst)>;

def : Pat<(TF_MeanOp $arg0, $arg1, BoolAttr:$arg2), (TFL_MeanOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_SumOp $arg, $axes, BoolAttr:$arg2), (TFL_SumOp $arg, $axes, $arg2)>;

// TopK in TFL is always sorted so we ignore that attribute here.
def : Pat<(TF_TopKV2Op $input, $k, $ignored_sorted), (TFL_TopKV2Op $input, $k)>;

def : Pat<(TF_MinOp $arg0, $arg1, BoolAttr:$arg2), (TFL_ReduceMinOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_MaxOp $arg0, $arg1, BoolAttr:$arg2), (TFL_ReduceMaxOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_ProdOp $arg0, $arg1, BoolAttr:$arg2), (TFL_ReduceProdOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_CastOp $arg0, BoolAttr:$arg1), (TFL_CastOp $arg0)>;

def : Pat<(TF_BatchToSpaceNDOp $input, $block_shape, $crops), (TFL_BatchToSpaceNdOp $input, $block_shape, $crops)>;

def : Pat<(TF_SpaceToBatchNDOp $input, $block_shape, $paddings), (TFL_SpaceToBatchNdOp $input, $block_shape, $paddings)>;

def : Pat<(TF_SpaceToDepthOp $input, $block_size, IsDataFormatNHWC:$data_format),
          (TFL_SpaceToDepthOp $input, (convertIntAttrTo32Bit $block_size))>;

def : Pat<(TF_ResizeBilinearOp $images, $size, $align_corners, ConstBoolAttrFalse:$half_pixel_centers), (TFL_ResizeBilinearOp $images, $size, $align_corners)>;

def : Pat<(TF_MirrorPadOp $arg0, $arg1, $cst), (TFL_MirrorPadOp $arg0, $arg1, $cst)>;

def : Pat<
  (TF_StridedSliceOp $input, $begin, $end, $strides, $begin_mask, $end_mask, $ellipsis_mask, $new_axis_mask, $shrink_axis_mask),
  (TFL_StridedSliceOp $input, $begin, $end, $strides,
    (convertIntAttrTo32Bit $begin_mask), (convertIntAttrTo32Bit $end_mask), (convertIntAttrTo32Bit $ellipsis_mask),
    (convertIntAttrTo32Bit $new_axis_mask), (convertIntAttrTo32Bit $shrink_axis_mask))>;

def : Pat<(TF_UniqueOp $arg0),(TFL_UniqueOp $arg0)>;

def : Pat<(TF_LRNOp $arg0, $radius, F32Attr:$bias, F32Attr:$alpha, F32Attr:$beta), (TFL_LocalResponseNormalizationOp $arg0, (convertIntAttrTo32Bit $radius), $bias, $alpha, $beta)>;
