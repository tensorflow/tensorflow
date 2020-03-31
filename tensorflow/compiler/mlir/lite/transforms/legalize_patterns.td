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
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def NonOpaqueElementsAttr : ElementsAttrBase<
  CPred<"!$_self.isa<OpaqueElementsAttr>()">,
  "non-opaque constant tensor">;

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
// numBits $3 and narrowRange $4 to a QuantizedType.
def ConvertToQuantTypeFromAttrs : NativeCodeCall<
    "quant::GetQuantizedTypeAttr($_builder, $0.getType(), $1, $2, -1, $3, $4, /*is_signed=*/false)">;

// Converts an integer attribute $0 to 32-bit with builder.
def convertIntAttrTo32Bit : NativeCodeCall<
    "$_builder.getI32IntegerAttr($0.cast<IntegerAttr>().getInt())">;

// Extracts the single integer element from $_self.
def ExtractSingleElementAsInteger : NativeCodeCall<
    "ExtractSingleElementAsInteger($_self.cast<ElementsAttr>())">;

// Extracts the single int32 element from $_self.
def ExtractSingleElementAsInt32 : NativeCodeCall<
    "$_builder.getI32IntegerAttr(ExtractSingleElementAsInteger($_self.cast<ElementsAttr>()).getInt())">;

// Checks whether the given operation has static shapes and same shapes of all inputs.
def HasSameStaticShapesPred : CPred<"HasSameStaticShapes($0.getDefiningOp())">;
def HasSameStaticShapes : Constraint<HasSameStaticShapesPred, "op must have static same input shapes">;
def HasNotSameStaticShapes : Constraint<Neg<HasSameStaticShapesPred>, "op must have not static same input shapes">;

// Checks if the value has only one user.
// TODO(karimnosseir): Move to a common place?
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

//===----------------------------------------------------------------------===//
// Nullary ops patterns.
//===----------------------------------------------------------------------===//

def : Pat<(TF_ConstOp ElementsAttr:$value), (TFL_ConstOp $value)>;

// Convert to std constant for statically shaped, non-opaque constants.
def : Pat<(TF_ConstOp:$res NonOpaqueElementsAttr:$value), (ConstantOp $value),
          [(AnyStaticShapeTensor $res)], (addBenefit 10)>;

//===----------------------------------------------------------------------===//
// Unary ops patterns.
//===----------------------------------------------------------------------===//
def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "NHWC">;
def IsIntList1XY1 : AttrConstraint<CPred<"TFIntListIs1XY1($_self)">>;
def IsAllOnes : AttrConstraint<CPred<"TFIntListIsAllOnes($_self)">>;
def IsSameOrValid : AttrConstraint<
    CPred<"$_self.cast<StringAttr>().getValue() == \"SAME\" || " #
          "$_self.cast<StringAttr>().getValue() == \"VALID\"">,
    "'SAME' or 'VALID' paddings">;

def : Pat<(TF_AbsOp $arg), (TFL_AbsOp $arg)>;
def : Pat<(TF_AddNOp $inputs), (TFL_AddNOp $inputs)>;

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
def : Pat<(TF_LogOp $arg), (TFL_LogOp $arg)>;
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
def : Pat<(TF_NegOp $arg), (TFL_NegOp $arg)>;
def : Pat<(TF_OneHotOp $indices, $depth, $on_value, $off_value, $axis),
          (TFL_OneHotOp $indices, $depth, $on_value, $off_value,
           (convertIntAttrTo32Bit $axis))>;
def : Pat<(TF_PowOp $x, $y), (TFL_PowOp $x, $y)>;
def : Pat<(TF_RangeOp $start, $limit, $delta), (TFL_RangeOp $start, $limit, $delta)>;
def : Pat<(TF_Relu6Op $arg), (TFL_Relu6Op $arg)>;
def : Pat<(TF_ReluOp $arg), (TFL_ReluOp $arg)>;
def : Pat<(TF_ReverseSequenceOp $input, $seq_lengths, $seq_dim, $batch_dim),
          (TFL_ReverseSequenceOp $input, $seq_lengths,
           (convertIntAttrTo32Bit $seq_dim),
           (convertIntAttrTo32Bit $batch_dim))>;
def : Pat<(TF_RoundOp $arg), (TFL_RoundOp $arg)>;
def : Pat<(TF_RsqrtOp $arg), (TFL_RsqrtOp $arg)>;
def : Pat<(TF_SqrtOp $arg), (TFL_SqrtOp $arg)>;
def : Pat<(TF_SquareOp $arg), (TFL_SquareOp $arg)>;
def : Pat<(TF_SegmentSumOp $data, I32Tensor:$segment_ids), (TFL_SegmentSumOp $data, $segment_ids)>;
def : Pat<(TF_SelectOp $cond, $x, $y), (TFL_SelectOp $cond, $x, $y)>;
def : Pat<(TF_SelectV2Op:$src_op $cond, $x, $y), (TFL_SelectOp $cond, $x, $y), [(HasSameStaticShapes $src_op)]>;
def : Pat<(TF_SelectV2Op:$src_op $cond, $x, $y), (TFL_SelectV2Op $cond, $x, $y), [(HasNotSameStaticShapes $src_op)]>;
def : Pat<(TF_ShapeOp $arg), (TFL_ShapeOp $arg)>;
def : Pat<(TF_SigmoidOp $arg), (TFL_LogisticOp $arg)>;
def : Pat<(TF_SinOp F32Tensor:$arg), (TFL_SinOp $arg)>;
def : Pat<(TF_SliceOp $input, $begin, $size), (TFL_SliceOp $input, $begin, $size)>;
def : Pat<(TF_SoftmaxOp $arg), (TFL_SoftmaxOp $arg, ConstF32Attr<"1.0">)>;
def : Pat<(TF_SoftplusOp F32Tensor:$arg0), (TFL_LogOp (TFL_AddOp (TFL_ExpOp $arg0), (ConstantOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">), TFL_AF_None))>;
def : Pat<(TF_SqueezeOp $arg, $squeeze_dims), (TFL_SqueezeOp $arg, $squeeze_dims)>;
def : Pat<(TF_TanhOp $arg), (TFL_TanhOp $arg)>;
def : Pat<(TF_TransposeOp $arg, $perm), (TFL_TransposeOp $arg, $perm)>;
def : Pat<(TF_WhereOp $arg), (TFL_WhereOp $arg)>;
def : Pat<(TF_ZerosLikeOp $arg), (TFL_ZerosLikeOp $arg)>;

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
                        ExtractSingleElementAsInt32:$axis)>;

def : Pat<(TF_FloorDivOp $l, $r), (TFL_FloorDivOp $l, $r)>;

def : Pat<(TF_NotEqualOp $l, $r, /*incompatible_shape_error=*/ConstBoolAttrTrue),
          (TFL_NotEqualOp $l, $r)>;

def : Pat<(TF_LogicalAndOp $l, $r), (TFL_LogicalAndOp $l, $r)>;

def : Pat<(TF_LogicalOrOp $l, $r), (TFL_LogicalOrOp $l, $r)>;

def : Pat<(TF_AddOp $lhs, $rhs), (TFL_AddOp $lhs, $rhs, TFL_AF_None)>;
def : Pat<(TF_AddV2Op $lhs, $rhs), (TFL_AddOp $lhs, $rhs, TFL_AF_None)>;
def : Pat<(TF_SubOp $lhs, $rhs), (TFL_SubOp $lhs, $rhs, TFL_AF_None)>;
def : Pat<(TF_MulOp $lhs, $rhs), (TFL_MulOp $lhs, $rhs, TFL_AF_None)>;
def : Pat<(TF_RealDivOp $lhs, $rhs), (TFL_DivOp $lhs, $rhs, TFL_AF_None)>;
def : Pat<(TF_DivOp $lhs, $rhs), (TFL_DivOp $lhs, $rhs, TFL_AF_None)>;

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

// TODO(rocky): Not all of the attributes are handled correctly.  Make this
// more general if there is a need.
def : Pat<(TF_QuantizeAndDequantizeV2Op $inputs,
              (ConstantOp F32ElementsAttr:$min),
              (ConstantOp F32ElementsAttr:$max),
              $signed_input, $num_bits, $range_given, $round_mode,
              $narrow_range, $axis),
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

def : Pat<(TF_EqualOp $arg0, $arg1, /*incompatible_shape_error=*/ConstBoolAttrTrue), (TFL_EqualOp $arg0, $arg1)>;

def : Pat<(TF_PadOp $arg0, $arg1), (TFL_PadOp $arg0, $arg1)>;

def : Pat<(TF_TileOp $arg0, $arg1), (TFL_TileOp $arg0, $arg1)>;

def : Pat<(TF_PadV2Op $arg0, $arg1, $cst), (TFL_PadV2Op $arg0, $arg1, $cst)>;

def : Pat<(TF_MeanOp $arg0, $arg1, BoolAttr:$arg2), (TFL_MeanOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_SumOp $arg, $axes, BoolAttr:$arg2), (TFL_SumOp $arg, $axes, $arg2)>;

// TopK in TFL is always sorted so we ignore that attribute here.
def : Pat<(TF_TopKV2Op $input, $k, $ignored_sorted), (TFL_TopKV2Op $input, $k)>;

def : Pat<(TF_MinOp $arg0, $arg1, BoolAttr:$arg2), (TFL_ReduceMinOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_MaxOp $arg0, $arg1, BoolAttr:$arg2), (TFL_ReduceMaxOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_ProdOp $arg0, $arg1, BoolAttr:$arg2), (TFL_ReduceProdOp $arg0, $arg1, $arg2)>;

def : Pat<(TF_AnyOp $input, $reduction_indices, $keep_dims),
          (TFL_ReduceAnyOp $input, $reduction_indices, $keep_dims)>;

def : Pat<(TF_CastOp $arg0, BoolAttr:$arg1), (TFL_CastOp $arg0)>;

def : Pat<(TF_BatchToSpaceNDOp $input, $block_shape, $crops), (TFL_BatchToSpaceNdOp $input, $block_shape, $crops)>;

def : Pat<(TF_SpaceToBatchNDOp $input, $block_shape, $paddings), (TFL_SpaceToBatchNdOp $input, $block_shape, $paddings)>;

def : Pat<(TF_SpaceToDepthOp $input, $block_size, IsDataFormatNHWC:$data_format),
          (TFL_SpaceToDepthOp $input, (convertIntAttrTo32Bit $block_size))>;

def : Pat<(TF_DepthToSpaceOp $input, $block_size, IsDataFormatNHWC:$data_format),
          (TFL_DepthToSpaceOp $input, (convertIntAttrTo32Bit $block_size))>;

def : Pat<(TF_ResizeBilinearOp $images, $size, $align_corners, $half_pixel_centers), (TFL_ResizeBilinearOp $images, $size, $align_corners, $half_pixel_centers)>;
def : Pat<(TF_ResizeNearestNeighborOp $images, $size, $align_corners, ConstBoolAttrFalse:$half_pixel_centers), (TFL_ResizeNearestNeighborOp $images, $size, $align_corners)>;

def : Pat<(TF_MirrorPadOp $arg0, $arg1, $cst), (TFL_MirrorPadOp $arg0, $arg1, $cst)>;

def : Pat<(TF_SparseToDenseOp $sparse_indices, $output_shape, $sparse_values, $default_value, $validate_indices),
          (TFL_SparseToDenseOp $sparse_indices, $output_shape, $sparse_values, $default_value)>;

def : Pat<(TF_UniqueOp $arg0),(TFL_UniqueOp $arg0)>;

def : Pat<(TF_FloorModOp $arg0, $arg1), (TFL_FloorModOp $arg0, $arg1)>;
def : Pat<(TF_ExpOp $arg0), (TFL_ExpOp $arg0)>;

def : Pat<(TF_LRNOp $arg0, $radius, F32Attr:$bias, F32Attr:$alpha, F32Attr:$beta), (TFL_LocalResponseNormalizationOp $arg0, (convertIntAttrTo32Bit $radius), $bias, $alpha, $beta)>;

def : Pat<
  (TF_NonMaxSuppressionV4Op $boxes, $scores, $max_output_size, $iou_threshold, $score_threshold, $pad_to_max_output_size),
  (TFL_NonMaxSuppressionV4Op $boxes, $scores, $max_output_size, $iou_threshold, $score_threshold)>;

def : Pat<
  (TF_NonMaxSuppressionV5Op $boxes, $scores, $max_output_size, $iou_threshold, $score_threshold, $soft_nms_sigma, $pad_to_max_output_size),
  (TFL_NonMaxSuppressionV5Op $boxes, $scores, $max_output_size, $iou_threshold, $score_threshold, $soft_nms_sigma)>;

def : Pat<(TF_MatrixDiagOp $diagonal), (TFL_MatrixDiagOp $diagonal)>;

class I32VectorElementsAttr<int len> : ElementsAttrBase<
  CPred<"$_self.isa<DenseIntElementsAttr>() &&"
      "$_self.cast<DenseIntElementsAttr>().getType()."
      "getElementType().isSignlessInteger(32)">,
  "32-bit int elements attribute of shape [" # len # "]"> {

  let storageType = [{ DenseIntElementsAttr }];
  let returnType = [{ DenseIntElementsAttr }];

  let constBuilderCall = "DenseElementsAttr::get("
    "RankedTensorType::get({" # len # "}, $_builder.getIntegerType(32)), $0)";
}

def : Pat<
  (TF_Conv2DBackpropInputOp $input_sizes, $filter, $out_backprop,
     IsIntList1XY1:$strides,
     BoolAttr:$use_cudnn_on_gpu,
     IsSameOrValid:$padding,
     I64ArrayAttr:$explicit_paddings,
     IsDataFormatNHWC:$data_format,
     IsAllOnes:$dilations),
  (TFL_TransposeConvOp $input_sizes,
     (TF_TransposeOp $filter,
       (ConstantOp ConstantAttr<I32VectorElementsAttr<4>, "{2, 0, 1, 3}">)),
     $out_backprop,
     /*padding=*/ $padding,
     /*stride_h=*/ ExtractI32At<1>:$strides,
     /*stride_w=*/ ExtractI32At<2>:$strides)>;

def : Pat<
  (TF_MatrixSetDiagOp $input, $diagonal),
  (TFL_MatrixSetDiagOp $input, $diagonal)>;
