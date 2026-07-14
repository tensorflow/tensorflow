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

// This is the canonicalize pattern definition file.

include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.td"

/// TODO(b/130756570): Support OpBase constraints in PatternRewrites.
def SingleResultAndOperandHaveSameElementType : Constraint<
  CPred<"getElementTypeOrSelf($0) == getElementTypeOrSelf($1)">>;

def SingleResultAndOperandHaveSameType : Constraint<
  CPred<"$0.getType() == $1.getType()">>;

def IsRank2Tensor : Type<HasAnyRankOfPred<[2]>, "Rank 2 tensor">;

def IsRank4Tensor : Type<HasAnyRankOfPred<[4]>, "Rank 4 tensor">;

// Checks if all the users is ReadVariableOp.
def HasOnlyReadVariableOpUsers : Constraint<
  CPred<"llvm::all_of($0.getUsers(), [](mlir::OpOperand op) { "
        "return llvm::isa<mlir::TF::ReadVariableOp>(op.getOwner()); })">>;

def CopyAttrs: NativeCodeCallVoid<"CopyDeviceAndUnderscoredAttributesAdaptor($0, $1)">;

//===----------------------------------------------------------------------===//
// Add op patterns.
//===----------------------------------------------------------------------===//

def AddToAddV2 : Pat<
  (TF_AddOp:$src TF_NumberTensor:$arg0, TF_NumberTensor:$arg1),
  (TF_AddV2Op:$dest $arg0, $arg1), [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// AddV2 op patterns.
//===----------------------------------------------------------------------===//

def AddV2OfNegLeft : Pat<
  (TF_AddV2Op:$src (TF_NegOp $arg0), $arg1), (TF_SubOp:$dest $arg1, $arg0),
  [], [(CopyAttrs $src, $dest)]>;

def AddV2OfNegRight : Pat<
  (TF_AddV2Op:$src $arg0, (TF_NegOp $arg1)), (TF_SubOp:$dest $arg0, $arg1), [],
  [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// Gather op patterns.
//===----------------------------------------------------------------------===//

class GetI32Attr<int x>: NativeCodeCall<
  "$_builder.getI32IntegerAttr(" # x # ")">;

class GetI64Attr<int x>: NativeCodeCall<
  "$_builder.getI64IntegerAttr(" # x # ")">;

// The deprecated validate_indices attribute is NOT verified during
// canonicalization because it is neither supported in the old bridge
// nor in the reference TF implementation.
def GatherToV2 : Pat<
  (TF_GatherOp:$src $params, $indices, $ignored_validate_indices),
  (TF_GatherV2Op:$dest $params, $indices, (TF_ConstOp (GetI32Attr<0>)),
    (GetI64Attr<0>)),
  [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// BatchMatMul op patterns.
//===----------------------------------------------------------------------===//

// Static shaped operands in a legal BatchMatMul op will have matching batch
// dimensions and can be upgraded to the BatchMatMulV2 op. Canonicalizing
// dynamically shaped operands is not correct as that will execute ops that
// have non matching batch dimensions but are broadcastable which should fail
// with V1.
def BatchMatMulToV2 : Pat<
  (TF_BatchMatMulOp:$src AnyStaticShapeTensor:$x, AnyStaticShapeTensor:$y,
    $adj_x, $adj_y, $grad_x, $grad_y),
  (TF_BatchMatMulV2Op:$dest $x, $y, $adj_x, $adj_y, $grad_x, $grad_y),
  [], [(CopyAttrs $src, $dest)]>;

def BatchMatMulToMatMul : Pat<
  (TF_BatchMatMulOp:$src $x, $y, $adj_x, $adj_y, $grad_x, $grad_y),
  (TF_MatMulOp:$dest $x, $y, $adj_x, $adj_y, $grad_x, $grad_y),
  [(IsRank2Tensor $x), (IsRank2Tensor $y)],
  [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// BatchMatMulV2 op patterns.
//===----------------------------------------------------------------------===//

def BatchMatMulV2ToMatMul : Pat<
  (TF_BatchMatMulV2Op:$src $x, $y, $adj_x, $adj_y, $grad_x, $grad_y),
  (TF_MatMulOp:$dest $x, $y, $adj_x, $adj_y, $grad_x, $grad_y),
  [(IsRank2Tensor $x), (IsRank2Tensor $y)],
  [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// BatchToSpace op patterns.
//===----------------------------------------------------------------------===//

def BatchToSpaceBlockSizeToBlockShape : NativeCodeCall<
  "DenseElementsAttr::get(RankedTensorType::get({2}, $_builder.getI64Type()), "
  "ArrayRef<APInt>{$0.getValue(), $0.getValue()})">;

def BatchToSpaceToBatchToSpaceND : Pat<
  (TF_BatchToSpaceOp:$src $input, $crops, $block_size),
  (TF_BatchToSpaceNDOp:$dest $input,
    (TF_ConstOp (BatchToSpaceBlockSizeToBlockShape $block_size)), $crops),
  [(IsRank4Tensor $input), (IsRank2Tensor $crops)], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// BiasAddV1 op patterns.
//===----------------------------------------------------------------------===//

def BiasAddV1ToBiasAdd : Pat<
  (TF_BiasAddV1Op:$src $arg0, $arg1),
  (TF_BiasAddOp:$dest $arg0, $arg1,
    ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">),
  [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// Bitcast op patterns.
//===----------------------------------------------------------------------===//

// Parameter attributes are part of function attributes, which is not affected
// by pattern rewrites. Therefore we do not need to explicilty copy attributes
// over.
def BitcastSameType : Pat<
  (TF_BitcastOp:$res $arg),
  (replaceWithValue $arg),
  [(SingleResultAndOperandHaveSameElementType $res, $arg)]>;

def BitcastNested : Pat<
  (TF_BitcastOp:$src (TF_BitcastOp $arg)), (TF_BitcastOp:$dest $arg), [],
  [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// Concat op patterns.
//===----------------------------------------------------------------------===//

def ConvertToConcatV2 : Pat<
  (TF_ConcatOp:$src $axis, $inputs), (TF_ConcatV2Op:$dest $inputs, $axis),
  [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// Div op patterns.
//===----------------------------------------------------------------------===//

/// Favor Mul over Div.
def DivWithSqrtDivisor : Pat<
  (TF_DivOp:$src $arg0, (TF_SqrtOp $arg1)),
  (TF_MulOp:$dest1 $arg0, (TF_RsqrtOp:$dest2 $arg1)), [],
  [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

//===----------------------------------------------------------------------===//
// Log op patterns.
//===----------------------------------------------------------------------===//

def LogOfSoftmax : Pat<
  (TF_LogOp:$src (TF_SoftmaxOp $arg)), (TF_LogSoftmaxOp:$dest $arg), [],
  [(CopyAttrs $src, $dest)]>;

// Canonicalize: Log(1.0 + x) to Log1p(x)
//
// We currently do this rewrite only if the constant `1` is a scalar, because
// it is safely broadcastable to any shape. To be able to canonicalize when
// constant values is not a scalar, we have to first prove that it is
// broadcastable to `x`, which requires static shape information.
def LogToLog1p : Pat<
  (TF_LogOp:$src (TF_AddV2Op $arg,
    (TF_ConstOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">))),
  (TF_Log1pOp:$dest $arg), [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// LogicalNot op patterns.
//===----------------------------------------------------------------------===//

def LogicalNotOfEqual : Pat<
  (TF_LogicalNotOp:$src (TF_EqualOp $arg0, $arg1, $shape_error)),
  (TF_NotEqualOp:$dest $arg0, $arg1, $shape_error), [],
  [(CopyAttrs $src, $dest)]>;

def LogicalNotOfNotEqual : Pat<
  (TF_LogicalNotOp:$src (TF_NotEqualOp $arg0, $arg1, $shape_error)),
  (TF_EqualOp:$dest $arg0, $arg1, $shape_error), [], [(CopyAttrs $src, $dest)]>;

def LogicalNotOfGreater : Pat<
  (TF_LogicalNotOp:$src (TF_GreaterOp $arg0, $arg1)),
  (TF_LessEqualOp:$dest $arg0, $arg1), [], [(CopyAttrs $src, $dest)]>;

def LogicalNotOfGreaterEqual : Pat<
  (TF_LogicalNotOp:$src (TF_GreaterEqualOp $arg0, $arg1)),
  (TF_LessOp:$dest $arg0, $arg1), [], [(CopyAttrs $src, $dest)]>;

def LogicalNotOfLess : Pat<(
  TF_LogicalNotOp:$src (TF_LessOp $arg0, $arg1)),
  (TF_GreaterEqualOp:$dest $arg0, $arg1), [], [(CopyAttrs $src, $dest)]>;

def LogicalNotOfLessEqual : Pat<
  (TF_LogicalNotOp:$src (TF_LessEqualOp $arg0, $arg1)),
  (TF_GreaterOp:$dest $arg0, $arg1), [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// MatrixSetDiag op patterns.
//===----------------------------------------------------------------------===//

class GetStrAttr<string x>: NativeCodeCall<
  "$_builder.getStringAttr(\"" # x # "\")">;

def MatrixSetDiagToV3 : Pat<
  (TF_MatrixSetDiagOp:$src $input, $diag),
  (TF_MatrixSetDiagV3Op:$dest $input, $diag, (TF_ConstOp (GetI32Attr<0>)),
    (GetStrAttr<"RIGHT_LEFT">)),
  [], [(CopyAttrs $src, $dest)]>;

// MatrixSetDiagToV2 op implicitly used LEFT_LEFT alignment.
def MatrixSetDiagV2ToV3 : Pat<
  (TF_MatrixSetDiagV2Op:$src $input, $diag, $k),
  (TF_MatrixSetDiagV3Op:$dest $input, $diag, $k, (GetStrAttr<"LEFT_LEFT">)),
  [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// MatrixDiag Op.
//===----------------------------------------------------------------------===//

def MatrixDiagToV3 : Pat<
  (TF_MatrixDiagOp:$src $diag),
  (TF_MatrixDiagV3Op:$dest $diag, (TF_ConstOp (GetI32Attr<0>)),
    (TF_ConstOp (GetI32Attr<-1>)), (TF_ConstOp (GetI32Attr<-1>)),
    (TF_ConstOp (GetScalarOfType<0> $diag)), (GetStrAttr<"RIGHT_LEFT">)),
  [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// QuantizeAndDequantizeV2 op patterns.
//===----------------------------------------------------------------------===//

def QuantizeAndDequantizeV2ToQuantizeAndDequantizeV4 : Pat<
  (TF_QuantizeAndDequantizeV2Op:$src $inputs, $min, $max, $signed_input,
    $num_bits, $range_given, $round_mode, $narrow_range, $axis),
  (TF_QuantizeAndDequantizeV4Op:$dest $inputs, $min, $max, $signed_input,
    $num_bits, $range_given, $round_mode, $narrow_range, $axis),
  [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// RealDiv op patterns.
//===----------------------------------------------------------------------===//

def RealDivWithSqrtDivisor : Pat<
  (TF_RealDivOp:$src $arg0, (TF_SqrtOp $arg1)),
    (TF_MulOp:$dest1 $arg0, (TF_RsqrtOp:$dest2 $arg1)),
  [], [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

// Replace division by a constant with a multiplication by a reciprocal of that
// constant. Floating point division can be ~10x more expensive than a
// multiplication.
def RealDivWithConstDivisor : Pat<
  (TF_RealDivOp:$src $arg0, (TF_ConstOp FloatElementsAttr<32>:$value)),
  (TF_MulOp:$dest1 $arg0, (TF_ReciprocalOp:$dest2 (TF_ConstOp $value))),
  [], [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

//===----------------------------------------------------------------------===//
// Reshape op patterns.
//===----------------------------------------------------------------------===//

def RedundantReshape : Pat<
  (TF_ReshapeOp:$src (TF_ReshapeOp $arg, $unused), $shape),
  (TF_ReshapeOp:$dest $arg, $shape), [], [(CopyAttrs $src, $dest)]>;

def ReshapeToSelfShape : Pat<
  (TF_ReshapeOp:$op $x, (TF_ShapeOp $x)), (replaceWithValue $x),
  [(SingleResultAndOperandHaveSameType $x, $op)]>;

//===----------------------------------------------------------------------===//
// Square op patterns.
//===----------------------------------------------------------------------===//

// Restrict the pattern to the types supported by SquaredDifferenceOp.
def TF_NumberTensorNoSmallInt: TensorOf<[TF_Bfloat16, TF_Complex128, TF_Complex64,
    TF_Float16, TF_Float32, TF_Int32, TF_Int64]>;

def SquareOfSub : Pat<
  (TF_SquareOp:$src (TF_SubOp TF_NumberTensorNoSmallInt:$arg0,
    TF_NumberTensorNoSmallInt:$arg1)),
  (TF_SquaredDifferenceOp:$dest $arg0, $arg1), [], [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// Sub op patterns.
//===----------------------------------------------------------------------===//

def SubOfNeg : Pat<
  (TF_SubOp:$src $arg0, (TF_NegOp $arg1)), (TF_AddV2Op:$dest $arg0, $arg1), [],
  [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// TruncateDiv op patterns.
//===----------------------------------------------------------------------===//

def TruncateDivWithSqrtDivisor : Pat<
  (TF_TruncateDivOp:$src $arg0, (TF_SqrtOp $arg1)),
  (TF_MulOp:$dest1 $arg0, (TF_RsqrtOp:$dest2 $arg1)), [],
  [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

//===----------------------------------------------------------------------===//
// Xdivy op patterns.
//===----------------------------------------------------------------------===//

def XdivyWithSqrtDivisor : Pat<
  (TF_XdivyOp:$src $arg0, (TF_SqrtOp $arg1)),
  (TF_MulNoNanOp:$dest1 (TF_RsqrtOp:$dest2 $arg1), $arg0), [],
  [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;


//===----------------------------------------------------------------------===//
// Cast op followed by a ReadVariable op can be folded into the ReadVariable
//===----------------------------------------------------------------------===//

def ReadVariableOfCast : Pat<
  (TF_ReadVariableOp:$src (TF_CastOp:$output $x, BoolAttr:$Truncate)),
  (TF_ReadVariableOp:$dest $x), [(HasOnlyReadVariableOpUsers $output)],
  [(CopyAttrs $src, $dest)]>;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Variable ops to tf.VariableV2 ops
//===----------------------------------------------------------------------===//

def VariableToVariableV2 : Pat<
  (TF_VariableOp:$src $shape, $container, $shard_name),
  (TF_VariableV2Op:$dest $shape, $container, $shard_name),
  [], [(CopyAttrs $src, $dest)]>;


//===----------------------------------------------------------------------===//
// Canonicalize tf.HashTable and its user ops to V2 versions.
//===----------------------------------------------------------------------===//
def HashTableAndInitializeTableToV2 : Pat<
  (TF_InitializeTableFromTextFileOp:$src (TF_HashTableOp $container,
    $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype), $filename,
    $key_index, $value_index, $vocab_size, $delimiter, $offset),
  (TF_InitializeTableFromTextFileV2Op:$dest1 (TF_HashTableV2Op:$dest2 $container,
    $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype),
    $filename, $key_index, $value_index, $vocab_size, $delimiter, $offset),
  [], [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

def HashTableAndLookupTableSizeToV2 : Pat<
  (TF_LookupTableSizeOp:$src (TF_HashTableOp $container, $shared_name,
    $use_node_name_sharing, $key_dtype, $value_dtype)),
  (TF_LookupTableSizeV2Op:$dest1 (TF_HashTableV2Op:$dest2 $container, $shared_name,
    $use_node_name_sharing, $key_dtype, $value_dtype)),
  [], [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

def HashTableAndLookupTableFindToV2 : Pat<
  (TF_LookupTableFindOp:$src (TF_HashTableOp $container, $shared_name,
    $use_node_name_sharing, $key_dtype, $value_dtype), $keys, $default_value),
  (TF_LookupTableFindV2Op:$dest1 (TF_HashTableV2Op:$dest2 $container, $shared_name,
    $use_node_name_sharing, $key_dtype, $value_dtype), $keys, $default_value),
  [], [(CopyAttrs $src, $dest1), (CopyAttrs $src, $dest2)]>;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Maximum of zero to tf.Relu
//===----------------------------------------------------------------------===//

def IsInteger32Pred: CPred<
  "getElementTypeOrSelf($0.getType()).isInteger(32)">;

// Whether the transformation is compatible with the device if given.
// Currently, Relu with int32 is not supported on GPU.
def IsDeviceCompatible: Constraint<
  Neg<And<[IsOnGpuDevicePred, IsInteger32Pred]>>>;

// TODO(b/239736969): Remove the device compatibility constraint to make the
// canonicalization device independent, by adding an int32 GPU kernel for Relu.
def MaximumOfZeroToRelu : Pat<
  (TF_MaximumOp:$maximum_op $x, $y), (TF_ReluOp:$dest $x),
  [(IsConstantValueOf<0> $y), (IsDeviceCompatible $maximum_op)],
  [(CopyAttrs $maximum_op, $dest)]>;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Relu of Minimul six to tf.Relu6
//===----------------------------------------------------------------------===//

def ReluOfMinimum6ToRelu6 : Pat<
  (TF_ReluOp:$src (TF_MinimumOp $x, $y)), (TF_Relu6Op:$dest $x),
  [(IsConstantValueOf<6> $y)], [(CopyAttrs $src, $dest)]>;
