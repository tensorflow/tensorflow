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

//===----------------------------------------------------------------------===//
// Add op patterns.
//===----------------------------------------------------------------------===//

def AddToAddV2 : Pat<(TF_AddOp TF_NumberTensor:$arg0, TF_NumberTensor:$arg1),
                     (TF_AddV2Op $arg0, $arg1)>;

//===----------------------------------------------------------------------===//
// AddV2 op patterns.
//===----------------------------------------------------------------------===//

def AddV2OfNegLeft : Pat<(TF_AddV2Op (TF_NegOp $arg0), $arg1),
                         (TF_SubOp $arg1, $arg0)>;

def AddV2OfNegRight : Pat<(TF_AddV2Op $arg0, (TF_NegOp $arg1)),
                          (TF_SubOp $arg0, $arg1)>;

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
def GatherToV2 :
  Pat<(TF_GatherOp $params, $indices, $ignored_validate_indices),
      (TF_GatherV2Op $params, $indices,
        (TF_ConstOp (GetI32Attr<0>)), (GetI64Attr<0>))>;

//===----------------------------------------------------------------------===//
// BatchMatMul op patterns.
//===----------------------------------------------------------------------===//

// Static shaped operands in a legal BatchMatMul op will have matching batch
// dimensions and can be upgraded to the BatchMatMulV2 op. Canonicalizing
// dynamically shaped operands is not correct as that will execute ops that
// have non matching batch dimensions but are broadcastable which should fail
// with V1.
def BatchMatMulToV2 :
  Pat<(TF_BatchMatMulOp AnyStaticShapeTensor:$x, AnyStaticShapeTensor:$y,
                        $adj_x, $adj_y),
      (TF_BatchMatMulV2Op $x, $y, $adj_x, $adj_y)>;

def BatchMatMulToMatMul : Pat<(TF_BatchMatMulOp $x, $y, $adj_x, $adj_y),
                              (TF_MatMulOp $x, $y, $adj_x, $adj_y),
                              [(IsRank2Tensor $x), (IsRank2Tensor $y)]>;

//===----------------------------------------------------------------------===//
// BatchMatMulV2 op patterns.
//===----------------------------------------------------------------------===//

def BatchMatMulV2ToMatMul : Pat<(TF_BatchMatMulV2Op $x, $y, $adj_x, $adj_y),
                              (TF_MatMulOp $x, $y, $adj_x, $adj_y),
                              [(IsRank2Tensor $x), (IsRank2Tensor $y)]>;

//===----------------------------------------------------------------------===//
// BatchToSpace op patterns.
//===----------------------------------------------------------------------===//

def BatchToSpaceBlockSizeToBlockShape : NativeCodeCall<
  "DenseElementsAttr::get(RankedTensorType::get({2}, $_builder.getI64Type()), "
  "ArrayRef<APInt>{$0.getValue(), $0.getValue()})">;

def BatchToSpaceToBatchToSpaceND :
    Pat<(TF_BatchToSpaceOp $input, $crops, $block_size),
        (TF_BatchToSpaceNDOp $input,
         (TF_ConstOp (BatchToSpaceBlockSizeToBlockShape $block_size)),
         $crops),
        [(IsRank4Tensor $input), (IsRank2Tensor $crops)]>;

//===----------------------------------------------------------------------===//
// BiasAddV1 op patterns.
//===----------------------------------------------------------------------===//

def BiasAddV1ToBiasAdd : Pat<(TF_BiasAddV1Op $arg0, $arg1),
  (TF_BiasAddOp $arg0, $arg1, ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">)>;

//===----------------------------------------------------------------------===//
// Bitcast op patterns.
//===----------------------------------------------------------------------===//

def BitcastSameType : Pat<(TF_BitcastOp:$res $arg), (replaceWithValue $arg),
                          [(SingleResultAndOperandHaveSameElementType $res,
                                                                      $arg)]>;

def BitcastNested : Pat<(TF_BitcastOp (TF_BitcastOp $arg)),
                        (TF_BitcastOp $arg)>;

//===----------------------------------------------------------------------===//
// Concat op patterns.
//===----------------------------------------------------------------------===//

def ConvertToConcatV2 : Pat<(TF_ConcatOp $axis, $inputs),
                            (TF_ConcatV2Op $inputs, $axis)>;

//===----------------------------------------------------------------------===//
// Div op patterns.
//===----------------------------------------------------------------------===//

/// Favor Mul over Div.
def DivWithSqrtDivisor : Pat<(TF_DivOp $arg0, (TF_SqrtOp $arg1)),
                             (TF_MulOp $arg0, (TF_RsqrtOp $arg1))>;

//===----------------------------------------------------------------------===//
// Log op patterns.
//===----------------------------------------------------------------------===//

def LogOfSoftmax : Pat<(TF_LogOp (TF_SoftmaxOp $arg)), (TF_LogSoftmaxOp $arg)>;

// Canonicalize: Log(1.0 + x) to Log1p(x)
//
// We currently do this rewrite only if the constant `1` is a scalar, because
// it is safely broadcastable to any shape. To be able to canonicalize when
// constant values is not a scalar, we have to first prove that it is
// broadcastable to `x`, which requires static shape information.
def LogToLog1p : Pat<
    (TF_LogOp (TF_AddV2Op
                $arg,
                (TF_ConstOp ConstantAttr<RankedF32ElementsAttr<[]>, "1.0f">))),
    (TF_Log1pOp $arg)>;

//===----------------------------------------------------------------------===//
// LogicalNot op patterns.
//===----------------------------------------------------------------------===//

def LogicalNotOfEqual : Pat<
    (TF_LogicalNotOp (TF_EqualOp $arg0, $arg1, $shape_error)),
    (TF_NotEqualOp $arg0, $arg1, $shape_error)>;

def LogicalNotOfNotEqual : Pat<
    (TF_LogicalNotOp (TF_NotEqualOp $arg0, $arg1, $shape_error)),
    (TF_EqualOp $arg0, $arg1, $shape_error)>;

def LogicalNotOfGreater : Pat<(TF_LogicalNotOp (TF_GreaterOp $arg0, $arg1)),
                              (TF_LessEqualOp $arg0, $arg1)>;

def LogicalNotOfGreaterEqual : Pat<(TF_LogicalNotOp (TF_GreaterEqualOp $arg0,
                                                                       $arg1)),
                                   (TF_LessOp $arg0, $arg1)>;

def LogicalNotOfLess : Pat<(TF_LogicalNotOp (TF_LessOp $arg0, $arg1)),
                           (TF_GreaterEqualOp $arg0, $arg1)>;

def LogicalNotOfLessEqual : Pat<(TF_LogicalNotOp (TF_LessEqualOp $arg0, $arg1)),
                                (TF_GreaterOp $arg0, $arg1)>;

//===----------------------------------------------------------------------===//
// MatrixSetDiag op patterns.
//===----------------------------------------------------------------------===//

class GetStrAttr<string x>: NativeCodeCall<
  "$_builder.getStringAttr(\"" # x # "\")">;

def MatrixSetDiagToV3 : Pat<(TF_MatrixSetDiagOp $input, $diag),
                            (TF_MatrixSetDiagV3Op $input, $diag,
                              (TF_ConstOp (GetI32Attr<0>)),
                              (GetStrAttr<"RIGHT_LEFT">))>;

// MatrixSetDiagToV2 op implicitly used LEFT_LEFT alignment.
def MatrixSetDiagV2ToV3 : Pat<(TF_MatrixSetDiagV2Op $input, $diag, $k),
                              (TF_MatrixSetDiagV3Op $input, $diag, $k,
                                (GetStrAttr<"LEFT_LEFT">))>;

//===----------------------------------------------------------------------===//
// MatrixDiag Op.
//===----------------------------------------------------------------------===//

def MatrixDiagToV3 : Pat<(TF_MatrixDiagOp $diag),
                         (TF_MatrixDiagV3Op $diag,
                           (TF_ConstOp (GetI32Attr<0>)),
                           (TF_ConstOp (GetI32Attr<-1>)),
                           (TF_ConstOp (GetI32Attr<-1>)),
                           (TF_ConstOp (GetScalarOfType<0> $diag)),
                           (GetStrAttr<"RIGHT_LEFT">))>;

//===----------------------------------------------------------------------===//
// QuantizeAndDequantizeV2 op patterns.
//===----------------------------------------------------------------------===//

def QuantizeAndDequantizeV2ToQuantizeAndDequantizeV4 : Pat<
  (TF_QuantizeAndDequantizeV2Op $inputs, $min, $max, $signed_input, $num_bits,
    $range_given, $round_mode, $narrow_range, $axis),
  (TF_QuantizeAndDequantizeV4Op $inputs, $min, $max, $signed_input, $num_bits,
    $range_given, $round_mode, $narrow_range, $axis)>;

//===----------------------------------------------------------------------===//
// RealDiv op patterns.
//===----------------------------------------------------------------------===//

def RealDivWithSqrtDivisor : Pat<(TF_RealDivOp $arg0, (TF_SqrtOp $arg1)),
                                 (TF_MulOp $arg0, (TF_RsqrtOp $arg1))>;

// Replace division by a constant with a multiplication by a reciprocal of that
// constant. Floating point division can be ~10x more expensive than a
// multiplication.
def RealDivWithConstDivisor : Pat<
  (TF_RealDivOp $arg0, (TF_ConstOp FloatElementsAttr<32>:$value)),
  (TF_MulOp $arg0, (TF_ReciprocalOp (TF_ConstOp $value)))>;

//===----------------------------------------------------------------------===//
// Reshape op patterns.
//===----------------------------------------------------------------------===//

def RedundantReshape : Pat<(TF_ReshapeOp (TF_ReshapeOp $arg, $unused), $shape),
                           (TF_ReshapeOp $arg, $shape)>;

def ReshapeToSelfShape : Pat<(TF_ReshapeOp:$op $x, (TF_ShapeOp $x)),
                         (replaceWithValue $x),
                         [(SingleResultAndOperandHaveSameType $x, $op)]>;

//===----------------------------------------------------------------------===//
// Square op patterns.
//===----------------------------------------------------------------------===//

def SquareOfSub : Pat<(TF_SquareOp (TF_SubOp $arg0, $arg1)),
                      (TF_SquaredDifferenceOp $arg0, $arg1)>;

//===----------------------------------------------------------------------===//
// Sub op patterns.
//===----------------------------------------------------------------------===//

def SubOfNeg : Pat<(TF_SubOp $arg0, (TF_NegOp $arg1)),
                   (TF_AddV2Op $arg0, $arg1)>;

//===----------------------------------------------------------------------===//
// TruncateDiv op patterns.
//===----------------------------------------------------------------------===//

def TruncateDivWithSqrtDivisor : Pat<(TF_TruncateDivOp $arg0,
                                                       (TF_SqrtOp $arg1)),
                                     (TF_MulOp $arg0, (TF_RsqrtOp $arg1))>;

//===----------------------------------------------------------------------===//
// Xdivy op patterns.
//===----------------------------------------------------------------------===//

def XdivyWithSqrtDivisor : Pat<(TF_XdivyOp $arg0, (TF_SqrtOp $arg1)),
                               (TF_MulNoNanOp (TF_RsqrtOp $arg1), $arg0)>;


//===----------------------------------------------------------------------===//
// Cast op followed by a ReadVariable op can be folded into the ReadVariable
//===----------------------------------------------------------------------===//

def ReadVariableOfCast : Pat<(TF_ReadVariableOp (TF_CastOp:$output $x, BoolAttr:$Truncate)), (TF_ReadVariableOp $x), [(HasOnlyReadVariableOpUsers $output)]>;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Variable ops to tf.VariableV2 ops
//===----------------------------------------------------------------------===//

def VariableToVariableV2 : Pat<(TF_VariableOp $shape, $container, $shard_name), (TF_VariableV2Op $shape, $container, $shard_name)>;


//===----------------------------------------------------------------------===//
// Canonicalize tf.HashTable and its user ops to V2 versions.
//===----------------------------------------------------------------------===//
def HashTableAndInitializeTableToV2 :
 Pat<
   (TF_InitializeTableFromTextFileOp
     (TF_HashTableOp $container, $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype),
     $filename, $key_index, $value_index, $vocab_size, $delimiter, $offset),
   (TF_InitializeTableFromTextFileV2Op
     (TF_HashTableV2Op $container, $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype),
     $filename, $key_index, $value_index, $vocab_size, $delimiter, $offset)
  >;

def HashTableAndLookupTableSizeToV2 :
 Pat<
   (TF_LookupTableSizeOp
     (TF_HashTableOp $container, $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype)),
   (TF_LookupTableSizeV2Op
     (TF_HashTableV2Op $container, $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype))
  >;

def HashTableAndLookupTableFindToV2 :
 Pat<
   (TF_LookupTableFindOp
     (TF_HashTableOp $container, $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype),
     $keys, $default_value),
   (TF_LookupTableFindV2Op
     (TF_HashTableV2Op $container, $shared_name, $use_node_name_sharing, $key_dtype, $value_dtype),
     $keys, $default_value)
 >;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Maximum of zero to tf.Relu
//===----------------------------------------------------------------------===//

def MaximumOfZeroToRelu : Pat<(TF_MaximumOp $x, $y), (TF_ReluOp $x),
                              [(IsConstantValueOf<0> $y)]>;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Relu of Minimul six to tf.Relu6
//===----------------------------------------------------------------------===//

def ReluOfMinimum6ToRelu6 :
  Pat<(TF_ReluOp (TF_MinimumOp $x, $y)),
      (TF_Relu6Op $x),
      [(IsConstantValueOf<6> $y)]>;
