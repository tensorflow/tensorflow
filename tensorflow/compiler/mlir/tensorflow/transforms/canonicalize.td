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

// Either the permutation tensors are shared or they are constants with the
// same static values
def PermutationsAreEqual : Constraint<
  CPred<"$0 == $1 || ($0.getDefiningOp() && $1.getDefiningOp() && "
        "$0.getDefiningOp()->hasAttr(\"value\") &&"
        "$1.getDefiningOp()->hasAttr(\"value\") &&"
        "$0.getDefiningOp()->getAttr(\"value\") == "
        "$1.getDefiningOp()->getAttr(\"value\"))">>;

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

def AddV2TransDistr : Pat<(TF_AddV2Op (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_AddV2Op $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

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
  (TF_BiasAddOp $arg0, $arg1, ConstantAttr<TF_ConvnetDataFormatAttr, "NHWC">)>;

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

def DivTransDistr : Pat<(TF_DivOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_DivOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

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
// LogicalAnd op patterns.
//===----------------------------------------------------------------------===//

def LogicalAndTransDistr : Pat<(TF_LogicalAndOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_LogicalAndOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// LogicalOr op patterns.
//===----------------------------------------------------------------------===//

def LogicalOrTransDistr : Pat<(TF_LogicalOrOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_LogicalOrOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// BitwiseAnd op patterns.
//===----------------------------------------------------------------------===//

def BitwiseAndTransDistr : Pat<(TF_BitwiseAndOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_BitwiseAndOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// BitwiseOr op patterns.
//===----------------------------------------------------------------------===//

def BitwiseOrTransDistr : Pat<(TF_BitwiseOrOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_BitwiseOrOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// BitwiseXor op patterns.
//===----------------------------------------------------------------------===//

def BitwiseXorTransDistr : Pat<(TF_BitwiseXorOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_BitwiseXorOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// LeftShift op patterns.
//===----------------------------------------------------------------------===//

def LeftShiftTransDistr : Pat<(TF_LeftShiftOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_LeftShiftOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// RightShift op patterns.
//===----------------------------------------------------------------------===//

def RightShiftTransDistr : Pat<(TF_RightShiftOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_RightShiftOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Maximum op patterns.
//===----------------------------------------------------------------------===//

def MaximumTransDistr : Pat<(TF_MaximumOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_MaximumOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Minimum op patterns.
//===----------------------------------------------------------------------===//

def MinimumTransDistr : Pat<(TF_MinimumOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_MinimumOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// MulNoNan op patterns.
//===----------------------------------------------------------------------===//

def MulNoNanTransDistr : Pat<(TF_MulNoNanOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_MulNoNanOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;


//===----------------------------------------------------------------------===//
// MatrixSetDiag op patterns.
//===----------------------------------------------------------------------===//

class GetI32Attr<int x>: NativeCodeCall<
  "$_builder.getI32IntegerAttr(" # x # ")">;

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
// Mul op patterns.
//===----------------------------------------------------------------------===//

def MulTransDistr : Pat<(TF_MulOp (TF_TransposeOp $tensor0, $perm0),
                                  (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_MulOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

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

def RealDivTransDistr : Pat<(TF_RealDivOp (TF_TransposeOp $tensor0, $perm0),
                                    (TF_TransposeOp $tensor1, $perm1)),
                        (TF_TransposeOp (TF_RealDivOp $tensor0, $tensor1),
                                        $perm0),
                        [(PermutationsAreEqual $perm0, $perm1)]>;
                        
//===----------------------------------------------------------------------===//
// Reshape op patterns.
//===----------------------------------------------------------------------===//

def RedundantReshape : Pat<(TF_ReshapeOp (TF_ReshapeOp $arg, $unused), $shape),
                           (TF_ReshapeOp $arg, $shape)>;

def ReshapeToSelfShape : Pat<(TF_ReshapeOp:$op $x, (TF_ShapeOp $x)),
                         (replaceWithValue $x),
                         [(SingleResultAndOperandHaveSameType $x, $op)]>;

//===----------------------------------------------------------------------===//
// SelectV2 op patterns.
//===----------------------------------------------------------------------===//

def SelectV2TransDistr : Pat<(TF_SelectV2Op (TF_TransposeOp $cond, $perm0), 
                                            (TF_TransposeOp $tensor0, $perm1),
                                            (TF_TransposeOp $tensor1, $perm2)),
                          (TF_TransposeOp (TF_SelectV2Op $cond, $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1),
                           (PermutationsAreEqual $perm1, $perm2)]>;

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Square op patterns.
//===----------------------------------------------------------------------===//

def SquareOfSub : Pat<(TF_SquareOp (TF_SubOp $arg0, $arg1)),
                      (TF_SquaredDifferenceOp $arg0, $arg1)>;

//===----------------------------------------------------------------------===//
// SquaredDifference op patterns.
//===----------------------------------------------------------------------===//

def SquaredDifferenceTransDistr : Pat<(TF_SquaredDifferenceOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_SquaredDifferenceOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Sub op patterns.
//===----------------------------------------------------------------------===//

def SubOfNeg : Pat<(TF_SubOp $arg0, (TF_NegOp $arg1)),
                   (TF_AddV2Op $arg0, $arg1)>;

def SubTransDistr : Pat<(TF_SubOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_SubOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// TruncateDiv op patterns.
//===----------------------------------------------------------------------===//

def TruncateDivWithSqrtDivisor : Pat<(TF_TruncateDivOp $arg0,
                                                       (TF_SqrtOp $arg1)),
                                     (TF_MulOp $arg0, (TF_RsqrtOp $arg1))>;

def TruncateDivTransDistr : Pat<(TF_TruncateDivOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_TruncateDivOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Xdivy op patterns.
//===----------------------------------------------------------------------===//

def XdivyWithSqrtDivisor : Pat<(TF_XdivyOp $arg0, (TF_SqrtOp $arg1)),
                               (TF_MulNoNanOp (TF_RsqrtOp $arg1), $arg0)>;

def XdivyTransDistr : Pat<(TF_XdivyOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_XdivyOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Cast op followed by a ReadVariable op can be folded into the ReadVariable
//===----------------------------------------------------------------------===//

def ReadVariableOfCast : Pat<(TF_ReadVariableOp (TF_CastOp:$output $x, BoolAttr:$Truncate)), (TF_ReadVariableOp $x), [(HasOnlyReadVariableOpUsers $output)]>;

//===----------------------------------------------------------------------===//
// Canonicalize tf.Variable ops to tf.VariableV2 ops
//===----------------------------------------------------------------------===//

def VariableToVariableV2 : Pat<(TF_VariableOp $shape, $container, $shard_name), (TF_VariableV2Op $shape, $container, $shard_name)>;

//===----------------------------------------------------------------------===//
// Equal op patterns.
//===----------------------------------------------------------------------===//

def EqualTransDistr : Pat<(TF_EqualOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1),
                                      $shape_error),
                          (TF_TransposeOp (TF_EqualOp $tensor0, $tensor1, $shape_error),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// NotEqual op patterns.
//===----------------------------------------------------------------------===//

def NotEqualTransDistr : Pat<(TF_NotEqualOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1),
                                      $shape_error),
                          (TF_TransposeOp (TF_NotEqualOp $tensor0, $tensor1, $shape_error),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;
                          
//===----------------------------------------------------------------------===//
// FloorDiv op patterns.
//===----------------------------------------------------------------------===//

def FloorDivTransDistr : Pat<(TF_FloorDivOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_FloorDivOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// FloorMod op patterns.
//===----------------------------------------------------------------------===//

def FloorModTransDistr : Pat<(TF_FloorModOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_FloorModOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// GreaterEqual op patterns.
//===----------------------------------------------------------------------===//

def GreaterEqualTransDistr : Pat<(TF_GreaterEqualOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_GreaterEqualOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Greater op patterns.
//===----------------------------------------------------------------------===//

def GreaterTransDistr : Pat<(TF_GreaterOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_GreaterOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// LessEqual op patterns.
//===----------------------------------------------------------------------===//

def LessEqualTransDistr : Pat<(TF_LessEqualOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_LessEqualOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Less op patterns.
//===----------------------------------------------------------------------===//

def LessTransDistr : Pat<(TF_LessOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_LessOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;

//===----------------------------------------------------------------------===//
// Mod op patterns.
//===----------------------------------------------------------------------===//

def ModTransDistr : Pat<(TF_ModOp (TF_TransposeOp $tensor0, $perm0),
                                      (TF_TransposeOp $tensor1, $perm1)),
                          (TF_TransposeOp (TF_ModOp $tensor0, $tensor1),
                                          $perm0),
                          [(PermutationsAreEqual $perm0, $perm1)]>;
