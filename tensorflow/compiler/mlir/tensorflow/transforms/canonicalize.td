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

def IsRank2Tensor : Type<HasAnyRankOfPred<[2]>, "Rank 2 tensor">;

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
// BatchMatMul op patterns.
//===----------------------------------------------------------------------===//

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
// Bitcast op patterns.
//===----------------------------------------------------------------------===//

def BitcastSameType : Pat<(TF_BitcastOp:$res $arg), (replaceWithValue $arg),
                          [(SingleResultAndOperandHaveSameElementType $res,
                                                                      $arg)]>;

def BitcastNested : Pat<(TF_BitcastOp (TF_BitcastOp $arg)),
                        (TF_BitcastOp $arg)>;

//===----------------------------------------------------------------------===//
// Cast op patterns.
//===----------------------------------------------------------------------===//

def CastSameType : Pat<(TF_CastOp:$res $arg, $truncate),
                       (replaceWithValue $arg),
                       [(SingleResultAndOperandHaveSameElementType $res,
                                                                   $arg)]>;

//===----------------------------------------------------------------------===//
// Conj op patterns.
//===----------------------------------------------------------------------===//

def ConjNested : Pat<(TF_ConjOp (TF_ConjOp $arg)), (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// Div op patterns.
//===----------------------------------------------------------------------===//

/// Favor Mul over Div.
def DivWithSqrtDivisor : Pat<(TF_DivOp $arg0, (TF_SqrtOp $arg1)),
                             (TF_MulOp $arg0, (TF_RsqrtOp $arg1))>;

//===----------------------------------------------------------------------===//
// Invert op patterns.
//===----------------------------------------------------------------------===//

def InvertNested : Pat<(TF_InvertOp (TF_InvertOp $arg)),
                       (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// Log op patterns.
//===----------------------------------------------------------------------===//

def LogOfSoftmax : Pat<(TF_LogOp (TF_SoftmaxOp $arg)), (TF_LogSoftmaxOp $arg)>;

//===----------------------------------------------------------------------===//
// LogicalNot op patterns.
//===----------------------------------------------------------------------===//

def LogicalNotNested : Pat<(TF_LogicalNotOp (TF_LogicalNotOp $arg)),
                           (replaceWithValue $arg)>;

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
// Neg op patterns.
//===----------------------------------------------------------------------===//

def NegNested : Pat<(TF_NegOp (TF_NegOp $arg)), (replaceWithValue $arg)>;

//===----------------------------------------------------------------------===//
// RealDiv op patterns.
//===----------------------------------------------------------------------===//

def RealDivWithSqrtDivisor : Pat<(TF_RealDivOp $arg0, (TF_SqrtOp $arg1)),
                                 (TF_MulOp $arg0, (TF_RsqrtOp $arg1))>;

//===----------------------------------------------------------------------===//
// Reciprocal op patterns.
//===----------------------------------------------------------------------===//

def ReciprocalNested : Pat<(TF_ReciprocalOp (TF_ReciprocalOp $arg)),
                           (replaceWithValue $arg)>;

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
