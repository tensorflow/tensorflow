/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Marks the op as no fallback.
def MarkNoFallback : NativeCodeCall<"SetNoFallbackAttr($_builder, $0)">;

def NoFallbackAttrNotSet : Constraint<CPred<
  "!$0.getDefiningOp()->hasAttr(kNoFallbackAttr)">>;

class FloatValueEquals<string val> : Constraint<CPred<
  "FloatValueEquals($0, " # val # ")">>;

class RankEquals<string rank> : Constraint<CPred<
  "RankEquals($0, " # rank # ")">>;

def IsFusibleWithBias : Constraint<CPred<
  "IsFusibleWithBiasOp($0.getDefiningOp())">>;

// Folds TF IdentityOp with constant input.
def RemoveConstIdentityOp : Pat<
  (TF_IdentityOp (TF_ConstOp $input)),
  (TF_ConstOp $input)>;

// Standardizes the Max and Min ops by moving constant value to rhs. This will
// make it easier to create Relu1 matching patterns.
def SwapMaximumOperands : Pat<
  (TF_MaximumOp (TF_ConstOp:$cst $cst_val), $input),
  (TF_MaximumOp $input, $cst)>;

def SwapMinimumOperands : Pat<
  (TF_MinimumOp (TF_ConstOp:$cst $cst_val), $input),
  (TF_MinimumOp $input, $cst)>;

// Relu1 activation is represented as a couple of Max and Min ops, The following
// patterns recognize and keep them as TF ops so they can be converted to the
// TFLite Relu1 op.
def MatchRelu1Pattern1 : Pat<
  (TF_MinimumOp:$min_op
    (TF_MaximumOp $input, (TF_ConstOp:$cst_negone $NegOne)),
    (TF_ConstOp:$cst_one $One)),
  (MarkNoFallback
    (TF_MinimumOp
      (MarkNoFallback (TF_MaximumOp $input, $cst_negone)),
      $cst_one)),
  [(FloatValueEquals<"-1"> $NegOne), (FloatValueEquals<"1"> $One),
   (NoFallbackAttrNotSet $min_op)]>;

def MatchRelu1Pattern2 : Pat<
  (TF_MaximumOp:$max_op
    (TF_MinimumOp $input, (TF_ConstOp:$cst_one $One)),
    (TF_ConstOp:$cst_negone $NegOne)),
  (MarkNoFallback
    (TF_MaximumOp
      (MarkNoFallback (TF_MinimumOp $input, $cst_one)),
      $cst_negone)),
  [(FloatValueEquals<"-1"> $NegOne), (FloatValueEquals<"1"> $One),
   (NoFallbackAttrNotSet $max_op)]>;

// Keeps Add and Sub ops if the second operand is bias.
def KeepAddV2Op : Pat<
  (TF_AddV2Op:$add_op $input, (TF_ConstOp:$bias_cst $bias)),
  (MarkNoFallback (TF_AddV2Op $input, $bias_cst)),
  [(IsFusibleWithBias $input), (RankEquals<"1"> $bias_cst),
  (NoFallbackAttrNotSet $add_op)]>;

def KeepSubOp : Pat<
  (TF_SubOp:$sub_op $input, (TF_ConstOp:$bias_cst $bias)),
  (MarkNoFallback (TF_SubOp $input, $bias_cst)),
  [(IsFusibleWithBias $input), (RankEquals<"1"> $bias_cst),
   (NoFallbackAttrNotSet $sub_op)]>;
