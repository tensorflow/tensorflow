/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This is the optimization pattern definition file for TensorFlow.js.

include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;

// Constraint that makes sure both operands are the same operands.
// TODO(b/154826385): Reconsider once equal source pattern symbols are allowed.
def EqualOperands : Constraint<CPred<"$0 == $1">>;

// Checks if the operand0's rank is one less than operand1's rank.
def PReluAlphaRankCheck : Constraint<
  CPred<"$0.getType().cast<ShapedType>().getRank() == "
  "$1.getType().cast<ShapedType>().getRank() - 1">>;


// PReLU pattern from Keras:
// f(x) = Relu(x) + (-alpha * Relu(-x))
def : Pat<(TF_AddV2Op
           (TF_ReluOp:$relu_out $input1),
           (TF_MulOp:$mul_out
            (TF_ReluOp (TF_NegOp:$input_neg_out $input2)),
            $neg_alpha)),
          (TFJS_PReluOp $input1, (TF_NegOp $neg_alpha)),
          [(EqualOperands $input1, $input2),
           (PReluAlphaRankCheck $neg_alpha, $input1),
           (HasOneUse $relu_out),
           (HasOneUse $mul_out),
           (HasOneUse $input_neg_out)
          ]>;
