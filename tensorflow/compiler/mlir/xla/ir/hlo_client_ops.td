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

// Defines "client" aligned HLO ops.
// These ops are not necessarily orthogonal or optimized for transformation but
// for ease of expression in certain cases deemed important for client
// libraries (i.e. implicit broadcasting, helper ops, etc).
// This dialect is considered to exist in addition to augment the xla_hlo
// dialect for ergonomic needs, not duplicate/replace it.
//
// The typical use of this dialect is for client libraries to be able to emit
// less constrained ops and rely on the conversion framework to lower any
// xla_hlo_client ops to canonical xla_hlo ops.
//
// See: https://www.tensorflow.org/xla/operation_semantics

#ifndef HLO_CLIENT_OPS
#define HLO_CLIENT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops_base.td"

def HLOClient_Dialect : Dialect {
  let name = "xla_hlo_client";
  let cppNamespace = "xla_hlo_client";
}

class HLOClient_Op<string mnemonic, list<OpTrait> traits> :
    Op<HLOClient_Dialect, mnemonic, traits> {
  // TODO(b/129012527) Much of this custom verification should be expressed as
  // type constraints.
  let verifier = [{ return Verify(*this); }];
}

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
// From the client perspective, each of these support both explicit rank
// broadcasting (via the broadcast_dimensions attribute) and implicit degenerate
// shape broadcasting.
//
// These have 1:1 correspondance with same-named ops in the xla_hlo dialect;
// however, those operations do not support broadcasting.
//
// See:
//   https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
//   https://www.tensorflow.org/xla/broadcasting
//===----------------------------------------------------------------------===//

class HLOClient_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        HLOClient_Op<mnemonic, traits> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value left, Value  right, "
    "DenseIntElementsAttr broadcast_dimensions"
  >];

  let results = (outs HLO_Tensor);
  let parser = [{ return mlir::impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ return mlir::impl::printOneResultOp(getOperation(), p); }];
}

def HLOClient_AddOp : HLOClient_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_AddOp;

def HLOClient_Atan2Op : HLOClient_BinaryElementwiseOp<"atan2",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_Atan2Op;

def HLOClient_DivOp : HLOClient_BinaryElementwiseOp<"divide",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_DivOp;

def HLOClient_MaxOp : HLOClient_BinaryElementwiseOp<"maximum",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MaxOp;

def HLOClient_MinOp : HLOClient_BinaryElementwiseOp<"minimum",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MinOp;

def HLOClient_MulOp : HLOClient_BinaryElementwiseOp<"multiply",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MulOp;

def HLOClient_PowOp : HLOClient_BinaryElementwiseOp<"pow",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_PowOp;

def HLOClient_RemOp : HLOClient_BinaryElementwiseOp<"remainder",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_RemOp;

def HLOClient_ShiftLeftOp : HLOClient_BinaryElementwiseOp<"shift_left",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ShiftLeftOp;

def HLOClient_ShiftRightArithmeticOp : HLOClient_BinaryElementwiseOp<"shift_right_arithmetic",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ShiftRightArithmeticOp;

def HLOClient_ShiftRightLogicalOp : HLOClient_BinaryElementwiseOp<"shift_right_logical",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ShiftRightLogicalOp;

def HLOClient_SubOp : HLOClient_BinaryElementwiseOp<"subtract",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_SubOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
// The same description as the arithmetic binary elementwise ops applies.
//===----------------------------------------------------------------------===//

class HLOClient_BinaryLogicalElementwiseOp<string mnemonic> :
        HLOClient_BinaryElementwiseOp<mnemonic, [Commutative, NoSideEffect]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs,
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );
}

def HLOClient_AndOp: HLOClient_BinaryLogicalElementwiseOp<"and">, BASE_HLO_AndOp;
def HLOClient_OrOp: HLOClient_BinaryLogicalElementwiseOp<"or">, BASE_HLO_OrOp;
def HLOClient_XorOp : HLOClient_BinaryLogicalElementwiseOp<"xor">, BASE_HLO_XorOp;

#endif  // HLO_CLIENT_OPS
