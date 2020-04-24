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
// xla_chlo ops to canonical xla_hlo ops.
//
// See: https://www.tensorflow.org/xla/operation_semantics

#ifndef CHLO_OPS
#define CHLO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"
include "tensorflow/compiler/mlir/xla/ir/hlo_ops_base.td"

def HLOClient_Dialect : Dialect {
  let name = "xla_chlo";
  let cppNamespace = "xla_chlo";
  let summary = [{
    XLA Client HLO Ops
  }];

  let description = [{
    This dialect contains ops that align closely with the API surface area
    of the XlaBuilder C++ API, where such ops have semantics that go beyond
    what exists in the lower level dialects (such as `xla_hlo`). Essentially,
    whenever the client library uses syntactic sugar or composition
    of multiple ops for an API call, this dialect tries to model the API call
    and provide conversion patterns to fully materialize into lower level
    dialects.
  }];
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
// These correspond to operations in the xla_hlo dialect without the
// "broadcast_" prefix, except that those ops require same-shaped operands and
// results.
//
// See:
//   https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
//   https://www.tensorflow.org/xla/broadcasting
//===----------------------------------------------------------------------===//

class HLOClient_BroadcastBinaryElementwiseOp<
  string mnemonic, list<OpTrait> traits> :
        HLOClient_Op<mnemonic, traits> {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value left, Value  right, "
    "DenseIntElementsAttr broadcast_dimensions"
  >];

  let results = (outs HLO_Tensor);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type(results)
  }];
}

def HLOClient_BroadcastAddOp : HLOClient_BroadcastBinaryElementwiseOp<"broadcast_add",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Addition operator (with optional broadcasting)";

  string description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastAtan2Op : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_atan2",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Atan2 operator (with optional broadcasting)";

  string description = [{
    Returns `atan2(lhs/rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastDivOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_divide",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Division operator (with optional broadcasting)";

  string description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastMaxOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_maximum",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Maximum operator (with optional broadcasting)";

  string description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastMinOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_minimum",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Minimum operator (with optional broadcasting)";

  string description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastMulOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_multiply",
    [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Multiplication operator (with optional broadcasting)";

  string description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastPowOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_power",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Power operator (with optional broadcasting)";

  string description = [{
    Returns `lhs ^ rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastRemOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_remainder",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Remainder operator (with optional broadcasting)";

  string description = [{
    Returns `lhs % rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastShiftLeftOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_shift_left",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift left operator (with optional broadcasting)";

  string description = [{
    Returns `lhs << rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastShiftRightArithmeticOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_shift_right_arithmetic",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift right arithmetic operator (with optional broadcasting)";

  string description = [{
    Returns `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastShiftRightLogicalOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_shift_right_logical",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Shift right logical operator (with optional broadcasting)";

  string description = [{
    Returns `lhs >> rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastSubOp : HLOClient_BroadcastBinaryElementwiseOp<
    "broadcast_subtract",
    [NoSideEffect, SameOperandsAndResultElementType]> {
  string summary = "Subtraction operator (with optional broadcasting)";

  string description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
// The same description as the arithmetic binary elementwise ops applies.
//===----------------------------------------------------------------------===//

class HLOClient_BroadcastBinaryLogicalElementwiseOp<string mnemonic> :
    HLOClient_BroadcastBinaryElementwiseOp<
      mnemonic, [Commutative, NoSideEffect]> {
  let arguments = (ins
    HLO_PredOrIntTensor:$lhs,
    HLO_PredOrIntTensor:$rhs,
    // Explicit rank-broadcast dimension mappings. Defaults to "numpy" prefix
    // padded rank-broadcast semantics if omitted.
    OptionalAttr<BroadcastDimAttr>:$broadcast_dimensions
  );

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    `(` type($lhs) `,` type($rhs) `)` `->` type(results)
  }];
}

def HLOClient_BroadcastAndOp: HLOClient_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_and"> {
  string summary = "Logical and operator (with optional broadcasting)";

  string description = [{
    Returns `logical_and(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastOrOp: HLOClient_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_or"> {
  string summary = "Logical or operator (with optional broadcasting)";

  string description = [{
    Returns `logical_or(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def HLOClient_BroadcastXorOp : HLOClient_BroadcastBinaryLogicalElementwiseOp<
    "broadcast_xor"> {
  string summary = "Logical xor operator (with optional broadcasting)";

  string description = [{
    Returns `logical_xor(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

#endif  // CHLO_OPS
