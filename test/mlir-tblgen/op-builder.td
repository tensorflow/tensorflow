// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"


def NS_OpA : Op<"op_same_value_type", [SameValueType]> {
  let arguments = (ins Tensor:$input);
  let results = (outs Tensor:$result);
}

// Test that with SameValueType trait we can generate a builder without
// requiring result type
// ---

// CHECK-LABEL: OpA::build(Builder *, OperationState *tblgen_state, Value *input)
// CHECK: tblgen_state->addTypes({input->getType()});

def NS_OpB : Op<"op_same_value_type_variadic_input", [SameValueType]> {
  let arguments = (ins Variadic<Tensor>:$input);
  let results = (outs Tensor:$result);
}

// Test that if the only operand is variadic, we acess the first value in the
// pack to set result type
// ---

// CHECK-LABEL: OpB::build(Builder *, OperationState *tblgen_state, ArrayRef<Value *> input)
// CHECK: tblgen_state->addTypes({input.front()->getType()});
