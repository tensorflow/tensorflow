// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OpA : Op<"one_operand_op", []> {
  let arguments = (ins I32:$input);
}

// CHECK-LABEL: OpA definitions

// CHECK:      void OpA::build
// CHECK-SAME:   Value *input
// CHECK:        tblgen_state->addOperands({input});

// CHECK:      void OpA::build
// CHECK-SAME:   ArrayRef<Value *> operands
// CHECK:        assert(operands.size() == 1u && "mismatched number of parameters");
// CHECK:        tblgen_state->addOperands(operands);

// CHECK:      LogicalResult OpA::verify() {
// CHECK:        if (!((this->getOperation()->getOperand(0)->getType().isInteger(32))))
// CHECK-NEXT:     return emitOpError("operand #0 must be 32-bit integer");

def OpB : Op<"variadic_operand_op", []> {
  let arguments = (ins Variadic<I32>:$input);
}

// CHECK-LABEL: OpB::build
// CHECK-SAME:  ArrayRef<Value *> input
// CHECK-NOT:   assert
// CHECK:       tblgen_state->addOperands(input);
