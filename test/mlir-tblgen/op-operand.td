// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OneOperandOp : Op<"one_operand_op", []> {
  let arguments = (ins I32:$input);
}

// CHECK-LABEL: OneOperandOp definitions
// CHECK:      bool OneOperandOp::verify() {
// CHECK:        if (!((this->getInstruction()->getOperand(0)->getType().isInteger(32))))
// CHECK-NEXT:     return emitOpError("operand #0 must be 32-bit integer");
