// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}
class NS_Op<string mnemonic, list<OpTrait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def OpA : NS_Op<"one_normal_operand_op", []> {
  let arguments = (ins I32:$input);
}

// CHECK-LABEL: OpA definitions

// CHECK:      void OpA::build
// CHECK-SAME:   Value *input
// CHECK:        tblgen_state->operands.push_back(input);

// CHECK:      void OpA::build
// CHECK-SAME:   ArrayRef<Value *> operands
// CHECK:        assert(operands.size() == 1u && "mismatched number of parameters");
// CHECK:        tblgen_state->addOperands(operands);

// CHECK:      LogicalResult OpA::verify() {
// CHECK:        if (!((this->getOperation()->getOperand(0)->getType().isInteger(32))))
// CHECK-NEXT:     return emitOpError("operand #0 must be 32-bit integer");

def OpB : NS_Op<"one_variadic_operand_op", []> {
  let arguments = (ins Variadic<I32>:$input);
}

// CHECK-LABEL: OpB::build
// CHECK-SAME:    ArrayRef<Value *> input
// CHECK-NOT:     assert
// CHECK:         tblgen_state->addOperands(input);

def OpC : NS_Op<"all_variadic_inputs_op", [SameVariadicOperandSize]> {
  let arguments = (ins Variadic<Tensor>:$input1, Variadic<Tensor>:$input2);
}

// CHECK-LABEL: Operation::operand_range OpC::input1()
// CHECK-NEXT:    unsigned variadicOperandSize = (this->getNumOperands() - 0) / 2;
// CHECK-NEXT:    unsigned offset = 0 + variadicOperandSize * 0;
// CHECK-NEXT:    return {std::next(operand_begin(), offset), std::next(operand_begin(), offset + variadicOperandSize)};

// CHECK-LABEL: Operation::operand_range OpC::input2()
// CHECK-NEXT:    unsigned variadicOperandSize = (this->getNumOperands() - 0) / 2;
// CHECK-NEXT:    unsigned offset = 0 + variadicOperandSize * 1;
// CHECK-NEXT:    return {std::next(operand_begin(), offset), std::next(operand_begin(), offset + variadicOperandSize)};

// CHECK-LABEL: OpC::build
// CHECK-NEXT: tblgen_state->addOperands(input1);
// CHECK-NEXT: tblgen_state->addOperands(input2);

def OpD : NS_Op<"mix_variadic_and_normal_inputs_op", [SameVariadicOperandSize]> {
  let arguments = (ins Variadic<Tensor>:$input1, Tensor:$input2, Variadic<Tensor>:$input3);
}

// CHECK-LABEL: Operation::operand_range OpD::input1()
// CHECK-NEXT:    unsigned variadicOperandSize = (this->getNumOperands() - 1) / 2;
// CHECK-NEXT:    unsigned offset = 0 + variadicOperandSize * 0;
// CHECK-NEXT:    return {std::next(operand_begin(), offset), std::next(operand_begin(), offset + variadicOperandSize)};

// CHECK-LABEL: Value *OpD::input2()
// CHECK-NEXT:    unsigned variadicOperandSize = (this->getNumOperands() - 1) / 2;
// CHECK-NEXT:    unsigned offset = 0 + variadicOperandSize * 1;
// CHECK-NEXT:    return this->getOperand(offset);

// CHECK-LABEL: Operation::operand_range OpD::input3()
// CHECK-NEXT:    unsigned variadicOperandSize = (this->getNumOperands() - 1) / 2;
// CHECK-NEXT:    unsigned offset = 1 + variadicOperandSize * 1;
// CHECK-NEXT:    return {std::next(operand_begin(), offset), std::next(operand_begin(), offset + variadicOperandSize)};

// CHECK-LABEL: OpD::build
// CHECK-NEXT: tblgen_state->addOperands(input1);
// CHECK-NEXT: tblgen_state->operands.push_back(input2);
// CHECK-NEXT: tblgen_state->addOperands(input3);

def OpE : NS_Op<"one_variadic_among_multi_normal_inputs_op", []> {
  let arguments = (ins Tensor:$input1, Tensor:$input2, Variadic<Tensor>:$input3, Tensor:$input4, Tensor:$input5);
}

// CHECK-LABEL: Value *OpE::input1()
// CHECK-NEXT:    return this->getOperation()->getOperand(0);

// CHECK-LABEL: Value *OpE::input2()
// CHECK-NEXT:    return this->getOperation()->getOperand(1);

// CHECK-LABEL: Operation::operand_range OpE::input3()
// CHECK-NEXT:    return {std::next(operand_begin(), 2), std::next(operand_begin(), 2 + this->getNumOperands() - 4)};

// CHECK-LABEL: Value *OpE::input4()
// CHECK-NEXT:    return this->getOperation()->getOperand(this->getNumOperands() - 2);

// CHECK-LABEL: Value *OpE::input5()
// CHECK-NEXT:    return this->getOperation()->getOperand(this->getNumOperands() - 1);
