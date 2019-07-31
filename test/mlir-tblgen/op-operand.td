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

// CHECK:      OpAOperandAdaptor::OpAOperandAdaptor
// CHECK-NEXT: tblgen_operands = values

// CHECK:      void OpA::build
// CHECK-SAME:   Value *input
// CHECK:        tblgen_state->addOperands(input);

// CHECK:      void OpA::build
// CHECK-SAME:   ArrayRef<Value *> operands
// CHECK:        assert(operands.size() == 1u && "mismatched number of parameters");
// CHECK:        tblgen_state->addOperands(operands);

def OpB : NS_Op<"one_variadic_operand_op", []> {
  let arguments = (ins Variadic<I32>:$input);
}

// CHECK-LABEL: OpB::build
// CHECK-SAME:    ArrayRef<Value *> input
// CHECK-NOT:     assert
// CHECK:         tblgen_state->addOperands(input);

def OpD : NS_Op<"mix_variadic_and_normal_inputs_op", [SameVariadicOperandSize]> {
  let arguments = (ins Variadic<AnyTensor>:$input1, AnyTensor:$input2, Variadic<AnyTensor>:$input3);
}

// CHECK-LABEL: ArrayRef<Value *> OpDOperandAdaptor::input1
// CHECK-NEXT:    return getODSOperands(0);

// CHECK-LABEL: Value *OpDOperandAdaptor::input2
// CHECK-NEXT:    return *getODSOperands(1).begin();

// CHECK-LABEL: ArrayRef<Value *> OpDOperandAdaptor::input3
// CHECK-NEXT:    return getODSOperands(2);

// CHECK-LABEL: Operation::operand_range OpD::input1
// CHECK-NEXT: return getODSOperands(0);

// CHECK-LABEL: Value *OpD::input2
// CHECK-NEXT: return *getODSOperands(1).begin();

// CHECK-LABEL: OpD::build
// CHECK-NEXT: tblgen_state->addOperands(input1);
// CHECK-NEXT: tblgen_state->addOperands(input2);
// CHECK-NEXT: tblgen_state->addOperands(input3);
