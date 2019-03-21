// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def MixOperandsAndAttrs : Op<"mix_operands_and_attrs", []> {
  let arguments = (ins F32Attr:$attr, F32:$operand, F32Attr:$otherAttr, F32:$otherArg);
}

// CHECK-LABEL: MixOperandsAndAttrs definitions
// CHECK-DAG: Value *MixOperandsAndAttrs::operand()
// CHECK-DAG: Value *MixOperandsAndAttrs::otherArg()
// CHECK-DAG: void MixOperandsAndAttrs::build(Builder *builder, OperationState *result, FloatAttr attr, Value *operand, FloatAttr otherAttr, Value *otherArg)
// CHECK-DAG: APFloat MixOperandsAndAttrs::attr()
// CHECK-DAG: APFloat MixOperandsAndAttrs::otherAttr()

def OpWithArgs : Op<"op_with_args", []> {
  let arguments = (ins I32:$x, F32Attr:$attr, OptionalAttr<F32Attr>:$optAttr);
}

// CHECK-LABEL: OpWithArgs definitions
// CHECK-DAG: void OpWithArgs::build(Builder *builder, OperationState *result, Value *x, FloatAttr attr, /*optional*/FloatAttr optAttr)
// CHECK-DAG: APFloat OpWithArgs::attr()
// CHECK-DAG: Optional< APFloat > OpWithArgs::optAttr()

