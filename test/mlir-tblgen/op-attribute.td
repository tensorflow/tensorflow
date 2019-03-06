// RUN: mlir-tblgen -gen-op-definitions -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def MixOperandsAndAttrs : Op<"mix_operands_and_attrs", []> {
  let arguments = (ins F32Attr:$attr, F32:$operand, F32Attr:$otherAttr, F32:$otherArg);
}

// CHECK-LABEL: class MixOperandsAndAttrs
// CHECK-DAG: Value *operand()
// CHECK-DAG: Value *otherArg()
// CHECK-DAG: void build(Builder *builder, OperationState *result, FloatAttr attr, Value *operand, FloatAttr otherAttr, Value *otherArg)
// CHECK-DAG: APFloat attr()
// CHECK-DAG: APFloat otherAttr()

def OpWithArgs : Op<"op_with_args", []> {
  let arguments = (ins I32:$x, F32Attr:$attr, OptionalAttr<F32Attr>:$optAttr);
}

// CHECK-LABEL: class OpWithArgs
// CHECK: void build(Builder *builder, OperationState *result, Value *x, FloatAttr attr, /*optional*/FloatAttr optAttr)
// CHECK: APFloat attr()
// CHECK: Optional< APFloat > optAttr()

