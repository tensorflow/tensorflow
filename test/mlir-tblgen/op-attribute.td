// RUN: mlir-tblgen -gen-op-definitions -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OpWithArgs : Op<"op_with_args", []> {
  let arguments = (ins I32:$x, F32Attr:$attr, OptionalAttr<F32Attr>:$optAttr);
}

// CHECK-LABEL: OpWithArgs
// CHECK: void build(Builder *builder, OperationState *result, Value *x, FloatAttr attr, /*optional*/FloatAttr optAttr)
// CHECK: APFloat attr()
// CHECK: Optional< APFloat > optAttr()

