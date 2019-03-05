// RUN: mlir-tblgen -gen-op-definitions -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def SameTypeOp : Op<"same_type_op", [SameValueType]> {
  let arguments = (ins I32:$x);
  let results = (outs I32:$y);
}

// CHECK-LABEL: class SameTypeOp
// CHECK: void build(Builder *builder, OperationState *result, Type y, Value *x)
// CHECK:   result->addTypes({y});
// CHECK: void build(Builder *builder, OperationState *result, Value *x)
// CHECK:   result->addTypes({x->getType()});

def ThreeResultOp : Op<"three_result_op", []> {
  let results = (outs I32:$x, /*unnamed*/I32, I32:$z);
}

// CHECK-LABEL: class ThreeResultOp
// CHECK: void build(Builder *builder, OperationState *result, Type x, Type resultType1, Type z)
// CHECK:   result->addTypes({x, resultType1, z});

def VariadicResultOp : Op<"variadic_op", []> {
  let results = (outs I32:$x, Variadic<I32>:$y);
}

// CHECK-LABEL: VariadicResultOp
// CHECK: void build(Builder *builder, OperationState *result, Type x, ArrayRef<Type> y)
// CHECK:   result->addTypes({x});
// CHECK:   result->addTypes(y);
