// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OneResultOp : Op<"one_result_op", []> {
  let results = (outs I32:$result);
}

// CHECK-LABEL: OneResultOp definitions
// CHECK:      bool OneResultOp::verify() {
// CHECK:        if (!((this->getInstruction()->getResult(0)->getType().isInteger(32))))
// CHECK-NEXT:     return emitOpError("result #0 must be 32-bit integer");


def SameTypeOp : Op<"same_type_op", [SameValueType]> {
  let arguments = (ins I32:$x);
  let results = (outs I32:$y);
}

// CHECK-LABEL: SameTypeOp definitions
// CHECK: void SameTypeOp::build(Builder *builder, OperationState *result, Type y, Value *x)
// CHECK:   result->addTypes({y});
// CHECK: void SameTypeOp::build(Builder *builder, OperationState *result, Value *x)
// CHECK:   result->addTypes({x->getType()});

def ThreeResultOp : Op<"three_result_op", []> {
  let results = (outs I32:$x, /*unnamed*/I32, I32:$z);
}

// CHECK-LABEL: ThreeResultOp definitions
// CHECK: void ThreeResultOp::build(Builder *builder, OperationState *result, Type x, Type resultType1, Type z)
// CHECK:   result->addTypes({x, resultType1, z});

def IntegerTypeAttr : TypeAttrBase<"IntegerType", "Integer type attribute">;
def TypeAttrResultTypeOp : Op<"type_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, IntegerTypeAttr:$attr, F32Attr:$f32);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: TypeAttrResultTypeOp definitions
// CHECK: void TypeAttrResultTypeOp::build(Builder *builder, OperationState *result, Value *x, TypeAttr attr, FloatAttr f32)
// CHECK: result->addTypes({attr.getValue()});

def ValueAttrResultTypeOp : Op<"value_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, F32Attr:$attr);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: ValueAttrResultTypeOp definitions
// CHECK: void ValueAttrResultTypeOp::build(Builder *builder, OperationState *result, Value *x, FloatAttr attr)
// CHECK: result->addTypes({attr.getType()});

def VariadicResultOp : Op<"variadic_op", []> {
  let results = (outs I32:$x, Variadic<I32>:$y);
}

// CHECK-LABEL: VariadicResultOp definitions
// CHECK: void VariadicResultOp::build(Builder *builder, OperationState *result, Type x, ArrayRef<Type> y)
// CHECK:   result->addTypes({x});
// CHECK:   result->addTypes(y);
