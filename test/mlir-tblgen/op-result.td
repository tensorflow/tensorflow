// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OneResultOp : Op<"one_result_op", []> {
  let results = (outs I32:$result);
}

// CHECK-LABEL: OneResultOp definitions
// CHECK:      LogicalResult OneResultOp::verify() {
// CHECK:        if (!((this->getOperation()->getResult(0)->getType().isInteger(32))))
// CHECK-NEXT:     return emitOpError("result #0 must be 32-bit integer");


def SameTypeOp : Op<"same_type_op", [SameValueType]> {
  let arguments = (ins I32:$x);
  let results = (outs I32:$y);
}

// CHECK-LABEL: SameTypeOp definitions
// CHECK: void SameTypeOp::build(Builder *, OperationState *tblgen_state, Type y, Value *x)
// CHECK:   tblgen_state->addTypes({y});
// CHECK: void SameTypeOp::build(Builder *, OperationState *tblgen_state, Value *x)
// CHECK:   tblgen_state->addTypes({x->getType()});

def ThreeResultOp : Op<"three_result_op", []> {
  let results = (outs I32:$x, /*unnamed*/I32, I32:$z);
}

// CHECK-LABEL: ThreeResultOp definitions
// CHECK: void ThreeResultOp::build(Builder *, OperationState *tblgen_state, Type x, Type resultType1, Type z)
// CHECK:   tblgen_state->addTypes({x, resultType1, z});

def IntegerTypeAttr : TypeAttrBase<"IntegerType", "Integer type attribute">;
def TypeAttrResultTypeOp : Op<"type_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, IntegerTypeAttr:$attr, F32Attr:$f32);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: TypeAttrResultTypeOp definitions
// CHECK: void TypeAttrResultTypeOp::build(Builder *, OperationState *tblgen_state, Value *x, TypeAttr attr, FloatAttr f32)
// CHECK: tblgen_state->addTypes({attr.getValue()});

def ValueAttrResultTypeOp : Op<"value_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, F32Attr:$attr);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: ValueAttrResultTypeOp definitions
// CHECK: void ValueAttrResultTypeOp::build(Builder *, OperationState *tblgen_state, Value *x, FloatAttr attr)
// CHECK: tblgen_state->addTypes({attr.getType()});

def VariadicResultOp : Op<"variadic_op", []> {
  let results = (outs I32:$x, Variadic<I32>:$y);
}

// CHECK-LABEL: VariadicResultOp definitions
// CHECK: void VariadicResultOp::build(Builder *, OperationState *tblgen_state, Type x, ArrayRef<Type> y)
// CHECK:   tblgen_state->addTypes({x});
// CHECK:   tblgen_state->addTypes(y);
