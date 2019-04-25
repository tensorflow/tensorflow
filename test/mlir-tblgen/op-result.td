// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def OpA : Op<"one_normal_result_op", []> {
  let results = (outs I32:$result);
}

// CHECK-LABEL: Value *OpA::result()
// CHECK-NEXT:    return this->getOperation()->getResult(0)

// CHECK-LABEL: void OpA::build
// CHECK:         ArrayRef<Type> resultTypes, ArrayRef<Value *> operands
// CHECK:         assert(resultTypes.size() == 1u && "mismatched number of return types");
// CHECK-NEXT:    tblgen_state->addTypes(resultTypes);

// CHECK-LABEL: LogicalResult OpA::verify()
// CHECK:         if (!((this->getOperation()->getResult(0)->getType().isInteger(32))))
// CHECK-NEXT:      return emitOpError("result #0 must be 32-bit integer");

def OpB : Op<"same_input_output_type_op", [SameValueType]> {
  let arguments = (ins I32:$x);
  let results = (outs I32:$y);
}

// CHECK-LABEL: OpB definitions
// CHECK: void OpB::build(Builder *, OperationState *tblgen_state, Type y, Value *x)
// CHECK:   tblgen_state->types.push_back(y);
// CHECK: void OpB::build(Builder *, OperationState *tblgen_state, Value *x)
// CHECK:   tblgen_state->addTypes({x->getType()});

def OpC : Op<"three_normal_result_op", []> {
  let results = (outs I32:$x, /*unnamed*/I32, I32:$z);
}

// CHECK-LABEL: OpC definitions
// CHECK:       void OpC::build(Builder *, OperationState *tblgen_state, Type x, Type resultType1, Type z)
// CHECK-NEXT:   tblgen_state->types.push_back(x)
// CHECK-NEXT:   tblgen_state->types.push_back(resultType1)
// CHECK-NEXT:   tblgen_state->types.push_back(z)

def IntegerTypeAttr : TypeAttrBase<"IntegerType", "Integer type attribute">;
def OpD : Op<"type_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, IntegerTypeAttr:$attr, F32Attr:$f32);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: OpD definitions
// CHECK: void OpD::build(Builder *, OperationState *tblgen_state, Value *x, TypeAttr attr, FloatAttr f32)
// CHECK: tblgen_state->addTypes({attr.getValue()});

def OpE : Op<"value_attr_as_result_type", [FirstAttrDerivedResultType]> {
  let arguments = (ins I32:$x, F32Attr:$attr);
  let results = (outs Tensor:$y);
}

// CHECK-LABEL: OpE definitions
// CHECK: void OpE::build(Builder *, OperationState *tblgen_state, Value *x, FloatAttr attr)
// CHECK: tblgen_state->addTypes({attr.getType()});

def OpF : Op<"one_variadic_result_op", []> {
  let results = (outs Variadic<I32>:$x);
}

// CHECK-LABEL: Operation::result_range OpF::x()
// CHECK-NEXT:    return {std::next(result_begin(), 0), std::next(result_begin(), 0 + this->getNumResults() - 0)};

// CHECK-LABEL: void OpF::build
// CHECK-SAME:    ArrayRef<Type> x
// CHECK-NOT:     assert
// CHECK:         tblgen_state->addTypes(x);

def OpG : Op<"one_normal_and_one_variadic_result_op", []> {

  let results = (outs I32:$x, Variadic<I32>:$y);
}

// CHECK-LABEL: OpG definitions

// CHECK:      void OpG::build(Builder *, OperationState *tblgen_state, Type x, ArrayRef<Type> y)
// CHECK-NEXT:   tblgen_state->types.push_back(x);
// CHECK-NEXT:   tblgen_state->addTypes(y);

// CHECK: 			void OpG::build
// CHECK:         ArrayRef<Type> resultTypes
// CHECK:         assert(resultTypes.size() >= 1u && "mismatched number of return types");
// CHECK-NEXT:    tblgen_state->addTypes(resultTypes);


def OpH : Op<"all_variadic_results_op", [SameVariadicResultSize]> {
  let results = (outs Variadic<Tensor>:$output1, Variadic<Tensor>:$output2);
}

// CHECK-LABEL: Operation::result_range OpH::output1()
// CHECK-NEXT:      unsigned variadicResultSize = (this->getNumResults() - 0) / 2;
// CHECK-NEXT:      unsigned offset = 0 + variadicResultSize * 0;
// CHECK-NEXT:      return {std::next(result_begin(), offset), std::next(result_begin(), offset + variadicResultSize)};

// CHECK-LABEL: Operation::result_range OpH::output2()
// CHECK-NEXT:      unsigned variadicResultSize = (this->getNumResults() - 0) / 2;
// CHECK-NEXT:      unsigned offset = 0 + variadicResultSize * 1;
// CHECK-NEXT:      return {std::next(result_begin(), offset), std::next(result_begin(), offset + variadicResultSize)};


// CHECK-LABEL: OpH::build
// CHECK-NEXT:    tblgen_state->addTypes(output1);
// CHECK-NEXT:    tblgen_state->addTypes(output2);

def OpI : Op<"mix_variadic_and_normal_results_op", [SameVariadicResultSize]> {
  let results = (outs Variadic<Tensor>:$output1, Tensor:$output2, Variadic<Tensor>:$output3);
}

// CHECK-LABEL: Operation::result_range OpI::output1()
// CHECK-NEXT:    unsigned variadicResultSize = (this->getNumResults() - 1) / 2;
// CHECK-NEXT:    unsigned offset = 0 + variadicResultSize * 0;
// CHECK-NEXT:    return {std::next(result_begin(), offset), std::next(result_begin(), offset + variadicResultSize)};

// CHECK-LABEL: Value *OpI::output2()
// CHECK-NEXT:    unsigned variadicResultSize = (this->getNumResults() - 1) / 2;
// CHECK-NEXT:    unsigned offset = 0 + variadicResultSize * 1;
// CHECK-NEXT:    return this->getResult(offset);

// CHECK-LABEL: Operation::result_range OpI::output3()
// CHECK-NEXT:    unsigned variadicResultSize = (this->getNumResults() - 1) / 2;
// CHECK-NEXT:    unsigned offset = 1 + variadicResultSize * 1;
// CHECK-NEXT:    return {std::next(result_begin(), offset), std::next(result_begin(), offset + variadicResultSize)};

// CHECK-LABEL: OpI::build
// CHECK-NEXT:    tblgen_state->addTypes(output1);
// CHECK-NEXT:    tblgen_state->types.push_back(output2);
// CHECK-NEXT:    tblgen_state->addTypes(output3);

def OpJ : Op<"one_variadic_among_multi_normal_results_op", []> {
  let results = (outs Tensor:$output1, Tensor:$output2, Variadic<Tensor>:$output3, Tensor:$output4, Tensor:$output5);
}

// CHECK-LABEL: Value *OpJ::output1()
// CHECK-NEXT:    return this->getOperation()->getResult(0);

// CHECK-LABEL: Value *OpJ::output2()
// CHECK-NEXT:    return this->getOperation()->getResult(1);

// CHECK-LABEL: Operation::result_range OpJ::output3()
// CHECK-NEXT:    return {std::next(result_begin(), 2), std::next(result_begin(), 2 + this->getNumResults() - 4)};

// CHECK-LABEL: Value *OpJ::output4()
// CHECK-NEXT:    return this->getOperation()->getResult(this->getNumResults() - 2);

// CHECK-LABEL: Value *OpJ::output5()
// CHECK-NEXT:    return this->getOperation()->getResult(this->getNumResults() - 1);

// Test that if the only operand is variadic, we acess the first value in the
// pack to set result type
// ---
def OpK : Op<"only_input_is_variadic_with_same_value_type_op", [SameValueType]> {
  let arguments = (ins Variadic<Tensor>:$input);
  let results = (outs Tensor:$result);
}

// CHECK-LABEL: OpK::build(Builder *, OperationState *tblgen_state, ArrayRef<Value *> input)
// CHECK: tblgen_state->addTypes({input.front()->getType()});
