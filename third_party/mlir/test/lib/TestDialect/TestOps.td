//===-- TestOps.td - Test dialect operation definitions ----*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TEST_OPS
#define TEST_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Analysis/CallInterfaces.td"
include "mlir/Analysis/InferTypeOpInterface.td"

def TEST_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "";
}

class TEST_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TEST_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Test Types
//===----------------------------------------------------------------------===//

def ComplexF64 : Complex<F64>;
def ComplexOp : TEST_Op<"complex_f64"> {
  let results = (outs ComplexF64);
}

def ComplexTensorOp : TEST_Op<"complex_f64_tensor"> {
  let results = (outs TensorOf<[ComplexF64]>);
}

def AnyShaped: ShapedContainerType<[AnyType], IsShapedTypePred, "shaped">;

def TupleOp : TEST_Op<"tuple_32_bit"> {
  let results = (outs TupleOf<[I32, F32]>);
}

def NestedTupleOp : TEST_Op<"nested_tuple_32_bit"> {
  let results = (outs NestedTupleOf<[I32, F32]>);
}

def TakesStaticMemRefOp : TEST_Op<"takes_static_memref"> {
  let arguments = (ins AnyStaticShapeMemRef:$x);
}

def RankLessThan2I8F32MemRefOp : TEST_Op<"rank_less_than_2_I8_F32_memref"> {
  let results = (outs MemRefRankOf<[I8, F32], [0, 1]>);
}

def NDTensorOfOp : TEST_Op<"nd_tensor_of"> {
  let arguments = (ins
    0DTensorOf<[F32]>:$arg0,
    1DTensorOf<[F32]>:$arg1,
    2DTensorOf<[I16]>:$arg2,
    3DTensorOf<[I16]>:$arg3,
    4DTensorOf<[I16]>:$arg4
  );
}

def RankedTensorOp : TEST_Op<"ranked_tensor_op"> {
  let arguments = (ins AnyRankedTensor:$input);
}

def MultiTensorRankOf : TEST_Op<"multi_tensor_rank_of"> {
  let arguments = (ins
    TensorRankOf<[I8, I32, F32], [0, 1]>:$arg0
  );
}

//===----------------------------------------------------------------------===//
// Test Operands
//===----------------------------------------------------------------------===//

def SymbolScopeOp : TEST_Op<"symbol_scope",
    [SymbolTable, SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let summary =  "operation which defines a new symbol table";
  let regions = (region SizedRegion<1>:$region);
}

def SymbolTableRegionOp : TEST_Op<"symbol_table_region", [SymbolTable]> {
  let summary =  "operation which defines a new symbol table without a "
                 "restriction on a terminator";
  let regions = (region SizedRegion<1>:$region);
}

//===----------------------------------------------------------------------===//
// Test Operands
//===----------------------------------------------------------------------===//

def MixedNormalVariadicOperandOp : TEST_Op<
    "mixed_normal_variadic_operand", [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<AnyTensor>:$input1,
    AnyTensor:$input2,
    Variadic<AnyTensor>:$input3
  );
}

//===----------------------------------------------------------------------===//
// Test Results
//===----------------------------------------------------------------------===//

def MixedNormalVariadicResults : TEST_Op<
    "mixed_normal_variadic_result", [SameVariadicResultSize]> {
  let results = (outs
    Variadic<AnyTensor>:$output1,
    AnyTensor:$output2,
    Variadic<AnyTensor>:$output3
  );
}

//===----------------------------------------------------------------------===//
// Test Attributes
//===----------------------------------------------------------------------===//

def NonNegIntAttrOp : TEST_Op<"non_negative_int_attr"> {
  let arguments = (ins
      NonNegativeI32Attr:$i32attr,
      NonNegativeI64Attr:$i64attr
  );
}

def PositiveIntAttrOp : TEST_Op<"positive_int_attr"> {
  let arguments = (ins
      PositiveI32Attr:$i32attr,
      PositiveI64Attr:$i64attr
  );
}

def TypeArrayAttrOp : TEST_Op<"type_array_attr"> {
  let arguments = (ins TypeArrayAttr:$attr);
}
def TypeStringAttrWithTypeOp : TEST_Op<"string_attr_with_type"> {
  let arguments = (ins StrAttr:$attr);
  let printer = [{ p << getAttr("attr"); }];
  let parser = [{
    Attribute attr;
    Type stringType = OpaqueType::get(Identifier::get("foo",
                                      result.getContext()), "string",
                                      result.getContext());
    return parser.parseAttribute(attr, stringType, "attr", result.attributes);
  }];
}

def StrCaseA: StrEnumAttrCase<"A">;
def StrCaseB: StrEnumAttrCase<"B">;

def SomeStrEnum: StrEnumAttr<
  "SomeStrEnum", "", [StrCaseA, StrCaseB]>;

def StrEnumAttrOp : TEST_Op<"str_enum_attr"> {
  let arguments = (ins SomeStrEnum:$attr);
  let results = (outs I32:$val);
}

def I32Case5:  I32EnumAttrCase<"case5", 5>;
def I32Case10: I32EnumAttrCase<"case10", 10>;

def SomeI32Enum: I32EnumAttr<
  "SomeI32Enum", "", [I32Case5, I32Case10]>;

def I32EnumAttrOp : TEST_Op<"i32_enum_attr"> {
  let arguments = (ins SomeI32Enum:$attr);
  let results = (outs I32:$val);
}

def I64Case5:  I64EnumAttrCase<"case5", 5>;
def I64Case10: I64EnumAttrCase<"case10", 10>;

def SomeI64Enum: I64EnumAttr<
  "SomeI64Enum", "", [I64Case5, I64Case10]>;

def I64EnumAttrOp : TEST_Op<"i64_enum_attr"> {
  let arguments = (ins SomeI64Enum:$attr);
  let results = (outs I32:$val);
}

def FloatElementsAttrOp : TEST_Op<"float_elements_attr"> {
  let arguments = (ins
      RankedF32ElementsAttr<[2]>:$scalar_f32_attr,
      RankedF64ElementsAttr<[4, 8]>:$tensor_f64_attr
  );
}

// A pattern that updates dense<[3.0, 4.0]> to dense<[5.0, 6.0]>.
// This tests both matching and generating float elements attributes.
def UpdateFloatElementsAttr : Pat<
  (FloatElementsAttrOp
    ConstantAttr<RankedF32ElementsAttr<[2]>, "{3.0f, 4.0f}">:$f32attr,
    $f64attr),
  (FloatElementsAttrOp
    ConstantAttr<RankedF32ElementsAttr<[2]>, "{5.0f, 6.0f}">:$f32attr,
    $f64attr)>;

//===----------------------------------------------------------------------===//
// Test Attribute Constraints
//===----------------------------------------------------------------------===//

def SymbolRefOp : TEST_Op<"symbol_ref_attr"> {
  let arguments = (ins
    Confined<FlatSymbolRefAttr, [ReferToOp<"FuncOp">]>:$symbol
  );
}

//===----------------------------------------------------------------------===//
// Test Regions
//===----------------------------------------------------------------------===//

def OneRegionOp : TEST_Op<"one_region_op", []> {
  let regions = (region AnyRegion);
}

def TwoRegionOp : TEST_Op<"two_region_op", []> {
  let regions = (region AnyRegion, AnyRegion);
}

def SizedRegionOp : TEST_Op<"sized_region_op", []> {
  let regions = (region SizedRegion<2>:$my_region, SizedRegion<1>);
}

//===----------------------------------------------------------------------===//
// Test Call Interfaces
//===----------------------------------------------------------------------===//

def ConversionCallOp : TEST_Op<"conversion_call_op",
    [CallOpInterface]> {
  let arguments = (ins Variadic<AnyType>:$inputs, FlatSymbolRefAttr:$callee);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() { return inputs(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<FlatSymbolRefAttr>("callee");
    }
  }];
}

def FunctionalRegionOp : TEST_Op<"functional_region_op",
    [CallableOpInterface]> {
  let regions = (region AnyRegion:$body);
  let results = (outs FunctionType);

  let extraClassDeclaration = [{
    Region *getCallableRegion(CallInterfaceCallable) { return &body(); }
    void getCallableRegions(SmallVectorImpl<Region *> &callables) {
      callables.push_back(&body());
    }
    ArrayRef<Type> getCallableResults(Region *) {
      return getType().cast<FunctionType>().getResults();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Test Traits
//===----------------------------------------------------------------------===//

def SameOperandElementTypeOp : TEST_Op<"same_operand_element_type",
    [SameOperandsElementType]> {
  let arguments = (ins AnyType, AnyType);
  let results = (outs AnyType);
}

def SameOperandAndResultElementTypeOp : TEST_Op<"same_operand_and_result_element_type",
    [SameOperandsAndResultElementType]> {
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
}

def SameOperandShapeOp : TEST_Op<"same_operand_shape", [SameOperandsShape]> {
  let arguments = (ins Variadic<AnyShaped>);
}

def SameOperandAndResultShapeOp : TEST_Op<"same_operand_and_result_shape",
    [SameOperandsAndResultShape]> {
  let arguments = (ins Variadic<AnyShaped>);
  let results = (outs Variadic<AnyShaped>);
}

def SameOperandAndResultTypeOp : TEST_Op<"same_operand_and_result_type",
    [SameOperandsAndResultType]> {
  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
}

def ArgAndResHaveFixedElementTypesOp :
    TEST_Op<"arg_and_res_have_fixed_element_types",
      [PredOpTrait<"fixed type combination",
         And<[ElementTypeIsPred<"x", I32>,
              ElementTypeIsPred<"y", F32>]>>,
      ElementTypeIs<"res", I16>]> {
  let arguments = (ins
    AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def OperandsHaveSameElementType : TEST_Op<"operands_have_same_element_type", [
    AllElementTypesMatch<["x", "y"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
}

def OperandZeroAndResultHaveSameElementType : TEST_Op<
    "operand0_and_result_have_same_element_type",
    [AllElementTypesMatch<["x", "res"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
  let results = (outs AnyType:$res);
}

def OperandsHaveSameType :
    TEST_Op<"operands_have_same_type", [AllTypesMatch<["x", "y"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
}

def OperandZeroAndResultHaveSameType :
    TEST_Op<"operand0_and_result_have_same_type",
            [AllTypesMatch<["x", "res"]>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
  let results = (outs AnyType:$res);
}

def OperandsHaveSameRank :
    TEST_Op<"operands_have_same_rank", [AllRanksMatch<["x", "y"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
}

def OperandZeroAndResultHaveSameRank :
    TEST_Op<"operand0_and_result_have_same_rank",
            [AllRanksMatch<["x", "res"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def OperandZeroAndResultHaveSameShape :
    TEST_Op<"operand0_and_result_have_same_shape",
            [AllShapesMatch<["x", "res"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def OperandZeroAndResultHaveSameElementCount :
    TEST_Op<"operand0_and_result_have_same_element_count",
            [AllElementCountsMatch<["x", "res"]>]> {
  let arguments = (ins AnyShaped:$x, AnyShaped:$y);
  let results = (outs AnyShaped:$res);
}

def FourEqualsFive :
    TEST_Op<"four_equals_five", [AllMatch<["5", "4"], "4 equals 5">]>;

def OperandRankEqualsResultSize :
    TEST_Op<"operand_rank_equals_result_size",
            [AllMatch<[Rank<"operand">.result, ElementCount<"result">.result],
                      "operand rank equals result size">]> {
  let arguments = (ins AnyShaped:$operand);
  let results = (outs AnyShaped:$result);
}

def IfFirstOperandIsNoneThenSoIsSecond :
    TEST_Op<"if_first_operand_is_none_then_so_is_second", [PredOpTrait<
    "has either both none type operands or first is not none",
     Or<[
        And<[TypeIsPred<"x", NoneType>, TypeIsPred<"y", NoneType>]>,
        Neg<TypeIsPred<"x", NoneType>>]>>]> {
  let arguments = (ins AnyType:$x, AnyType:$y);
}

def BroadcastableOp : TEST_Op<"broadcastable", [Broadcastable]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

// There the "HasParent" trait.
def ParentOp : TEST_Op<"parent">;
def ChildOp : TEST_Op<"child", [HasParent<"ParentOp">]>;


def TerminatorOp : TEST_Op<"finish", [Terminator]>;
def SingleBlockImplicitTerminatorOp : TEST_Op<"SingleBlockImplicitTerminator",
    [SingleBlockImplicitTerminator<"TerminatorOp">]> {
  let regions = (region SizedRegion<1>:$region);
}

def I32ElementsAttrOp : TEST_Op<"i32ElementsAttr"> {
  let arguments = (ins I32ElementsAttr:$attr);
}

def OpWithInferTypeInterfaceOp : TEST_Op<"op_with_infer_type_if", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins AnyTensor, AnyTensor);
  let results = (outs AnyTensor);
}

def IsNotScalar : Constraint<CPred<"$0.getType().getRank() != 0">>;

def UpdateAttr : Pat<(I32ElementsAttrOp $attr),
                     (I32ElementsAttrOp ConstantAttr<I32ElementsAttr, "0">),
                     [(IsNotScalar $attr)]>;

def TestBranchOp : TEST_Op<"br", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
}

def AttrSizedOperandOp : TEST_Op<"attr_sized_operands",
                                 [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d,
    I32ElementsAttr:$operand_segment_sizes
  );
}

def AttrSizedResultOp : TEST_Op<"attr_sized_results",
                                [AttrSizedResultSegments]> {
  let arguments = (ins
    I32ElementsAttr:$result_segment_sizes
  );
  let results = (outs
    Variadic<I32>:$a,
    Variadic<I32>:$b,
    I32:$c,
    Variadic<I32>:$d
  );
}

//===----------------------------------------------------------------------===//
// Test Patterns
//===----------------------------------------------------------------------===//

def OpA : TEST_Op<"op_a"> {
  let arguments = (ins I32, I32Attr:$attr);
  let results = (outs I32);
}

def OpB : TEST_Op<"op_b"> {
  let arguments = (ins I32, I32Attr:$attr);
  let results = (outs I32);
}

// Test named pattern.
def TestNamedPatternRule : Pat<(OpA $input, $attr), (OpB $input, $attr)>;

// Test with fused location.
def : Pat<(OpA (OpA $input, $attr), $bttr), (OpB $input, $bttr)>;

// Test added benefit.
def OpD : TEST_Op<"op_d">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpE : TEST_Op<"op_e">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpF : TEST_Op<"op_f">, Arguments<(ins I32)>, Results<(outs I32)>;
def OpG : TEST_Op<"op_g">, Arguments<(ins I32)>, Results<(outs I32)>;
// Verify that bumping benefit results in selecting different op.
def : Pat<(OpD $input), (OpE $input)>;
def : Pat<(OpD $input), (OpF $input), [], (addBenefit 10)>;
// Verify that patterns with more source nodes are selected before those with fewer.
def : Pat<(OpG $input), (OpB $input, ConstantAttr<I32Attr, "20">:$attr)>;
def : Pat<(OpG (OpG $input)), (OpB $input, ConstantAttr<I32Attr, "34">:$attr)>;

// Test patterns for zero-result op.
def OpH : TEST_Op<"op_h">, Arguments<(ins I32)>, Results<(outs)>;
def OpI : TEST_Op<"op_i">, Arguments<(ins I32)>, Results<(outs)>;
def : Pat<(OpH $input), (OpI $input)>;

// Test patterns for zero-input op.
def OpJ : TEST_Op<"op_j">, Arguments<(ins)>, Results<(outs I32)>;
def OpK : TEST_Op<"op_k">, Arguments<(ins)>, Results<(outs I32)>;
def : Pat<(OpJ), (OpK)>;

// Test `$_` for ignoring op argument match.
def TestIgnoreArgMatchSrcOp : TEST_Op<"ignore_arg_match_src"> {
  let arguments = (ins
    AnyType:$a, AnyType:$b, AnyType:$c,
    AnyAttr:$d, AnyAttr:$e, AnyAttr:$f);
}
def TestIgnoreArgMatchDstOp : TEST_Op<"ignore_arg_match_dst"> {
  let arguments = (ins AnyType:$b, AnyAttr:$f);
}
def : Pat<(TestIgnoreArgMatchSrcOp $_, $b, I32, I64Attr:$_, $_, $f),
          (TestIgnoreArgMatchDstOp $b, $f)>;

def OpInterleavedOperandAttribute1 : TEST_Op<"interleaved_operand_attr1"> {
  let arguments = (ins
    I32:$input1,
    I64Attr:$attr1,
    I32:$input2,
    I64Attr:$attr2
  );
}

def OpInterleavedOperandAttribute2 : TEST_Op<"interleaved_operand_attr2"> {
  let arguments = (ins
    I32:$input1,
    I64Attr:$attr1,
    I32:$input2,
    I64Attr:$attr2
  );
}

def ManyArgsOp : TEST_Op<"many_arguments"> {
  let arguments = (ins
    I32:$input1, I32:$input2, I32:$input3, I32:$input4, I32:$input5,
    I32:$input6, I32:$input7, I32:$input8, I32:$input9,
    I64Attr:$attr1, I64Attr:$attr2, I64Attr:$attr3, I64Attr:$attr4,
    I64Attr:$attr5, I64Attr:$attr6, I64Attr:$attr7, I64Attr:$attr8,
    I64Attr:$attr9
  );
}

// Test that DRR does not blow up when seeing lots of arguments.
def : Pat<(ManyArgsOp
            $input1, $input2, $input3, $input4, $input5,
            $input6, $input7, $input8, $input9,
            ConstantAttr<I64Attr, "42">,
            $attr2, $attr3, $attr4, $attr5, $attr6,
            $attr7, $attr8, $attr9),
          (ManyArgsOp
            $input1, $input2, $input3, $input4, $input5,
            $input6, $input7, $input8, $input9,
            ConstantAttr<I64Attr, "24">,
            $attr2, $attr3, $attr4, $attr5, $attr6,
            $attr7, $attr8, $attr9)>;

// Test that we can capture and reference interleaved operands and attributes.
def : Pat<(OpInterleavedOperandAttribute1 $input1, $attr1, $input2, $attr2),
          (OpInterleavedOperandAttribute2 $input1, $attr1, $input2, $attr2)>;

// Test NativeCodeCall.
def OpNativeCodeCall1 : TEST_Op<"native_code_call1"> {
  let arguments = (ins
    I32:$input1, I32:$input2,
    BoolAttr:$choice,
    I64Attr:$attr1, I64Attr:$attr2
  );
  let results = (outs I32);
}
def OpNativeCodeCall2 : TEST_Op<"native_code_call2"> {
  let arguments = (ins I32:$input, I64ArrayAttr:$attr);
  let results = (outs I32);
}
// Native code call to invoke a C++ function
def CreateOperand: NativeCodeCall<"chooseOperand($0, $1, $2)">;
// Native code call to invoke a C++ expression
def CreateArrayAttr: NativeCodeCall<"$_builder.getArrayAttr({$0, $1})">;
// Test that we can use NativeCodeCall to create operand and attribute.
// This pattern chooses between $input1 and $input2 according to $choice and
// it combines $attr1 and $attr2 into an array attribute.
def : Pat<(OpNativeCodeCall1 $input1, $input2,
                             ConstBoolAttrTrue:$choice, $attr1, $attr2),
          (OpNativeCodeCall2 (CreateOperand $input1, $input2, $choice),
                             (CreateArrayAttr $attr1, $attr2))>;
// Note: the following is just for testing purpose.
// Should use the replaceWithValue directive instead.
def UseOpResult: NativeCodeCall<"$0">;
// Test that we can use NativeCodeCall to create result.
def : Pat<(OpNativeCodeCall1 $input1, $input2,
                             ConstBoolAttrFalse, $attr1, $attr2),
          (UseOpResult $input2)>;

def OpNativeCodeCall3 : TEST_Op<"native_code_call3"> {
  let arguments = (ins I32:$input);
  let results = (outs I32);
}
// Test that NativeCodeCall is not ignored if it is not used to directly
// replace the matched root op.
def : Pattern<(OpNativeCodeCall3 $input),
              [(NativeCodeCall<"createOpI($_builder, $0)"> $input), (OpK)]>;

// Test AllAttrConstraintsOf.
def OpAllAttrConstraint1 : TEST_Op<"all_attr_constraint_of1"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def OpAllAttrConstraint2 : TEST_Op<"all_attr_constraint_of2"> {
  let arguments = (ins I64ArrayAttr:$attr);
  let results = (outs I32);
}
def Constraint0 : AttrConstraint<
    CPred<"$_self.cast<ArrayAttr>().getValue()[0]."
          "cast<IntegerAttr>().getInt() == 0">,
    "[0] == 0">;
def Constraint1 : AttrConstraint<
    CPred<"$_self.cast<ArrayAttr>().getValue()[1]."
          "cast<IntegerAttr>().getInt() == 1">,
    "[1] == 1">;
def : Pat<(OpAllAttrConstraint1
            AllAttrConstraintsOf<[Constraint0, Constraint1]>:$attr),
          (OpAllAttrConstraint2 $attr)>;

// Op for testing RewritePattern removing op with inner ops.
def TestOpWithRegionPattern : TEST_Op<"op_with_region_pattern"> {
  let regions = (region SizedRegion<1>:$region);
  let hasCanonicalizer = 1;
}

// Op for testing trivial removal via folding of op with inner ops and no uses.
def TestOpWithRegionFoldNoSideEffect : TEST_Op<
    "op_with_region_fold_no_side_effect", [NoSideEffect]> {
  let regions = (region SizedRegion<1>:$region);
}

// Op for testing folding of outer op with inner ops.
def TestOpWithRegionFold : TEST_Op<"op_with_region_fold"> {
  let arguments = (ins I32:$operand);
  let results = (outs I32);
  let regions = (region SizedRegion<1>:$region);
  let hasFolder = 1;
}

def TestOpWithVariadicResultsAndFolder: TEST_Op<"op_with_variadic_results_and_folder"> {
  let arguments = (ins Variadic<I32>:$operands);
  let results = (outs Variadic<I32>);
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Test Patterns (Symbol Binding)

// Test symbol binding.
def OpSymbolBindingA : TEST_Op<"symbol_binding_a", []> {
  let arguments = (ins I32:$operand, I64Attr:$attr);
  let results = (outs I32);
}
def OpSymbolBindingB : TEST_Op<"symbol_binding_b", []> {
  let arguments = (ins I32:$operand);
  let results = (outs I32);

  let builders = [
    OpBuilder<
      "Builder *builder, OperationState &state, Value *operand",
      [{
        state.types.assign({builder->getIntegerType(32)});
        state.addOperands({operand});
      }]>
  ];
}
def OpSymbolBindingC : TEST_Op<"symbol_binding_c", []> {
  let arguments = (ins I32:$operand);
  let results = (outs I32);
  let builders = OpSymbolBindingB.builders;
}
def OpSymbolBindingD : TEST_Op<"symbol_binding_d", []> {
  let arguments = (ins I32:$input1, I32:$input2, I64Attr:$attr);
  let results = (outs I32);
}
def HasOneUse: Constraint<CPred<"$0->hasOneUse()">, "has one use">;
def : Pattern<
    // Bind to source pattern op operand/attribute/result
    (OpSymbolBindingA:$res_a $operand, $attr), [
        // Bind to auxiliary op result
        (OpSymbolBindingC:$res_c (OpSymbolBindingB:$res_b $operand)),

        // Use bound symbols in resultant ops
        (OpSymbolBindingD $res_b, $res_c, $attr)],
    // Use bound symbols in additional constraints
    [(HasOneUse $res_a)]>;

def OpSymbolBindingNoResult : TEST_Op<"symbol_binding_no_result", []> {
  let arguments = (ins I32:$operand);
}

// Test that we can bind to an op without results and reference it later.
def : Pat<(OpSymbolBindingNoResult:$op $operand),
          (NativeCodeCall<"handleNoResultOp($_builder, $0)"> $op)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Attributes)

// Test matching against op attributes.
def OpAttrMatch1 : TEST_Op<"match_op_attribute1"> {
  let arguments = (ins
    I32Attr:$required_attr,
    OptionalAttr<I32Attr>:$optional_attr,
    DefaultValuedAttr<I32Attr, "42">:$default_valued_attr,
    I32Attr:$more_attr
  );
  let results = (outs I32);
}
def OpAttrMatch2 : TEST_Op<"match_op_attribute2"> {
  let arguments = OpAttrMatch1.arguments;
  let results = (outs I32);
}
def MoreConstraint : AttrConstraint<
    CPred<"$_self.cast<IntegerAttr>().getInt() == 4">, "more constraint">;
def : Pat<(OpAttrMatch1 $required, $optional, $default_valued,
                        MoreConstraint:$more),
          (OpAttrMatch2 $required, $optional, $default_valued, $more)>;

// Test unit attrs.
def OpAttrMatch3 : TEST_Op<"match_op_attribute3"> {
  let arguments = (ins UnitAttr:$attr);
  let results = (outs I32);
}
def OpAttrMatch4 : TEST_Op<"match_op_attribute4"> {
  let arguments = (ins UnitAttr:$attr1, UnitAttr:$attr2);
  let results = (outs I32);
}
def : Pat<(OpAttrMatch3 $attr), (OpAttrMatch4 ConstUnitAttr, $attr)>;

// Test with constant attr.
def OpC : TEST_Op<"op_c">, Arguments<(ins I32)>, Results<(outs I32)>;
def : Pat<(OpC $input), (OpB $input, ConstantAttr<I32Attr, "17">:$attr)>;

// Test string enum attribute in rewrites.
def : Pat<(StrEnumAttrOp StrCaseA), (StrEnumAttrOp StrCaseB)>;
// Test integer enum attribute in rewrites.
def : Pat<(I32EnumAttrOp I32Case5), (I32EnumAttrOp I32Case10)>;
def : Pat<(I64EnumAttrOp I64Case5), (I64EnumAttrOp I64Case10)>;

//===----------------------------------------------------------------------===//
// Test Patterns (Multi-result Ops)

def MultiResultOpKind1: I64EnumAttrCase<"kind1", 1>;
def MultiResultOpKind2: I64EnumAttrCase<"kind2", 2>;
def MultiResultOpKind3: I64EnumAttrCase<"kind3", 3>;
def MultiResultOpKind4: I64EnumAttrCase<"kind4", 4>;
def MultiResultOpKind5: I64EnumAttrCase<"kind5", 5>;
def MultiResultOpKind6: I64EnumAttrCase<"kind6", 6>;

def MultiResultOpEnum: I64EnumAttr<
  "MultiResultOpEnum", "Multi-result op kinds", [
    MultiResultOpKind1, MultiResultOpKind2, MultiResultOpKind3,
    MultiResultOpKind4, MultiResultOpKind5, MultiResultOpKind6
  ]>;

def ThreeResultOp : TEST_Op<"three_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1, F32:$result2, F32:$result3);
}

def AnotherThreeResultOp : TEST_Op<"another_three_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1, F32:$result2, F32:$result3);
}

def TwoResultOp : TEST_Op<"two_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1, F32:$result2);

  let builders = [
    OpBuilder<
      "Builder *builder, OperationState &state, IntegerAttr kind",
      [{
        auto i32 = builder->getIntegerType(32);
        auto f32 = builder->getF32Type();
        state.types.assign({i32, f32});
        state.addAttribute("kind", kind);
      }]>
  ];
}

def AnotherTwoResultOp : TEST_Op<"another_two_result"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs F32:$result1, F32:$result2);
}

def OneResultOp1 : TEST_Op<"one_result1"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs F32:$result1);
}

def OneResultOp2 : TEST_Op<"one_result2"> {
  let arguments = (ins MultiResultOpEnum:$kind);
  let results = (outs I32:$result1);
}

def OneResultOp3 : TEST_Op<"one_result3"> {
  let arguments = (ins F32);
  let results = (outs I32:$result1);
}

// Test using multi-result op as a whole
def : Pat<(ThreeResultOp MultiResultOpKind1),
          (AnotherThreeResultOp MultiResultOpKind1)>;

// Test using multi-result op as a whole for partial replacement
def : Pattern<(ThreeResultOp MultiResultOpKind2),
              [(TwoResultOp MultiResultOpKind2),
               (OneResultOp1 MultiResultOpKind2)]>;
def : Pattern<(ThreeResultOp MultiResultOpKind3),
              [(OneResultOp2 MultiResultOpKind3),
               (AnotherTwoResultOp MultiResultOpKind3)]>;

// Test using results separately in a multi-result op
def : Pattern<(ThreeResultOp MultiResultOpKind4),
              [(TwoResultOp:$res1__0 MultiResultOpKind4),
               (OneResultOp1 MultiResultOpKind4),
               (TwoResultOp:$res2__1 MultiResultOpKind4)]>;

// Test referencing a single value in the value pack
// This rule only matches TwoResultOp if its second result has no use.
def : Pattern<(TwoResultOp:$res MultiResultOpKind5),
              [(OneResultOp2 MultiResultOpKind5),
               (OneResultOp1 MultiResultOpKind5)],
              [(HasNoUseOf:$res__1)]>;

// Test using auxiliary ops for replacing multi-result op
def : Pattern<
    (ThreeResultOp MultiResultOpKind6), [
        // Auxiliary op generated to help building the final result but not
        // directly used to replace the source op's results.
        (TwoResultOp:$interm MultiResultOpKind6),

        (OneResultOp3 $interm__1),
        (AnotherTwoResultOp MultiResultOpKind6)
    ]>;

//===----------------------------------------------------------------------===//
// Test Patterns (Variadic Ops)

def OneVResOneVOperandOp1 : TEST_Op<"one_variadic_out_one_variadic_in1"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
}
def OneVResOneVOperandOp2 : TEST_Op<"one_variadic_out_one_variadic_in2"> {
  let arguments = (ins Variadic<I32>);
  let results = (outs Variadic<I32>);
}

// Rewrite an op with one variadic operand and one variadic result to
// another similar op.
def : Pat<(OneVResOneVOperandOp1 $inputs), (OneVResOneVOperandOp2 $inputs)>;

def MixedVOperandOp1 : TEST_Op<"mixed_variadic_in1",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    F32:$input2,
    Variadic<I32>:$input3
  );
}

def MixedVOperandOp2 : TEST_Op<"mixed_variadic_in2",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    Variadic<I32>:$input1,
    F32:$input2,
    Variadic<I32>:$input3
  );
}

// Rewrite an op with both variadic operands and normal operands.
def : Pat<(MixedVOperandOp1 $input1, $input2, $input3),
          (MixedVOperandOp2 $input1, $input2, $input3)>;

def MixedVResultOp1 : TEST_Op<"mixed_variadic_out1", [SameVariadicResultSize]> {
  let results = (outs
    Variadic<I32>:$output1,
    F32:$output2,
    Variadic<I32>:$output3
  );
}

def MixedVResultOp2 : TEST_Op<"mixed_variadic_out2", [SameVariadicResultSize]> {
  let results = (outs
    Variadic<I32>:$output1,
    F32:$output2,
    Variadic<I32>:$output3
  );
}

// Rewrite an op with both variadic results and normal results.
// Note that because we are generating the op with a top-level result pattern,
// we are able to deduce the correct result types for the generated op using
// the information from the matched root op.
def : Pat<(MixedVResultOp1), (MixedVResultOp2)>;

def OneI32ResultOp : TEST_Op<"one_i32_out"> {
  let results = (outs I32);
}

def MixedVOperandOp3 : TEST_Op<"mixed_variadic_in3",
                               [SameVariadicOperandSize]> {
  let arguments = (ins
    I32:$input1,
    Variadic<I32>:$input2,
    Variadic<I32>:$input3,
    I32Attr:$count
  );

  let results = (outs I32);
}

def MixedVResultOp3 : TEST_Op<"mixed_variadic_out3",
                               [SameVariadicResultSize]> {
  let arguments = (ins I32Attr:$count);

  let results = (outs
    I32:$output1,
    Variadic<I32>:$output2,
    Variadic<I32>:$output3
  );

  // We will use this op in a nested result pattern, where we cannot deduce the
  // result type. So need to provide a builder not requiring result types.
  let builders = [
    OpBuilder<
      "Builder *builder, OperationState &state, IntegerAttr count",
      [{
        auto i32Type = builder->getIntegerType(32);
        state.addTypes(i32Type); // $output1
        SmallVector<Type, 4> types(count.getInt(), i32Type);
        state.addTypes(types); // $output2
        state.addTypes(types); // $output3
        state.addAttribute("count", count);
      }]>
  ];
}

// Generates an op with variadic results using nested pattern.
def : Pat<(OneI32ResultOp),
          (MixedVOperandOp3
              (MixedVResultOp3:$results__0 ConstantAttr<I32Attr, "2">),
              (replaceWithValue $results__1),
              (replaceWithValue $results__2),
              ConstantAttr<I32Attr, "2">)>;

//===----------------------------------------------------------------------===//
// Test Legalization
//===----------------------------------------------------------------------===//

def Test_LegalizerEnum_Success : StrEnumAttrCase<"Success">;
def Test_LegalizerEnum_Failure : StrEnumAttrCase<"Failure">;

def Test_LegalizerEnum : StrEnumAttr<"Success", "Failure",
  [Test_LegalizerEnum_Success, Test_LegalizerEnum_Failure]>;

def ILLegalOpA : TEST_Op<"illegal_op_a">, Results<(outs I32)>;
def ILLegalOpB : TEST_Op<"illegal_op_b">, Results<(outs I32)>;
def ILLegalOpC : TEST_Op<"illegal_op_c">, Results<(outs I32)>;
def ILLegalOpD : TEST_Op<"illegal_op_d">, Results<(outs I32)>;
def ILLegalOpE : TEST_Op<"illegal_op_e">, Results<(outs I32)>;
def ILLegalOpF : TEST_Op<"illegal_op_f">, Results<(outs I32)>;
def LegalOpA : TEST_Op<"legal_op_a">,
  Arguments<(ins Test_LegalizerEnum:$status)>, Results<(outs I32)>;
def LegalOpB : TEST_Op<"legal_op_b">, Results<(outs I32)>;

// Check that smaller pattern depths are chosen, i.e. prioritize more direct
// mappings.
def : Pat<(ILLegalOpA), (LegalOpA Test_LegalizerEnum_Success)>;

def : Pat<(ILLegalOpA), (ILLegalOpB)>;
def : Pat<(ILLegalOpB), (LegalOpA Test_LegalizerEnum_Failure)>;

// Check that the higher benefit pattern is taken for multiple legalizations
// with the same depth.
def : Pat<(ILLegalOpC), (ILLegalOpD)>;
def : Pat<(ILLegalOpD), (LegalOpA Test_LegalizerEnum_Failure)>;

def : Pat<(ILLegalOpC), (ILLegalOpE), [], (addBenefit 10)>;
def : Pat<(ILLegalOpE), (LegalOpA Test_LegalizerEnum_Success)>;

// Check that patterns use the most up-to-date value when being replaced.
def TestRewriteOp : TEST_Op<"rewrite">,
  Arguments<(ins AnyType)>, Results<(outs AnyType)>;
def : Pat<(TestRewriteOp $input), (replaceWithValue $input)>;

//===----------------------------------------------------------------------===//
// Test Type Legalization
//===----------------------------------------------------------------------===//

def TestRegionBuilderOp : TEST_Op<"region_builder">;
def TestReturnOp : TEST_Op<"return", [Terminator]>,
  Arguments<(ins Variadic<AnyType>)>;
def TestCastOp : TEST_Op<"cast">,
  Arguments<(ins Variadic<AnyType>)>, Results<(outs AnyType)>;
def TestInvalidOp : TEST_Op<"invalid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>)>;
def TestTypeProducerOp : TEST_Op<"type_producer">,
  Results<(outs AnyType)>;
def TestTypeConsumerOp : TEST_Op<"type_consumer">,
  Arguments<(ins AnyType)>;
def TestValidOp : TEST_Op<"valid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>)>;

//===----------------------------------------------------------------------===//
// Test parser.
//===----------------------------------------------------------------------===//

def WrappedKeywordOp : TEST_Op<"wrapped_keyword"> {
  let arguments = (ins StrAttr:$keyword);
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
}

//===----------------------------------------------------------------------===//
// Test region argument list parsing.

def IsolatedRegionOp : TEST_Op<"isolated_region", [IsolatedFromAbove]> {
  let summary =  "isolated region operation";
  let description = [{
    Test op with an isolated region, to test passthrough region arguments. Each
    argument is of index type.
  }];

  let arguments = (ins Index);
  let regions = (region SizedRegion<1>:$region);
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
}

def WrappingRegionOp : TEST_Op<"wrapping_region",
    [SingleBlockImplicitTerminator<"TestReturnOp">]> {
  let summary =  "wrapping region operation";
  let description = [{
    Test op wrapping another op in a region, to test calling
    parseGenericOperation from the custom parser.
  }];

  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(p, *this); }];
}

def PolyForOp : TEST_Op<"polyfor">
{
  let summary =  "polyfor operation";
  let description = [{
    Test op with multiple region arguments, each argument of index type.
  }];

  let regions = (region SizedRegion<1>:$region);
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Test OpAsmInterface.

def AsmInterfaceOp : TEST_Op<"asm_interface_op"> {
  let results = (outs AnyType:$first, Variadic<AnyType>:$middle_results,
                      AnyType);
}

def AsmDialectInterfaceOp : TEST_Op<"asm_dialect_interface_op"> {
  let results = (outs AnyType);
}

#endif // TEST_OPS
