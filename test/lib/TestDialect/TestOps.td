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

#ifdef TEST_OPS
#else
#define TEST_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def TEST_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "";
}

class TEST_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TEST_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Test 'verifyUnusedValue'
//===----------------------------------------------------------------------===//

def VUVTwoResultOp : TEST_Op<"vuv_two_result_op", []> {
  let arguments = (ins I32:$input);
  let results = (outs I32:$r1, I32:$r2);
}

def VUVFoldTwoResultOp : Pattern<(VUVTwoResultOp $input), [
        (verifyUnusedValue),
        (replaceWithValue $input)
      ]>;

//===----------------------------------------------------------------------===//
// Test Types
//===----------------------------------------------------------------------===//

def AnyVectorOrTensor: AnyTypeOf<[AnyVector, AnyTensor]>;

def TupleOp : TEST_Op<"tuple_32_bit"> {
  let results = (outs TupleOf<[I32, F32]>);
}

def NestedTupleOp : TEST_Op<"nested_tuple_32_bit"> {
  let results = (outs NestedTupleOf<[I32, F32]>);
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

def TypeArrayAttrOp : TEST_Op<"type_array_attr"> {
  let arguments = (ins TypeArrayAttr:$attr);
}
def TypeStringAttrWithTypeOp : TEST_Op<"string_attr_with_type"> {
  let arguments = (ins StrAttr:$attr);
  let printer = [{ *p << getAttr("attr"); }];
  let parser = [{
    Attribute attr;
    Type stringType = OpaqueType::get(Identifier::get("foo", result->context),
                                      "string", result->context);
    return parser->parseAttribute(attr, stringType, "attr", result->attributes);
  }];
}

def StrCaseA: EnumAttrCase<"A">;
def StrCaseB: EnumAttrCase<"B">;

def SomeStrEnum: EnumAttr<
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

//===----------------------------------------------------------------------===//
// Test Traits
//===----------------------------------------------------------------------===//

def SameOperandAndResultElementTypeOp : TEST_Op<"same_operand_and_result_type",
    [SameOperandsAndResultElementType]> {
  let arguments = (ins AnyVectorOrTensor:$x, AnyVectorOrTensor:$y);
  let results = (outs AnyVectorOrTensor:$res);
}

def SameOperandAndResultShapeOp : TEST_Op<"same_operand_and_result_shape",
    [SameOperandsAndResultShape]> {
  let arguments = (ins AnyVectorOrTensor:$x, AnyVectorOrTensor:$y);
  let results = (outs AnyVectorOrTensor:$res);
}

def ArgAndResHaveFixedElementTypesOp :
    TEST_Op<"arg_and_res_have_fixed_element_types",
      [PredOpTrait<"fixed type combination",
         Or<[And<[ElementTypeIsPred<"x", I32>,
                  ElementTypeIsPred<"y", F32>]>,
             ElementTypeIsPred<"attr", I8>]>>,
      ElementTypeIs<"res", I16>]> {
  let arguments = (ins
    AnyVectorOrTensor:$x, AnyVectorOrTensor:$y, AnyAttr:$attr);
  let results = (outs AnyVectorOrTensor:$res);
}

def OperandsHaveSameElementType : TEST_Op<"operands_have_same_element_type", [
    AllElementTypesMatch<["x", "y"]>]> {
  let arguments = (ins AnyTensor:$x, AnyTensor:$y);
}

def OperandOneAndResultHaveSameElementType : TEST_Op<
    "operand_one_and_result_have_same_element_type",
    [AllElementTypesMatch<["x", "res"]>]> {
  let arguments = (ins AnyTensor:$x, AnyTensor:$y);
  let results = (outs AnyTensor:$res);
}

def OperandsHaveSameType :
    TEST_Op<"operands_have_same_type", [AllTypesMatch<["x", "y"]>]> {
  let arguments = (ins AnyTensor:$x, AnyTensor:$y);
}

def OperandOneAndResultHaveSameType :
    TEST_Op<"operand_one_and_result_have_same_type",
            [AllTypesMatch<["x", "res"]>]> {
  let arguments = (ins AnyTensor:$x, AnyTensor:$y);
  let results = (outs AnyTensor:$res);
}

//===----------------------------------------------------------------------===//
// Test Patterns
//===----------------------------------------------------------------------===//

def OpA : TEST_Op<"op_a"> {
  let arguments = (ins I32:$operand, I32Attr:$attr);
  let results = (outs I32:$result);
}

def OpB : TEST_Op<"op_b"> {
  let arguments = (ins I32:$operand, I32Attr:$attr);
  let results = (outs I32:$result);
}

// Test named pattern.
def TestNamedPatternRule : Pat<(OpA $input, $attr), (OpB $input, $attr)>;

// Test with constant attr.
def OpC : TEST_Op<"op_c">, Arguments<(ins I32:$arg)>, Results<(outs I32:$res)>;
def : Pat<(OpC $input), (OpB $input, ConstantAttr<I32Attr, "17">:$attr)>;

// Test with fused location.
def : Pat<(OpA (OpA $input, $attr), $bttr), (OpB $input, $bttr)>;

// Test added benefit.
def OpD : TEST_Op<"op_d">, Arguments<(ins I32:$arg)>, Results<(outs I32:$res)>;
def OpE : TEST_Op<"op_e">, Arguments<(ins I32:$arg)>, Results<(outs I32:$res)>;
def OpF : TEST_Op<"op_f">, Arguments<(ins I32:$arg)>, Results<(outs I32:$res)>;
def OpG : TEST_Op<"op_g">, Arguments<(ins I32:$arg)>, Results<(outs I32:$res)>;
// Verify that bumping benefit results in selecting different op.
def : Pat<(OpD $input), (OpE $input)>;
def : Pat<(OpD $input), (OpF $input), [], (addBenefit 10)>;
// Verify that patterns with more source nodes are selected before those with fewer.
def : Pat<(OpG $input), (OpB $input, ConstantAttr<I32Attr, "20">:$attr)>;
def : Pat<(OpG (OpG $input)), (OpB $input, ConstantAttr<I32Attr, "34">:$attr)>;

// Test string enum attribute in rewrites.
def : Pat<(StrEnumAttrOp StrCaseA), (StrEnumAttrOp StrCaseB)>;
// Test integer enum attribute in rewrites.
def : Pat<(I32EnumAttrOp I32Case5), (I32EnumAttrOp I32Case10)>;
def : Pat<(I64EnumAttrOp I64Case5), (I64EnumAttrOp I64Case10)>;

//===----------------------------------------------------------------------===//
// Test op regions
//===----------------------------------------------------------------------===//

def TwoRegionOp : TEST_Op<"two_region_op", []> {
  let regions = (region AnyRegion, AnyRegion);
}

def SizedRegionOp : TEST_Op<"sized_region_op", []> {
  let regions = (region SizedRegion<2>:$my_region, SizedRegion<1>);
}

//===----------------------------------------------------------------------===//
// Test Legalization
//===----------------------------------------------------------------------===//

def Test_LegalizerEnum_Success : EnumAttrCase<"Success">;
def Test_LegalizerEnum_Failure : EnumAttrCase<"Failure">;

def Test_LegalizerEnum : EnumAttr<"Success", "Failure",
  [Test_LegalizerEnum_Success, Test_LegalizerEnum_Failure]>;

def ILLegalOpA : TEST_Op<"illegal_op_a">, Results<(outs I32:$res)>;
def ILLegalOpB : TEST_Op<"illegal_op_b">, Results<(outs I32:$res)>;
def ILLegalOpC : TEST_Op<"illegal_op_c">, Results<(outs I32:$res)>;
def ILLegalOpD : TEST_Op<"illegal_op_d">, Results<(outs I32:$res)>;
def ILLegalOpE : TEST_Op<"illegal_op_e">, Results<(outs I32:$res)>;
def LegalOpA : TEST_Op<"legal_op_a">,
  Arguments<(ins Test_LegalizerEnum:$status)>, Results<(outs I32:$res)>;

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

//===----------------------------------------------------------------------===//
// Test Type Legalization
//===----------------------------------------------------------------------===//

def TestReturnOp : TEST_Op<"return", [Terminator]>,
  Arguments<(ins Variadic<AnyType>:$inputs)>;
def TestCastOp : TEST_Op<"cast">,
  Arguments<(ins Variadic<AnyType>:$inputs)>, Results<(outs AnyType:$res)>;
def TestInvalidOp : TEST_Op<"invalid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>:$inputs)>;
def TestValidOp : TEST_Op<"valid", [Terminator]>,
  Arguments<(ins Variadic<AnyType>:$inputs)>;

#endif // TEST_OPS
