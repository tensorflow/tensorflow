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
// Test Traits
//===----------------------------------------------------------------------===//

def SameOperandAndResultElementTypeOp : TEST_Op<"same_operand_and_result_type",
    [SameOperandsAndResultElementType]> {
  let arguments = (ins AnyVectorOrTensor:$x, AnyVectorOrTensor:$y);
  let results = (outs AnyVectorOrTensor:$res);
}

def SameOperandAndResultShapeOp : TEST_Op<"same_operand_and_result_shape",
    [SameValueShape]> {
  let arguments = (ins AnyVectorOrTensor:$x, AnyVectorOrTensor:$y);
  let results = (outs AnyVectorOrTensor:$res);
}

#endif // TEST_OPS
