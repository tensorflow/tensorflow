//===- VectorOps.td - Vector op definitions ---------------*- tablegen -*-====//
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
//
// Defines MLIR vector operations.
//
//===----------------------------------------------------------------------===//

#ifdef VECTOR_OPS
#else
#define VECTOR_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Vector_Dialect : Dialect {
  let name = "vector";
  let cppNamespace = "vector";
}

// Base class for Vector dialect ops.
class Vector_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Vector_Dialect, mnemonic, traits> {
  // For every vector op, there needs to be a:
  //   * void print(OpAsmPrinter *p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser *parser,
  //                                         OperationState *result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def ExtractElementOp :
  Vector_Op<"extractelement", [NoSideEffect,
     PredOpTrait<"operand and result have same element type",
                 TCresVTEtIsSameAsOpBase<0, 0>>]>,
    Arguments<(ins AnyVector:$vector, I32ArrayAttr:$position)>,
    Results<(outs AnyType)> {
  let summary = "extractelement operation";
  let description = [{
    Takes an n-D vector and a k-D position and extracts the (n-k)-D vector at
    the proper position. Degenerates to an element type in the 0-D case.

    Example:
      %1 = vector.extractelement %0[3]: vector<4x8x16xf32>
      %2 = vector.extractelement %0[3, 3, 3]: vector<4x8x16xf32>
  }];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return vector()->getType().cast<VectorType>();
    }
  }];
}

#endif // VECTOR_OPS
