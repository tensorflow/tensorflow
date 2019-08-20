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

    Examples:
    ```
      %1 = vector.extractelement %0[3]: vector<4x8x16xf32>
      %2 = vector.extractelement %0[3, 3, 3]: vector<4x8x16xf32>
    ```
  }];
  let extraClassDeclaration = [{
    VectorType getVectorType() {
      return vector()->getType().cast<VectorType>();
    }
  }];
}
def OuterProductOp :
  Vector_Op<"outerproduct", [NoSideEffect, SameOperandsAndResultElementType]>,
    Arguments<(ins AnyVector:$lhs, AnyVector:$rhs, Variadic<AnyVector>:$acc)>,
    Results<(outs AnyVector)> {
  let summary = "vector outerproduct with optional fused add";
  let description = [{
    Takes 2 1-D vectors and returns the 2-D vector containing the outer product.

    An optional extra 2-D vector argument may be specified in which case the
    operation returns the sum of the outer product and the extra vector. When
    lowered to the LLVMIR dialect, this form emits `llvm.fmuladd`, which can
    lower to actual `fma` instructions in LLVM.

    Examples

      %2 = vector.extractelement %0, %1: vector<4xf32>, vector<8xf32>
      return %2: vector<4x8xf32>

      %3 = vector.extractelement %0, %1, %2:
        vector<4xf32>, vector<8xf32>, vector<4x8xf32>
      return %3: vector<4x8xf32>
  }];
  let extraClassDeclaration = [{
    VectorType getOperandVectorTypeLHS() {
      return lhs()->getType().cast<VectorType>();
    }
    VectorType getOperandVectorTypeRHS() {
      return rhs()->getType().cast<VectorType>();
    }
    VectorType getOperandVectorTypeACC() {
      return (llvm::size(acc()) == 0) ? VectorType() :
        (*acc().begin())->getType().cast<VectorType>();
    }
    VectorType getVectorType() {
      return getResult()->getType().cast<VectorType>();
    }
  }];
}
#endif // VECTOR_OPS
