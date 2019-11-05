//===-- SPIRVBitOps.td - MLIR SPIR-V Bit Ops -*- tablegen -*-===//
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
// This file contains bit ops for the SPIR-V dialect. It corresponds
// to "3.32.13. Bit Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_BIT_OPS
#define SPIRV_BIT_OPS

#ifndef SPIRV_BASE
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

class SPV_BitBinaryOp<string mnemonic, list<OpTrait> traits = []> :
      // All the operands type used in bit instructions are SPV_Integer.
      SPV_BinaryOp<mnemonic, SPV_Integer, SPV_Integer,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

// -----

def SPV_BitwiseOrOp : SPV_BitBinaryOp<"BitwiseOr", [Commutative]> {
  let summary = [{
    Result is 1 if either Operand 1 or Operand 2 is 1. Result is 0 if both
    Operand 1 and Operand 2 are 0.
  }];

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitwise-or-op ::= ssa-id `=` `spv.BitwiseOr` ssa-use, ssa-use
                                  `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.BitwiseOr %0, %1 : i32
    %2 = spv.BitwiseOr %0, %1 : vector<4xi32>
    ```
  }];
}

// -----

def SPV_BitwiseXorOp : SPV_BitBinaryOp<"BitwiseXor", [Commutative]> {
  let summary = [{
    Result is 1 if exactly one of Operand 1 or Operand 2 is 1. Result is 0
    if Operand 1 and Operand 2 have the same value.
  }];

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitwise-xor-op ::= ssa-id `=` `spv.BitwiseXor` ssa-use, ssa-use
                                  `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.BitwiseXor %0, %1 : i32
    %2 = spv.BitwiseXor %0, %1 : vector<4xi32>
    ```
  }];
}

// -----

def SPV_BitwiseAndOp : SPV_BitBinaryOp<"BitwiseAnd", [Commutative]> {
  let summary = [{
    Result is 1 if both Operand 1 and Operand 2 are 1. Result is 0 if either
    Operand 1 or Operand 2 are 0.
  }];

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitwise-and-op ::= ssa-id `=` `spv.BitwiseAnd` ssa-use, ssa-use
                                  `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.BitwiseAnd %0, %1 : i32
    %2 = spv.BitwiseAnd %0, %1 : vector<4xi32>
    ```
  }];
}

#endif // SPIRV_BIT_OPS
