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

class SPV_BitUnaryOp<string mnemonic, list<OpTrait> traits = []> :
      SPV_UnaryOp<mnemonic, SPV_Integer, SPV_Integer,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

class SPV_ShiftOp<string mnemonic, list<OpTrait> traits = []> :
      SPV_BinaryOp<mnemonic, SPV_Integer, SPV_Integer,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultShape])> {
  let parser = [{ return ::parseShiftOp(parser, result); }];
  let printer = [{ ::printShiftOp(this->getOperation(), p); }];
  let verifier = [{ return ::verifyShiftOp(this->getOperation()); }];
}

// -----

def SPV_BitCountOp : SPV_BitUnaryOp<"BitCount", []> {
  let summary = "Count the number of set bits in an object.";

  let description = [{
     Results are computed per component.

    Result Type must be a scalar or vector of integer type.  The components
    must be wide enough to hold the unsigned Width of Base as an unsigned
    value. That is, no sign bit is needed or counted when checking for a
    wide enough result width.

    Base must be a scalar or vector of integer type.  It must have the same
    number of components as Result Type.

    The result is the unsigned value that is the number of bits in Base that
    are 1.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitcount-op ::= ssa-id `=` `spv.BitCount` ssa-use
                               `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.BitCount %0: i32
    %3 = spv.BitCount %1: vector<4xi32>
    ```
  }];
}

// -----

def SPV_BitReverseOp : SPV_BitUnaryOp<"BitReverse", []> {
  let summary = "Reverse the bits in an object.";

  let description = [{
     Results are computed per component.

    Result Type must be a scalar or vector of integer type.

     The type of Base must be the same as Result Type.

    The bit-number n of the result will be taken from bit-number Width - 1 -
    n of Base, where Width is the OpTypeInt operand of the Result Type.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    bitreverse-op ::= ssa-id `=` `spv.BitReverse` ssa-use
                                 `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.BitReverse %0 : i32
    %3 = spv.BitReverse %1 : vector<4xi32>
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

def SPV_ShiftLeftLogicalOp : SPV_ShiftOp<"ShiftLeftLogical", []> {
  let summary = [{
    Shift the bits in Base left by the number of bits specified in Shift.
    The least-significant bits will be zero filled.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of each Base and Shift must be a scalar or vector of integer
    type. Base and Shift must have the same number of components.  The
    number of components and bit width of the type of Base must be the same
    as in Result Type.

    Shift is treated as unsigned. The result is undefined if Shift is
    greater than or equal to the bit width of the components of Base.

    The number of components and bit width of Result Type must match those
    Base type. All types must be integer types.

     Results are computed per component.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    shift-left-logical-op ::= ssa-id `=` `spv.ShiftLeftLogical`
                                          ssa-use `,` ssa-use `:`
                                          integer-scalar-vector-type `,`
                                          integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.ShiftLeftLogical %0, %1 : i32, i16
    %5 = spv.ShiftLeftLogical %3, %4 : vector<3xi32>, vector<3xi16>
    ```
  }];
}

// -----

def SPV_ShiftRightArithmeticOp : SPV_ShiftOp<"ShiftRightArithmetic", []> {
  let summary = [{
    Shift the bits in Base right by the number of bits specified in Shift.
    The most-significant bits will be filled with the sign bit from Base.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of each Base and Shift must be a scalar or vector of integer
    type. Base and Shift must have the same number of components.  The
    number of components and bit width of the type of Base must be the same
    as in Result Type.

    Shift is treated as unsigned. The result is undefined if Shift is
    greater than or equal to the bit width of the components of Base.

     Results are computed per component.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    shift-right-arithmetic-op ::= ssa-id `=` `spv.ShiftRightArithmetic`
                                              ssa-use `,` ssa-use `:`
                                              integer-scalar-vector-type `,`
                                              integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.ShiftRightArithmetic %0, %1 : i32, i16
    %5 = spv.ShiftRightArithmetic %3, %4 : vector<3xi32>, vector<3xi16>
    ```
  }];
}

// -----

def SPV_ShiftRightLogicalOp : SPV_ShiftOp<"ShiftRightLogical", []> {
  let summary = [{
    Shift the bits in Base right by the number of bits specified in Shift.
    The most-significant bits will be zero filled.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of each Base and Shift must be a scalar or vector of integer
    type. Base and Shift must have the same number of components.  The
    number of components and bit width of the type of Base must be the same
    as in Result Type.

    Shift is consumed as an unsigned integer. The result is undefined if
    Shift is greater than or equal to the bit width of the components of
    Base.

     Results are computed per component.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    shift-right-logical-op ::= ssa-id `=` `spv.ShiftRightLogical`
                                           ssa-use `,` ssa-use `:`
                                           integer-scalar-vector-type `,`
                                           integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.ShiftRightLogical %0, %1 : i32, i16
    %5 = spv.ShiftRightLogical %3, %4 : vector<3xi32>, vector<3xi16>
    ```
  }];
}

// -----

def SPV_NotOp : SPV_BitUnaryOp<"Not", []> {
  let summary = "Complement the bits of Operand.";

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.

    Operand’s type  must be a scalar or vector of integer type.  It must
    have the same number of components as Result Type.  The component width
    must equal the component width in Result Type.

    ### Custom assembly form

    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    not-op ::= ssa-id `=` `spv.BitNot` ssa-use `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %2 = spv.Not %0 : i32
    %3 = spv.Not %1 : vector<4xi32>
    ```
  }];
}

#endif // SPIRV_BIT_OPS
