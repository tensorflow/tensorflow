//===-- SPIRVArithmeticOps.td - MLIR SPIR-V Arithmetic Ops -*- tablegen -*-===//
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
// This file contains arithmetic ops for the SPIR-V dialect. It corresponds
// to "3.32.13. Arithmetic Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_ARITHMETIC_OPS
#define SPIRV_ARITHMETIC_OPS

#ifndef SPIRV_BASE
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

class SPV_ArithmeticBinaryOp<string mnemonic, Type type,
                       list<OpTrait> traits = []> :
      // Operands type same as result type.
      SPV_BinaryOp<mnemonic, type, type,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

class SPV_ArithmeticUnaryOp<string mnemonic, Type type,
                            list<OpTrait> traits = []> :
      // Operand type same as result type.
      SPV_UnaryOp<mnemonic, type, type,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

// -----

def SPV_FAddOp : SPV_ArithmeticBinaryOp<"FAdd", SPV_Float, [Commutative]> {
  let summary = "Floating-point addition of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fadd-op ::= ssa-id `=` `spv.FAdd` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.FAdd %0, %1 : f32
    %5 = spv.FAdd %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FDivOp : SPV_ArithmeticBinaryOp<"FDiv", SPV_Float, []> {
  let summary = "Floating-point division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fdiv-op ::= ssa-id `=` `spv.FDiv` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FDiv %0, %1 : f32
    %5 = spv.FDiv %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FModOp : SPV_ArithmeticBinaryOp<"FMod", SPV_Float, []> {
  let summary = [{
    The floating-point remainder whose sign matches the sign of Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 2.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmod-op ::= ssa-id `=` `spv.FMod` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.FMod %0, %1 : f32
    %5 = spv.FMod %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FMulOp : SPV_ArithmeticBinaryOp<"FMul", SPV_Float, [Commutative]> {
  let summary = "Floating-point multiplication of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    ### Custom assembly form

    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmul-op ::= `spv.FMul` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FMul %0, %1 : f32
    %5 = spv.FMul %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FNegateOp : SPV_ArithmeticUnaryOp<"FNegate", SPV_Float, []> {
  let summary = "Floating-point subtract of Operand from zero.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The type of Operand must be the same as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fmul-op ::= `spv.FNegate` ssa-use `:` float-scalar-vector-type
    ```

    For example:

    ```
    %1 = spv.FNegate %0 : f32
    %3 = spv.FNegate %2 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FRemOp : SPV_ArithmeticBinaryOp<"FRem", SPV_Float, []> {
  let summary = [{
    The floating-point remainder whose sign matches the sign of Operand 1.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 1.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    frem-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FRemOp %0, %1 : f32
    %5 = spv.FRemOp %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FSubOp : SPV_ArithmeticBinaryOp<"FSub", SPV_Float, []> {
  let summary = "Floating-point subtraction of Operand 2 from Operand 1.";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fsub-op ::= ssa-id `=` `spv.FRemOp` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.FRemOp %0, %1 : f32
    %5 = spv.FRemOp %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_IAddOp : SPV_ArithmeticBinaryOp<"IAdd", SPV_Integer, [Commutative]> {
  let summary = "Integer addition of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    iadd-op ::= ssa-id `=` `spv.IAdd` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.IAdd %0, %1 : i32
    %5 = spv.IAdd %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_IMulOp : SPV_ArithmeticBinaryOp<"IMul", SPV_Integer, [Commutative]> {
  let summary = "Integer multiplication of Operand 1 and Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    imul-op ::= ssa-id `=` `spv.IMul` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.IMul %0, %1 : i32
    %5 = spv.IMul %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_ISubOp : SPV_ArithmeticBinaryOp<"ISub", SPV_Integer, []> {
  let summary = "Integer subtraction of Operand 2 from Operand 1.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

    The resulting value will equal the low-order N bits of the correct
    result R, where N is the component width and R is computed with enough
    precision to avoid overflow and underflow.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    isub-op ::= `spv.ISub` ssa-use, ssa-use
                          `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.ISub %0, %1 : i32
    %5 = spv.ISub %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SDivOp : SPV_ArithmeticBinaryOp<"SDiv", SPV_Integer, []> {
  let summary = "Signed-integer division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sdiv-op ::= ssa-id `=` `spv.SDiv` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %4 = spv.SDiv %0, %1 : i32
    %5 = spv.SDiv %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SModOp : SPV_ArithmeticBinaryOp<"SMod", SPV_Integer, []> {
  let summary = [{
    Signed remainder operation for the remainder whose sign matches the sign
    of Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 2.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    smod-op ::= ssa-id `=` `spv.SMod` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SMod %0, %1 : i32
    %5 = spv.SMod %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SRemOp : SPV_ArithmeticBinaryOp<"SRem", SPV_Integer, []> {
  let summary = [{
    Signed remainder operation for the remainder whose sign matches the sign
    of Operand 1.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same number of components as Result
    Type. They must have the same component width as Result Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.  Otherwise, the result is the remainder r of Operand
    1 divided by Operand 2 where if r ≠ 0, the sign of r is the same as the
    sign of Operand 1.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    srem-op ::= ssa-id `=` `spv.SRem` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SRem %0, %1 : i32
    %5 = spv.SRem %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UDivOp : SPV_ArithmeticBinaryOp<"UDiv", SPV_Integer, []> {
  let summary = "Unsigned-integer division of Operand 1 divided by Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    udiv-op ::= ssa-id `=` `spv.UDiv` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UDiv %0, %1 : i32
    %5 = spv.UDiv %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UModOp : SPV_ArithmeticBinaryOp<"UMod", SPV_Integer> {
  let summary = "Unsigned modulo operation of Operand 1 modulo Operand 2.";

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

     The types of Operand 1 and Operand 2 both must be the same as Result
    Type.

     Results are computed per component.  The resulting value is undefined
    if Operand 2 is 0.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    umod-op ::= ssa-id `=` `spv.UMod` ssa-use, ssa-use
                           `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UMod %0, %1 : i32
    %5 = spv.UMod %2, %3 : vector<4xi32>

    ```
  }];
}

#endif // SPIRV_ARITHMETIC_OPS
