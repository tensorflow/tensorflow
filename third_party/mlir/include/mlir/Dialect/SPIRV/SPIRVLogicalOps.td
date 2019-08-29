//===-- SPIRVLogicalOps.td - MLIR SPIR-V Logical Ops -------*- tablegen -*-===//
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
// to "3.32.15. Relational and Logical Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_LOGICAL_OPS
#else
#define SPIRV_LOGICAL_OPS

#ifdef SPIRV_BASE
#else
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

class SPV_LogicalOp<string mnemonic, Type operandsType,
                    list<OpTrait> traits = []> :
      // Result type is SPV_Bool.
      SPV_BinaryOp<mnemonic, SPV_Bool, operandsType,
                   !listconcat(traits,
                               [NoSideEffect, SameTypeOperands,
                                SameOperandsAndResultShape])> {
  let parser = [{ return ::parseBinaryLogicalOp(parser, result); }];
  let printer = [{ return ::printBinaryLogicalOp(getOperation(), p); }];
}

// -----

def SPV_IEqualOp : SPV_LogicalOp<"IEqual", SPV_Integer, [Commutative]> {
  let summary = "Integer comparison for equality.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    iequal-op ::= ssa-id `=` `spv.IEqual` ssa-use, ssa-use
                             `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.IEqual %0, %1 : i32
    %5 = spv.IEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_INotEqualOp : SPV_LogicalOp<"INotEqual", SPV_Integer, [Commutative]> {
  let summary = "Integer comparison for inequality.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    inot-equal-op ::= ssa-id `=` `spv.INotEqual` ssa-use, ssa-use
                                 `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.INotEqual %0, %1 : i32
    %5 = spv.INotEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SGreaterThanOp : SPV_LogicalOp<"SGreaterThan", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is greater than  Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sgreater-than-op ::= ssa-id `=` `spv.SGreaterThan` ssa-use, ssa-use
                                    `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SGreaterThan %0, %1 : i32
    %5 = spv.SGreaterThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SGreaterThanEqualOp : SPV_LogicalOp<"SGreaterThanEqual", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is greater than or equal to
    Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sgreater-than-equal-op ::= ssa-id `=` `spv.SGreaterThanEqual` ssa-use, ssa-use
                                          `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SGreaterThanEqual %0, %1 : i32
    %5 = spv.SGreaterThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SLessThanOp : SPV_LogicalOp<"SLessThan", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is less than Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sless-than-op ::= ssa-id `=` `spv.SLessThan` ssa-use, ssa-use
                                 `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SLessThan %0, %1 : i32
    %5 = spv.SLessThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_SLessThanEqualOp : SPV_LogicalOp<"SLessThanEqual", SPV_Integer, []> {
  let summary = [{
    Signed-integer comparison if Operand 1 is less than or equal to Operand
    2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    sless-than-equal-op ::= ssa-id `=` `spv.SLessThanEqual` ssa-use, ssa-use
                                       `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.SLessThanEqual %0, %1 : i32
    %5 = spv.SLessThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UGreaterThanOp : SPV_LogicalOp<"UGreaterThan", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is greater than  Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    ugreater-than-op ::= ssa-id `=` `spv.UGreaterThan` ssa-use, ssa-use
                                    `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UGreaterhan %0, %1 : i32
    %5 = spv.UGreaterThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UGreaterThanEqualOp
    : SPV_LogicalOp<"UGreaterThanEqual", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is greater than or equal to
    Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    ugreater-than-equal-op ::= ssa-id `=` `spv.UGreaterThanEqual` ssa-use, ssa-use
                                          `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UGreaterThanEqual %0, %1 : i32
    %5 = spv.UGreaterThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_ULessThanOp : SPV_LogicalOp<"ULessThan", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is less than Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    uless-than-op ::= ssa-id `=` `spv.ULessThan` ssa-use, ssa-use
                                 `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.ULessThan %0, %1 : i32
    %5 = spv.ULessThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_ULessThanEqualOp : SPV_LogicalOp<"ULessThanEqual", SPV_Integer, []> {
  let summary = [{
    Unsigned-integer comparison if Operand 1 is less than or equal to
    Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ``` {.ebnf}
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    uless-than-equal-op ::= ssa-id `=` `spv.ULessThanEqual` ssa-use, ssa-use
                                       `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.ULessThanEqual %0, %1 : i32
    %5 = spv.ULessThanEqual %2, %3 : vector<4xi32>

    ```
  }];
}

#endif // SPIRV_LOGICAL_OPS
