//===-- SPIRVLogicalOps.td - MLIR SPIR-V Logical Ops -------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains arithmetic ops for the SPIR-V dialect. It corresponds
// to "3.32.15. Relational and Logical Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LOGICAL_OPS
#define SPIRV_LOGICAL_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"

class SPV_LogicalBinaryOp<string mnemonic, Type operandsType,
                    list<OpTrait> traits = []> :
      // Result type is SPV_Bool.
      SPV_BinaryOp<mnemonic, SPV_Bool, operandsType,
                   !listconcat(traits,
                               [NoSideEffect, SameTypeOperands,
                                SameOperandsAndResultShape])> {
  let parser = [{ return ::parseLogicalBinaryOp(parser, result); }];
  let printer = [{ return ::printLogicalOp(getOperation(), p); }];
}

class SPV_LogicalUnaryOp<string mnemonic, Type operandType,
                         list<OpTrait> traits = []> :
      // Result type is SPV_Bool.
      SPV_UnaryOp<mnemonic, SPV_Bool, operandType,
                  !listconcat(traits, [NoSideEffect, SameTypeOperands,
                                       SameOperandsAndResultShape])> {
  let parser = [{ return ::parseLogicalUnaryOp(parser, result); }];
  let printer = [{ return ::printLogicalOp(getOperation(), p); }];
}

// -----

def SPV_FOrdEqualOp : SPV_LogicalBinaryOp<"FOrdEqual", SPV_Float, [Commutative]> {
  let summary = "Floating-point comparison for being ordered and equal.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fordequal-op ::= ssa-id `=` `spv.FOrdEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FOrdEqual %0, %1 : f32
    %5 = spv.FOrdEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FOrdGreaterThanOp : SPV_LogicalBinaryOp<"FOrdGreaterThan", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are ordered and Operand 1 is
    greater than  Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fordgt-op ::= ssa-id `=` `spv.FOrdGreaterThan` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FOrdGreaterThan %0, %1 : f32
    %5 = spv.FOrdGreaterThan %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FOrdGreaterThanEqualOp : SPV_LogicalBinaryOp<"FOrdGreaterThanEqual", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are ordered and Operand 1 is
    greater than or equal to Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fordgte-op ::= ssa-id `=` `spv.FOrdGreaterThanEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FOrdGreaterThanEqual %0, %1 : f32
    %5 = spv.FOrdGreaterThanEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FOrdLessThanOp : SPV_LogicalBinaryOp<"FOrdLessThan", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are ordered and Operand 1 is less
    than Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fordlt-op ::= ssa-id `=` `spv.FOrdLessThan` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FOrdLessThan %0, %1 : f32
    %5 = spv.FOrdLessThan %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FOrdLessThanEqualOp : SPV_LogicalBinaryOp<"FOrdLessThanEqual", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are ordered and Operand 1 is less
    than or equal to Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fordlte-op ::= ssa-id `=` `spv.FOrdLessThanEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FOrdLessThanEqual %0, %1 : f32
    %5 = spv.FOrdLessThanEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FOrdNotEqualOp : SPV_LogicalBinaryOp<"FOrdNotEqual", SPV_Float, [Commutative]> {
  let summary = "Floating-point comparison for being ordered and not equal.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    fordneq-op ::= ssa-id `=` `spv.FOrdNotEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FOrdNotEqual %0, %1 : f32
    %5 = spv.FOrdNotEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FUnordEqualOp : SPV_LogicalBinaryOp<"FUnordEqual", SPV_Float, [Commutative]> {
  let summary = "Floating-point comparison for being unordered or equal.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    funordequal-op ::= ssa-id `=` `spv.FUnordEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FUnordEqual %0, %1 : f32
    %5 = spv.FUnordEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FUnordGreaterThanOp : SPV_LogicalBinaryOp<"FUnordGreaterThan", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are unordered or Operand 1 is
    greater than  Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    funordgt-op ::= ssa-id `=` `spv.FUnordGreaterThan` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FUnordGreaterThan %0, %1 : f32
    %5 = spv.FUnordGreaterThan %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FUnordGreaterThanEqualOp : SPV_LogicalBinaryOp<"FUnordGreaterThanEqual", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are unordered or Operand 1 is
    greater than or equal to Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    funordgte-op ::= ssa-id `=` `spv.FUnordGreaterThanEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FUnordGreaterThanEqual %0, %1 : f32
    %5 = spv.FUnordGreaterThanEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FUnordLessThanOp : SPV_LogicalBinaryOp<"FUnordLessThan", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are unordered or Operand 1 is less
    than Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    funordlt-op ::= ssa-id `=` `spv.FUnordLessThan` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FUnordLessThan %0, %1 : f32
    %5 = spv.FUnordLessThan %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FUnordLessThanEqualOp : SPV_LogicalBinaryOp<"FUnordLessThanEqual", SPV_Float, []> {
  let summary = [{
    Floating-point comparison if operands are unordered or Operand 1 is less
    than or equal to Operand 2.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    funordlte-op ::= ssa-id `=` `spv.FUnordLessThanEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FUnordLessThanEqual %0, %1 : f32
    %5 = spv.FUnordLessThanEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_FUnordNotEqualOp : SPV_LogicalBinaryOp<"FUnordNotEqual", SPV_Float, [Commutative]> {
  let summary = "Floating-point comparison for being unordered or not equal.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    floating-point type.  They must have the same type, and they must have
    the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    funordneq-op ::= ssa-id `=` `spv.FUnordNotEqual` ssa-use, ssa-use
    ```

    For example:

    ```
    %4 = spv.FUnordNotEqual %0, %1 : f32
    %5 = spv.FUnordNotEqual %2, %3 : vector<4xf32>
    ```
  }];
}

// -----

def SPV_IEqualOp : SPV_LogicalBinaryOp<"IEqual", SPV_Integer, [Commutative]> {
  let summary = "Integer comparison for equality.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ```
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

def SPV_INotEqualOp : SPV_LogicalBinaryOp<"INotEqual", SPV_Integer, [Commutative]> {
  let summary = "Integer comparison for inequality.";

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 and Operand 2  must be a scalar or vector of
    integer type.  They must have the same component width, and they must
    have the same number of components as Result Type.

     Results are computed per component.

    ### Custom assembly form
    ```
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

def SPV_LogicalAndOp : SPV_LogicalBinaryOp<"LogicalAnd", SPV_Bool, [Commutative]> {
  let summary = [{
    Result is true if both Operand 1 and Operand 2 are true. Result is false
    if either Operand 1 or Operand 2 are false.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 must be the same as Result Type.

     The type of Operand 2 must be the same as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    logical-and ::= `spv.LogicalAnd` ssa-use `,` ssa-use
                    `:` operand-type
    ```

    For example:

    ```
    %2 = spv.LogicalAnd %0, %1 : i1
    %2 = spv.LogicalAnd %0, %1 : vector<4xi1>
    ```
  }];
}

// -----

def SPV_LogicalEqualOp : SPV_LogicalBinaryOp<"LogicalEqual", SPV_Bool, [Commutative]> {
  let summary = [{
    Result is true if Operand 1 and Operand 2 have the same value. Result is
    false if Operand 1 and Operand 2 have different values.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 must be the same as Result Type.

     The type of Operand 2 must be the same as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    logical-equal ::= `spv.LogicalEqual` ssa-use `,` ssa-use
                      `:` operand-type
    ```

    For example:

    ```
    %2 = spv.LogicalEqual %0, %1 : i1
    %2 = spv.LogicalEqual %0, %1 : vector<4xi1>
    ```
  }];
}

// -----

def SPV_LogicalNotOp : SPV_LogicalUnaryOp<"LogicalNot", SPV_Bool, []> {
  let summary = [{
    Result is true if Operand is false.  Result is false if Operand is true.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand must be the same as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    logical-not ::= `spv.LogicalNot` ssa-use `:` operand-type
    ```

    For example:

    ```
    %2 = spv.LogicalNot %0 : i1
    %2 = spv.LogicalNot %0 : vector<4xi1>
    ```
  }];

  let hasCanonicalizer = 1;
}

// -----

def SPV_LogicalNotEqualOp : SPV_LogicalBinaryOp<"LogicalNotEqual", SPV_Bool, [Commutative]> {
  let summary = [{
    Result is true if Operand 1 and Operand 2 have different values. Result
    is false if Operand 1 and Operand 2 have the same value.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 must be the same as Result Type.

     The type of Operand 2 must be the same as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    logical-not-equal ::= `spv.LogicalNotEqual` ssa-use `,` ssa-use
                          `:` operand-type
    ```

    For example:

    ```
    %2 = spv.LogicalNotEqual %0, %1 : i1
    %2 = spv.LogicalNotEqual %0, %1 : vector<4xi1>
    ```
  }];
}

// -----

def SPV_LogicalOrOp : SPV_LogicalBinaryOp<"LogicalOr", SPV_Bool, [Commutative]> {
  let summary = [{
    Result is true if either Operand 1 or Operand 2 is true. Result is false
    if both Operand 1 and Operand 2 are false.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

     The type of Operand 1 must be the same as Result Type.

     The type of Operand 2 must be the same as Result Type.

     Results are computed per component.

    ### Custom assembly form

    ```
    logical-or ::= `spv.LogicalOr` ssa-use `,` ssa-use
                    `:` operand-type
    ```

    For example:

    ```
    %2 = spv.LogicalOr %0, %1 : i1
    %2 = spv.LogicalOr %0, %1 : vector<4xi1>
    ```
  }];
}

// -----

def SPV_SGreaterThanOp : SPV_LogicalBinaryOp<"SGreaterThan", SPV_Integer, []> {
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
    ```
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

def SPV_SGreaterThanEqualOp : SPV_LogicalBinaryOp<"SGreaterThanEqual", SPV_Integer, []> {
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
    ```
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

def SPV_SLessThanOp : SPV_LogicalBinaryOp<"SLessThan", SPV_Integer, []> {
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
    ```
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

def SPV_SLessThanEqualOp : SPV_LogicalBinaryOp<"SLessThanEqual", SPV_Integer, []> {
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
    ```
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

def SPV_SelectOp : SPV_Op<"Select", [NoSideEffect]> {
  let summary = [{
    Select between two objects. Before version 1.4, results are only
    computed per component.
  }];

  let description = [{
    Before version 1.4, Result Type must be a pointer, scalar, or vector.

     The types of Object 1 and Object 2 must be the same as Result Type.

    Condition must be a scalar or vector of Boolean type.

    If Condition is a scalar and true, the result is Object 1. If Condition
    is a scalar and false, the result is Object 2.

    If Condition is a vector, Result Type must be a vector with the same
    number of components as Condition and the result is a mix of Object 1
    and Object 2: When a component of Condition is true, the corresponding
    component in the result is taken from Object 1, otherwise it is taken
    from Object 2.

    ### Custom assembly form

    ```
    scalar-type ::= integer-type | float-type | boolean-type
    select-object-type ::= scalar-type
                           | `vector<` integer-literal `x` scalar-type `>`
                           | pointer-type
    select-condition-type ::= boolean-type
                              | `vector<` integer-literal `x` boolean-type `>`
    select-op ::= ssa-id `=` `spv.Select` ssa-use, ssa-use, ssa-use
                  `:` select-condition-type `,` select-object-type
    ```

    For example:

    ```
    %3 = spv.Select %0, %1, %2 : i1, f32
    %3 = spv.Select %0, %1, %2 : i1, vector<3xi32>
    %3 = spv.Select %0, %1, %2 : vector<3xi1>, vector<3xf32>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Bool>:$condition,
    SPV_SelectType:$true_value,
    SPV_SelectType:$false_value
  );

  let results = (outs
    SPV_SelectType:$result
  );

  let builders = [OpBuilder<[{Builder *builder, OperationState &state,
                              Value cond, Value trueValue,
                              Value falseValue}]>];
}

// -----

def SPV_UGreaterThanOp : SPV_LogicalBinaryOp<"UGreaterThan", SPV_Integer, []> {
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
    ```
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    ugreater-than-op ::= ssa-id `=` `spv.UGreaterThan` ssa-use, ssa-use
                                    `:` integer-scalar-vector-type
    ```
    For example:

    ```
    %4 = spv.UGreaterThan %0, %1 : i32
    %5 = spv.UGreaterThan %2, %3 : vector<4xi32>

    ```
  }];
}

// -----

def SPV_UGreaterThanEqualOp : SPV_LogicalBinaryOp<"UGreaterThanEqual", SPV_Integer, []> {
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
    ```
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

def SPV_ULessThanOp : SPV_LogicalBinaryOp<"ULessThan", SPV_Integer, []> {
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
    ```
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

def SPV_ULessThanEqualOp :
  SPV_LogicalBinaryOp<"ULessThanEqual", SPV_Integer, []> {
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
    ```
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
