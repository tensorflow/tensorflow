//===-- SPIRVOps.td - MLIR SPIR-V Op Definitions Spec ------*- tablegen -*-===//
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
// This is the main operation definition specification file for SPIR-V
// operations.
//
//===----------------------------------------------------------------------===//

// Note that for each op in this file, we use a tool to automatically generate
// certain sections in its definition: basic structure, summary, description.
// So modifications to these sections will not be respected. Modifications to
// op traits, arguments, results, and sections after the results are retained.
// Besides, ops in this file must be separated via the '// -----' marker.

#ifdef SPIRV_OPS
#else
#define SPIRV_OPS

#ifdef SPIRV_BASE
#else
include "mlir/Dialect/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

#ifdef SPIRV_STRUCTURE_OPS
#else
// Pull in ops for defining the SPIR-V module structure
include "mlir/Dialect/SPIRV/SPIRVStructureOps.td"
#endif // SPIRV_STRUCTURE_OPS

// -----

def SPV_AccessChainOp : SPV_Op<"AccessChain", [NoSideEffect]> {
  let summary = [{
    Create a pointer into a composite object that can be used with OpLoad
    and OpStore.
  }];

  let description = [{
    Result Type must be an OpTypePointer. Its Type operand must be the type
    reached by walking the Base’s type hierarchy down to the last provided
    index in Indexes, and its Storage Class operand must be the same as the
    Storage Class of Base.

    Base must be a pointer, pointing to the base of a composite object.

    Indexes walk the type hierarchy to the desired depth, potentially down
    to scalar granularity. The first index in Indexes will select the top-
    level member/element/component/element of the base composite. All
    composite constituents use zero-based numbering, as described by their
    OpType… instruction. The second index will apply similarly to that
    result, and so on. Once any non-composite type is reached, there must be
    no remaining (unused) indexes.

     Each index in Indexes

    - must be a scalar integer type,

    - is treated as a signed count, and

    - must be an OpConstant when indexing into a structure.

    ### Custom assembly form
    ``` {.ebnf}
    access-chain-op ::= ssa-id `=` `spv.AccessChain` ssa-use
                        `[` ssa-use (',' ssa-use)* `]`
                        `:` pointer-type
    ```

    For example:

    ```
    %0 = "spv.constant"() { value = 1: i32} : () -> i32
    %1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
    %2 = spv.AccessChain %1[%0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
    %3 = spv.Load "Function" %2 ["Volatile"] : !spv.array<4xf32>
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$base_ptr,
    Variadic<SPV_Integer>:$indices
  );

  let results = (outs
    SPV_AnyPtr:$component_ptr
  );
}

// -----

def SPV_CompositeExtractOp : SPV_Op<"CompositeExtract", [NoSideEffect]> {
  let summary = "Extract a part of a composite object.";

  let description = [{
    Result Type must be the type of object selected by the last provided
    index.  The instruction result is the extracted object.

    Composite is the composite to extract from.

    Indexes walk the type hierarchy, potentially down to component
    granularity, to select the part to extract. All indexes must be in
    bounds.  All composite constituents use zero-based numbering, as
    described by their OpType… instruction.

    ### Custom assembly form

    ``` {.ebnf}
    composite-extract-op ::= ssa-id `=` `spv.CompositeExtract` ssa-use
                             `[` integer-literal (',' integer-literal)* `]`
                             `:` composite-type
    ```

    For example:

    ```
    %0 = spv.Variable : !spv.ptr<!spv.array<4x!spv.array<4xf32>>, Function>
    %1 = spv.Load "Function" %0 ["Volatile"] : !spv.array<4x!spv.array<4xf32>>
    %2 = spv.CompositeExtract %1[1 : i32] : !spv.array<4x!spv.array<4xf32>>
    ```

  }];

  let arguments = (ins
    SPV_Composite:$composite,
    I32ArrayAttr:$indices
  );

  let results = (outs
    SPV_Type:$component
  );
}

// -----

def SPV_ExecutionModeOp : SPV_Op<"ExecutionMode", [ModuleOnly]> {
  let summary = "Declare an execution mode for an entry point.";

  let description = [{
    Entry Point must be the Entry Point <id> operand of an OpEntryPoint
    instruction.

    Mode is the execution mode. See Execution Mode.

    This instruction is only valid when the Mode operand is an execution
    mode that takes no Extra Operands, or takes Extra Operands that are not
    <id> operands.

    ### Custom assembly form

    ``` {.ebnf}
    execution-mode ::= "Invocations" | "SpacingEqual" |
                       <and other SPIR-V execution modes...>

    execution-mode-op ::= `spv.ExecutionMode ` ssa-use execution-mode
                          (integer-literal (`, ` integer-literal)* )?
    ```

    For example:

    ```
    spv.ExecutionMode @foo "ContractionOff"
    spv.ExecutionMode @bar "LocalSizeHint", 3, 4, 5
    ```
  }];

  let arguments = (ins
    SymbolRefAttr:$fn,
    SPV_ExecutionModeAttr:$execution_mode,
    OptionalAttr<I32ArrayAttr>:$values
  );

  let results = (outs);

  let verifier = [{ return success(); }];

  let autogenSerialization = 0;
}

// -----

def SPV_FAddOp : SPV_ArithmeticOp<"FAdd", SPV_Float, [Commutative]> {
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

def SPV_FDivOp : SPV_ArithmeticOp<"FDiv", SPV_Float> {
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

def SPV_FModOp : SPV_ArithmeticOp<"FMod", SPV_Float> {
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

def SPV_FMulOp : SPV_ArithmeticOp<"FMul", SPV_Float, [Commutative]> {
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

def SPV_FRemOp : SPV_ArithmeticOp<"FRem", SPV_Float> {
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

def SPV_FSubOp : SPV_ArithmeticOp<"FSub", SPV_Float> {
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

def SPV_IAddOp : SPV_ArithmeticOp<"IAdd", SPV_Integer, [Commutative]> {
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

def SPV_IMulOp : SPV_ArithmeticOp<"IMul", SPV_Integer, [Commutative]> {
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

def SPV_ISubOp : SPV_ArithmeticOp<"ISub", SPV_Integer> {
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

def SPV_LoadOp : SPV_Op<"Load", []> {
  let summary = "Load through a pointer.";

  let description = [{
    Result Type is the type of the loaded object. It must be a type with
    fixed size; i.e., it cannot be, nor include, any OpTypeRuntimeArray
    types.

    Pointer is the pointer to load through.  Its type must be an
    OpTypePointer whose Type operand is the same as Result Type.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory operand
    None.

    ### Custom assembly form

    ``` {.ebnf}
    memory-access ::= `"None"` | `"Volatile"` | `"Aligned", ` integer-literal
                    | `"NonTemporal"`

    load-op ::= ssa-id ` = spv.Load ` storage-class ssa-use
                (`[` memory-access `]`)? ` : ` spirv-element-type
    ```

    For example:

    ```
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.Load "Function" %0 : f32
    %2 = spv.Load "Function" %0 ["Volatile"] : f32
    %3 = spv.Load "Function" %0 ["Aligned", 4] : f32
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$ptr,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs
    SPV_Type:$value
  );
}

// -----

def SPV_ReturnOp : SPV_Op<"Return", [Terminator]> {
  let summary = "Return with no value from a function with void return type.";

  let description = [{
    This instruction must be the last instruction in a block.

    ### Custom assembly form

    ``` {.ebnf}
    return-op ::= `spv.Return`
    ```
  }];

  let arguments = (ins);

  let results = (outs);

  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];

  let verifier = [{ return verifyReturn(*this); }];
}

// -----

def SPV_SDivOp : SPV_ArithmeticOp<"SDiv", SPV_Integer> {
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

def SPV_SModOp : SPV_ArithmeticOp<"SMod", SPV_Integer> {
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

def SPV_SRemOp : SPV_ArithmeticOp<"SRem", SPV_Integer> {
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

def SPV_StoreOp : SPV_Op<"Store", []> {
  let summary = "Store through a pointer.";

  let description = [{
    Pointer is the pointer to store through.  Its type must be an
    OpTypePointer whose Type operand is the same as the type of Object.

    Object is the object to store.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory operand
    None.

    ### Custom assembly form

    ``` {.ebnf}
    store-op ::= `spv.Store ` storage-class ssa-use `, ` ssa-use `, `
                  (`[` memory-access `]`)? `:` spirv-element-type
    ```

    For example:

    ```
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.FMul ... : f32
    spv.Store "Function" %0, %1 : f32
    spv.Store "Function" %0, %1 ["Volatile"] : f32
    spv.Store "Function" %0, %1 ["Aligned", 4] : f32
  }];

  let arguments = (ins
    SPV_AnyPtr:$ptr,
    SPV_Type:$value,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs);
}

// -----

def SPV_UDivOp : SPV_ArithmeticOp<"UDiv", SPV_Integer> {
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

// -----

def SPV_UModOp : SPV_ArithmeticOp<"UMod", SPV_Integer> {
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

// -----

def SPV_VariableOp : SPV_Op<"Variable", []> {
  let summary = [{
    Allocate an object in memory, resulting in a pointer to it, which can be
    used with OpLoad and OpStore.
  }];

  let description = [{
    Result Type must be an OpTypePointer. Its Type operand is the type of
    object in memory.

    Storage Class is the Storage Class of the memory holding the object. It
    cannot be Generic. It must be the same as the Storage Class operand of
    the Result Type.

    Initializer is optional.  If Initializer is present, it will be the
    initial value of the variable’s memory content. Initializer must be an
    <id> from a constant instruction or a global (module scope) OpVariable
    instruction. Initializer must have the same type as the type pointed to
    by Result Type.

    ### Custom assembly form

    ``` {.ebnf}
    variable-op ::= ssa-id `=` `spv.Variable` (`init(` ssa-use `)`)?
                    (`bind(` integer-literal, integer-literal `)`)?
                    (`built_in(` string-literal `)`)?
                    attribute-dict? `:` spirv-pointer-type
    ```

    where `init` specifies initializer and `bind` specifies the
    descriptor set and binding number. `built_in` specifies SPIR-V
    BuiltIn decoration associated with the op.

    For example:

    ```
    %0 = spv.constant ...

    %1 = spv.Variable : !spv.ptr<f32, Function>
    %2 = spv.Variable init(%0): !spv.ptr<f32, Private>
    %3 = spv.Variable init(%0) bind(1, 2): !spv.ptr<f32, Uniform>
    %3 = spv.Variable built_in("GlobalInvocationID") : !spv.ptr<vector<3xi32>, Uniform>
    ```
  }];

  let arguments = (ins
    SPV_StorageClassAttr:$storage_class,
    SPV_Optional<AnyType>:$initializer
  );

  let results = (outs
    SPV_AnyPtr:$pointer
  );
}

// -----

#endif // SPIRV_OPS
