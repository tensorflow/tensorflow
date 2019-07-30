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

def SPV_EntryPointOp : SPV_Op<"EntryPoint", [ModuleOnly]> {
  let summary = [{
    Declare an entry point, its execution model, and its interface.
  }];

  let description = [{
    Execution Model is the execution model for the entry point and its
    static call tree. See Execution Model.

    Entry Point must be the Result <id> of an OpFunction instruction.

    Name is a name string for the entry point. A module cannot have two
    OpEntryPoint instructions with the same Execution Model and the same
    Name string.

    Interface is a list of <id> of global OpVariable instructions. These
    declare the set of global variables from a module that form the
    interface of this entry point. The set of Interface <id> must be equal
    to or a superset of the global OpVariable Result <id> referenced by the
    entry point’s static call tree, within the interface’s storage classes.
    Before version 1.4, the interface’s storage classes are limited to the
    Input and Output storage classes. Starting with version 1.4, the
    interface’s storage classes are all storage classes used in declaring
    all global variables referenced by the entry point’s call tree.

    Interface <id> are forward references. Before version 1.4, duplication
    of these <id> is tolerated. Starting with version 1.4, an <id> must not
    appear more than once.

    ### Custom assembly form

    ``` {.ebnf}
    execution-model ::= "Vertex" | "TesellationControl" |
                        <and other SPIR-V execution models...>

    entry-point-op ::= ssa-id ` = spv.EntryPoint ` execution-model fn-name
                       (ssa-use ( `, ` ssa-use)* ` : `
                        pointer-type ( `, ` pointer-type)* )?
    ```

    For example:

    ```
    spv.EntryPoint "GLCompute" @foo
    spv.EntryPoint "Kernel" @foo, %1, %2 : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>

    ```
  }];

  let arguments = (ins
    SPV_ExecutionModelAttr:$execution_model,
    SymbolRefAttr:$fn,
    Variadic<SPV_AnyPtr>:$interface
  );

  let results = (outs);
  let autogenSerialization = 0;
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

def SPV_FMulOp : SPV_Op<"FMul", [NoSideEffect, SameOperandsAndResultType]> {
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
    execution-mode-op ::= `spv.FMul` ssa-use, ssa-use
                          `:` float-scalar-vector-type
    ```

    For example:

    ```
    spv.FMul %0, %1 : f32
    spv.FMul %2, %3 : vector<4xf32>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Float>:$operand1,
    SPV_ScalarOrVectorOf<SPV_Float>:$operand2
  );

  let results = (outs
    SPV_ScalarOrVectorOf<AnyFloat>:$result
  );

  let parser = [{ return impl::parseBinaryOp(parser, result); }];
  let printer = [{ return impl::printBinaryOp(getOperation(), p); }];

  // No additional verification needed in addition to the ODS-generated ones.
  let verifier = [{ return success(); }];
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
                    attribute-dict? `:` spirv-pointer-type
    ```

    where `init` specifies initializer and `bind` specifies the descriptor set
    and binding number.

    For example:

    ```
    %0 = spv.constant ...

    %1 = spv.Variable : !spv.ptr<f32, Function>
    %2 = spv.Variable init(%0): !spv.ptr<f32, Private>
    %3 = spv.Variable init(%0) bind(1, 2): !spv.ptr<f32, Uniform>
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
