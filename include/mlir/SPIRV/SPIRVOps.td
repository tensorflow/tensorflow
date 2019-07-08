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

#ifdef SPIRV_OPS
#else
#define SPIRV_OPS

#ifdef SPIRV_BASE
#else
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

#ifdef SPIRV_STRUCTURE_OPS
#else
// Pull in ops for defining the SPIR-V module structure
include "mlir/SPIRV/SPIRVStructureOps.td"
#endif // SPIRV_STRUCTURE_OPS

def SPV_EntryPointOp : SPV_Op<"EntryPoint", [ModuleOnly]> {
  let summary = [{
    Declare an entry point, its execution model, and its interface.
  }];

  let description = [{
    Execution Model is the execution model for the entry point and its
    static call tree. See Execution Model.

    Entry Point must be the Result <id> of an OpFunction instruction.

    Name is a name string for the entry point. A module cannot have
    two OpEntryPoint instructions with the same Execution Model and
    the same Name string.

    Interface is a list of <id> of global OpVariable
    instructions. These declare the set of global variables from a
    module that form the interface of this entry point. The set of
    Interface <id> must be equal to or a superset of the global
    OpVariable Result <id> referenced by the entry point’s static call
    tree, within the interface’s storage classes. Before version 1.4,
    the interface’s storage classes are limited to the Input and
    Output storage classes. Starting with version 1.4, the interface’s
    storage classes are all storage classes used in declaring all
    global variables referenced by the entry point’s call tree.

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
    %4 = spv.EntryPoint "GLCompute" "foo"
    %3 = spv.EntryPoint "Kernel" "foo", %1, %2 : !spv.ptr<f32, Input>, !spv.ptr<f32, Output>

    ```
  }];

  let arguments = (ins
    SPV_ExecutionModelAttr:$execution_model,
    FunctionAttr:$fn,
    Variadic<SPV_AnyPtr>:$interface
  );

  let results = (outs SPV_EntryPoint:$id);

  let opcode = 15;
}

def SPV_ExecutionModeOp : SPV_Op<"ExecutionMode", [ModuleOnly]> {
  let summary = "Declare an execution mode for an entry point.";

  let description = [{
    Entry Point must be the Entry Point <id> operand of an
    OpEntryPoint instruction.

    Mode is the execution mode. See Execution Mode.

    This instruction is only valid when the Mode operand is an
    execution mode that takes no Extra Operands, or takes Extra
    Operands that are not <id> operands.

    ### Custom assembly form

    ``` {.ebnf}
    execution-mode ::= "Invocations" | "SpacingEqual" |
                       <and other SPIR-V execution modes...>

    execution-mode-op ::= `spv.ExecutionMode ` ssa-use execution-mode
                          (integer-literal (`, ` integer-literal)* )?
    ```

    For example:

    ```
    spv.ExecutionMode %7 "ContractionOff"
    spv.ExecutionMode %8 "LocalSizeHint", 3, 4, 5
    ```
  }];

  let arguments = (ins
    SPV_EntryPoint:$entry_point,
    SPV_ExecutionModeAttr:$execution_mode,
    OptionalAttr<I32ArrayAttr>:$values
  );

  let verifier = [{ return success(); }];

  let opcode = 16;
}

def SPV_FMulOp : SPV_Op<"FMul", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Floating-point multiplication of Operand 1 and Operand 2";

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    The types of Operand 1 and Operand 2 both must be the same as Result Type.

    Results are computed per component.
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

  let opcode = 133;
}

def SPV_LoadOp : SPV_Op<"Load"> {
  let summary = "Load value through a pointer.";

  let description = [{
    Result Type is the type of the loaded object. It must be a type
    with fixed size; i.e., it cannot be, nor include, any
    OpTypeRuntimeArray types.

    Pointer is the pointer to load through. Its type must be an
    OpTypePointer whose Type operand is the same as Result Type.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory
    operand None.

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

  let opcode = 61;
}

def SPV_ReturnOp : SPV_Op<"Return", [Terminator]> {
  let summary = "Return with no value from a function with void return type";

  let description = [{
    This instruction must be the last instruction in a block.
  }];

  let arguments = (ins);

  let results = (outs);

  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];

  let verifier = [{ return verifyReturn(*this); }];

  let opcode = 253;
}

def SPV_StoreOp : SPV_Op<"Store"> {
  let summary = "Store through a pointer.";

  let description = [{
    Pointer is the pointer to store through. Its type must be an
    OpTypePointer whose Type operand is the same as the type of
    Object.

    Object is the object to store.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory
    operand None.

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
    OptionalAttr<APIntAttr>:$alignment
  );

  let opcode = 62;
}

def SPV_VariableOp : SPV_Op<"Variable"> {
  let summary = [{
    Allocate an object in memory, resulting in a pointer to it, which can be
    used with OpLoad and OpStore
  }];

  let description = [{
    Result Type must be an OpTypePointer. Its Type operand is the type of object
    in memory.

    Storage Class is the Storage Class of the memory holding the object. It
    cannot be Generic. It must be the same as the Storage Class operand of the
    Result Type.

    Initializer is optional. If Initializer is present, it will be the initial
    value of the variable’s memory content. Initializer must be an <id> from a
    constant instruction or a global (module scope) OpVariable instruction.
    Initializer must have the same type as the type pointed to by Result Type.

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

  let opcode = 59;
}

#endif // SPIRV_OPS
