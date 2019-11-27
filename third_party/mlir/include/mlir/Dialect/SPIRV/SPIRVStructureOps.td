//===-- SPIRVStructureOps.td - MLIR SPIR-V Structure Ops ---*- tablegen -*-===//
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
// This file contains ops for defining the SPIR-V structure: module, function,
// and module-level operations. The representational form of these ops deviate
// from the SPIR-V binary format in order to utilize MLIR mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_STRUCTURE_OPS
#define SPIRV_STRUCTURE_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"

def SPV_AddressOfOp : SPV_Op<"_address_of", [InFunctionScope, NoSideEffect]> {
  let summary = "Get the address of a global variable.";

  let description = [{
    Variables in module scope are defined using symbol names. This op generates
    an SSA value that can be used to refer to the symbol within function scope
    for use in ops that expect an SSA value. This operation has no corresponding
    SPIR-V instruction; it's merely used for modelling purpose in the SPIR-V
    dialect. Since variables in module scope in SPIR-V dialect are of pointer
    type, this op returns a pointer type as well, and the type is the same as
    the variable referenced.

    ### Custom assembly form

    ``` {.ebnf}
    spv-address-of-op ::= ssa-id `=` `spv._address_of` symbol-ref-id
                                     `:` spirv-pointer-type
    ```

    For example:

    ```
    %0 = spv._address_of @global_var : !spv.ptr<f32, Input>
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$variable
  );

  let results = (outs
    SPV_AnyPtr:$pointer
  );

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let builders = [OpBuilder<[{Builder *builder, OperationState &state,
                              spirv::GlobalVariableOp var}]>];
}

def SPV_ConstantOp : SPV_Op<"constant", [NoSideEffect]> {
  let summary = "The op that declares a SPIR-V normal constant";

  let description = [{
    This op declares a SPIR-V normal constant. SPIR-V has multiple constant
    instructions covering different constant types:

    * `OpConstantTrue` and `OpConstantFalse` for boolean constants
    * `OpConstant` for scalar constants
    * `OpConstantComposite` for composite constants
    * `OpConstantNull` for null constants
    * ...

    Having such a plethora of constant instructions renders IR transformations
    more tedious. Therefore, we use a single `spv.constant` op to represent
    them all. Note that conversion between those SPIR-V constant instructions
    and this op is purely mechanical; so it can be scoped to the binary
    (de)serialization process.

    ### Custom assembly form

    ``` {.ebnf}
    spv-constant-op ::= ssa-id `=` `spv.constant` attribute-value
                        (`:` spirv-type)?
    ```

    For example:

    ```
    %0 = spv.constant true
    %1 = spv.constant dense<[2, 3]> : vector<2xf32>
    %2 = spv.constant [dense<3.0> : vector<2xf32>] : !spv.array<1xvector<2xf32>>
    ```

    TODO(antiagainst): support constant structs
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
    SPV_Type:$constant
  );

  let hasFolder = 1;

  let extraClassDeclaration = [{
    // Returns true if a constant can be built for the given `type`.
    static bool isBuildableWith(Type type);
  }];

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

def SPV_EntryPointOp : SPV_Op<"EntryPoint", [InModuleScope]> {
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

    Interface is a list of symbol references to `spv.globalVariable`
    operations. These declare the set of global variables from a
    module that form the interface of this entry point. The set of
    Interface symbols must be equal to or a superset of the
    `spv.globalVariable`s referenced by the entry point’s static call
    tree, within the interface’s storage classes.  Before version 1.4,
    the interface’s storage classes are limited to the Input and
    Output storage classes. Starting with version 1.4, the interface’s
    storage classes are all storage classes used in declaring all
    global variables referenced by the entry point’s call tree.

    ### Custom assembly form

    ``` {.ebnf}
    execution-model ::= "Vertex" | "TesellationControl" |
                        <and other SPIR-V execution models...>

    entry-point-op ::= ssa-id `=` `spv.EntryPoint` execution-model
                       symbol-reference (`, ` symbol-reference)*
    ```

    For example:

    ```
    spv.EntryPoint "GLCompute" @foo
    spv.EntryPoint "Kernel" @foo, @var1, @var2

    ```
  }];

  let arguments = (ins
    SPV_ExecutionModelAttr:$execution_model,
    FlatSymbolRefAttr:$fn,
    SymbolRefArrayAttr:$interface
  );

  let results = (outs);

  let autogenSerialization = 0;

  let builders = [OpBuilder<[{Builder *builder, OperationState &state,
                              spirv::ExecutionModel executionModel,
                              FuncOp function,
                              ArrayRef<Attribute> interfaceVars}]>];
}


def SPV_GlobalVariableOp : SPV_Op<"globalVariable", [InModuleScope, Symbol]> {
  let summary = [{
    Allocate an object in memory at module scope. The object is
    referenced using a symbol name.
  }];

  let description = [{
    The variable type must be an OpTypePointer. Its type operand is the type of
    object in memory.

    Storage Class is the Storage Class of the memory holding the object. It
    cannot be Generic. It must be the same as the Storage Class operand of
    the variable types. Only those storage classes that are valid at module
    scope (like Input, Output, StorageBuffer, etc.) are valid.

    Initializer is optional.  If Initializer is present, it will be
    the initial value of the variable’s memory content. Initializer
    must be an symbol defined from a constant instruction or other
    `spv.globalVariable` operation in module scope. Initializer must
    have the same type as the type of the defined symbol.

    ### Custom assembly form

    ``` {.ebnf}
    variable-op ::= `spv.globalVariable` spirv-type symbol-ref-id
                    (`initializer(` symbol-ref-id `)`)?
                    (`bind(` integer-literal, integer-literal `)`)?
                    (`built_in(` string-literal `)`)?
                    attribute-dict?
    ```

    where `initializer` specifies initializer and `bind` specifies the
    descriptor set and binding number. `built_in` specifies SPIR-V
    BuiltIn decoration associated with the op.

    For example:

    ```
    spv.globalVariable @var0 : !spv.ptr<f32, Input> @var0
    spv.globalVariable @var1 initializer(@var0) : !spv.ptr<f32, Output>
    spv.globalVariable @var2 bind(1, 2) : !spv.ptr<f32, Uniform>
    spv.globalVariable @var3 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    ```
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name,
    OptionalAttr<FlatSymbolRefAttr>:$initializer
  );

  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, "
      "TypeAttr type, ArrayRef<NamedAttribute> namedAttrs", [{
      state.addAttribute("type", type);
      state.addAttributes(namedAttrs);
    }]>,
    OpBuilder<[{Builder *builder, OperationState &state,
                Type type, StringRef name, unsigned descriptorSet,
                unsigned binding}]>
  ];

  let results = (outs);

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
    ::mlir::spirv::StorageClass storageClass() {
      return this->type().cast<::mlir::spirv::PointerType>().getStorageClass();
    }
  }];
}

def SPV_ModuleOp : SPV_Op<"module",
                          [IsolatedFromAbove,
                           SingleBlockImplicitTerminator<"ModuleEndOp">,
                           NativeOpTrait<"SymbolTable">]> {
  let summary = "The top-level op that defines a SPIR-V module";

  let description = [{
    This op defines a SPIR-V module using a MLIR region. The region contains
    one block. Module-level operations, including functions definitions,
    are all placed in this block.

    Using an op with a region to define a SPIR-V module enables "embedding"
    SPIR-V modules in other dialects in a clean manner: this op guarantees
    the validity and serializability of a SPIR-V module and thus serves as
    a clear-cut boundary.

    This op takes no operands and generates no results. This op should not
    implicitly capture values from the enclosing environment.

    This op has only one region, which only contains one block. The block
    must be terminated via the `spv._module_end` op.

    ### Custom assembly form

    ``` {.ebnf}
    addressing-model ::= `"Logical"` | `"Physical32"` | `"Physical64"`
    memory-model ::= `"Simple"` | `"GLSL450"` | `"OpenCL"` | `"VulkanKHR"`
    spv-module-op ::= `spv.module` addressing-model memory-model
                      region
                      (`attributes` attribute-dict)?
    ```

    For example:

    ```
    spv.module "Logical" "VulkanKHR" { }

    spv.module "Logical" "VulkanKHR" {
      func @do_nothing() -> () {
        spv.Return
      }
    } attributes {
      capability = ["Shader"],
      extension = ["SPV_KHR_16bit_storage"]
    }
    ```
  }];

  let arguments = (ins
    SPV_AddressingModelAttr:$addressing_model,
    SPV_MemoryModelAttr:$memory_model,
    OptionalAttr<StrArrayAttr>:$capabilities,
    OptionalAttr<StrArrayAttr>:$extensions,
    OptionalAttr<StrArrayAttr>:$extended_instruction_sets
  );

  let results = (outs);

  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<"Builder *, OperationState &state">,
                  OpBuilder<[{Builder *, OperationState &state,
                              IntegerAttr addressing_model,
                              IntegerAttr memory_model,
                              /*optional*/ArrayAttr capabilities = nullptr,
                              /*optional*/ArrayAttr extensions = nullptr,
                              /*optional*/ArrayAttr extended_instruction_sets = nullptr}]>];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let hasOpcode = 0;

  let autogenSerialization = 0;

  let extraClassDeclaration = [{
    Block& getBlock() {
      return this->getOperation()->getRegion(0).front();
    }
  }];
}

def SPV_ModuleEndOp : SPV_Op<"_module_end", [InModuleScope, Terminator]> {
  let summary = "The pseudo op that ends a SPIR-V module";

  let description = [{
    This op terminates the only block inside a `spv.module`'s only region.
    This op does not have a corresponding SPIR-V instruction and thus will
    not be serialized into the binary format; it is used solely to satisfy
    the structual requirement that an block must be ended with a terminator.
  }];

  let arguments = (ins);

  let results = (outs);

  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];

  let verifier = [{ return success(); }];

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

def SPV_ReferenceOfOp : SPV_Op<"_reference_of", [NoSideEffect]> {
  let summary = "Reference a specialization constant.";

  let description = [{
    Specialization constant in module scope are defined using symbol names.
    This op generates an SSA value that can be used to refer to the symbol
    within function scope for use in ops that expect an SSA value.
    This operation has no corresponding SPIR-V instruction; it's merely used
    for modelling purpose in the SPIR-V dialect. This op's return type is
    the same as the specialization constant.

    ### Custom assembly form

    ``` {.ebnf}
    spv-reference-of-op ::= ssa-id `=` `spv._reference_of` symbol-ref-id
                                       `:` spirv-scalar-type
    ```

    For example:

    ```
    %0 = spv._reference_of @spec_const : f32
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$spec_const
  );

  let results = (outs
    SPV_Type:$reference
  );

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

def SPV_SpecConstantOp : SPV_Op<"specConstant", [InModuleScope, Symbol]> {
  let summary = "The op that declares a SPIR-V specialization constant";

  let description = [{
    This op declares a SPIR-V scalar specialization constant. SPIR-V has
    multiple constant instructions covering different scalar types:

    * `OpSpecConstantTrue` and `OpSpecConstantFalse` for boolean constants
    * `OpSpecConstant` for scalar constants

    Similar as `spv.constant`, this op represents all of the above cases.
    `OpSpecConstantComposite` and `OpSpecConstantOp` are modelled with
    separate ops.

    ### Custom assembly form

    ``` {.ebnf}
    spv-spec-constant-op ::= `spv.specConstant` symbol-ref-id
                             `spec_id(` integer `)`
                             `=` attribute-value (`:` spirv-type)?
    ```

    where `spec_id` specifies the SPIR-V SpecId decoration associated with
    the op.

    For example:

    ```
    spv.specConstant @spec_const1 = true
    spv.specConstant @spec_const2 spec_id(5) = 42 : i32
    ```

    TODO(antiagainst): support composite spec constants with another op
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    AnyAttr:$default_value
  );

  let results = (outs);

  let hasOpcode = 0;

  let autogenSerialization = 0;
}

#endif // SPIRV_STRUCTURE_OPS
