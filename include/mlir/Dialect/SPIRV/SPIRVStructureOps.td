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
// This file contains ops for defining the SPIR-V structure: module, function,
// and module-level operations. The representational form of these ops deviate
// from the SPIR-V binary format in order to utilize MLIR mechanisms.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_STRUCTURE_OPS
#else
#define SPIRV_STRUCTURE_OPS

#ifdef SPIRV_BASE
#else
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

def SPV_ModuleOp : SPV_Op<"module", []> {
  let summary = "The top-level op that defines a SPIR-V module";

  let description = [{
    This op defines a SPIR-V module using a MLIR region. The region contains
    one block. Module-level operations, including functions definitions,
    are all placed in this block.

    Using an op with a region to define a SPIR-V module enables "embedding"
    SPIR-V modules in other dialects in a clean manner: this op guarantees
    the validaty and serializability of a SPIR-V module and thus serves as
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
    OptionalAttr<StrArrayAttr>:$capabilities,
    OptionalAttr<StrArrayAttr>:$extensions,
    OptionalAttr<StrArrayAttr>:$extended_instruction_sets,
    SPV_AddressingModelAttr:$addressing_model,
    SPV_MemoryModelAttr:$memory_model
  );

  let results = (outs);

  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<"Builder *, OperationState *state">];

  let hasOpcode = 0;

  let extraClassDeclaration = [{
    Block& getBlock() {
      return this->getOperation()->getRegion(0).front();
    }
  }];
}

def SPV_ModuleEndOp : SPV_Op<"_module_end", [Terminator, ModuleOnly]> {
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
}

def SPV_ConstantOp : SPV_Op<"constant", [NoSideEffect]> {
  let summary = "The op that declares a SPIR-V constant";

  let description = [{
    This op declares a SPIR-V constant. SPIR-V has multiple constant
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
    (de)serialzation process.

    ### Custom assembly form

    ``` {.ebnf}
    spv-constant-op ::= ssa-id `=` `spv.constant` attribute-value
                        (`:` spirv-type)?
    ```

    For example:

    ```
    %0 = spv.constant true
    %1 = spv.constant dense<vector<2xf32>, [2, 3]>
    %2 = spv.constant [dense<vector<2xf32>, 3.0>] : !spv.array<1xvector<2xf32>>
    ```
  }];

  let arguments = (ins
    AnyAttr:$value
  );

  let results = (outs
    SPV_Type:$constant
  );

  let hasOpcode = 0;
}

#endif // SPIRV_STRUCTURE_OPS
