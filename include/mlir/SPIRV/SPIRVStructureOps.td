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

    Regularity is one of the design goals of SPIR-V. All concepts are
    represented as SPIR-V instructions, including declaring extensions and
    capabilities, defining types and constants, defining functions, attaching
    additional properties to computation results, etc. This way favors driver
    consumption but not necessarily compiler transformations. The purpose of
    the SPIR-V dialect is to serve as the proxy of the binary format and to
    facilitate transformations. So we adjust how certain module-level SPIR-V
    instructions are represented in the SPIR-V dialect. Notably,

    * Requirements for capabilities, extensions, extended instruction sets,
      addressing model, and memory model is conveyed using op attributes.
      This is considered better because these information are for the
      exexcution environment. It's eaiser to probe them if on the module op
      itself.

    This op takes no operands and generates no results. This op should not
    implicitly capture values from the enclosing environment.
  }];

  let arguments = (ins
    OptionalAttr<StrArrayAttr>:$capabilities,
    OptionalAttr<StrArrayAttr>:$extensions,
    OptionalAttr<StrArrayAttr>:$extended_instruction_sets,
    SPV_AddressingModelAttr:$addressing_model,
    SPV_MemoryModelAttr:$memory_model
  );

  let results = (outs);

  let numRegions = 1;

  // Custom parser and printer implemented by static functions in SPVOps.cpp
  let parser = [{ return parseModule(parser, result); }];
  let printer = [{ printModule(getOperation(), p); }];
}

#endif // SPIRV_STRUCTURE_OPS
