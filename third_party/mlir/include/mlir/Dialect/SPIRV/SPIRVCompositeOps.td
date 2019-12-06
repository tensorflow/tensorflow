//===-- SPIRVCompositeOps.td - MLIR SPIR-V Composite Ops ---*- tablegen -*-===//
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
// This file contains composite ops for SPIR-V dialect. It corresponds
// to "3.32.12. Composite Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_COMPOSITE_OPS
#define SPIRV_COMPOSITE_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"

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

  let hasFolder = 1;
}

// -----

def SPV_CompositeInsertOp : SPV_Op<"CompositeInsert", [NoSideEffect]> {
  let summary = [{
    Make a copy of a composite object, while modifying one part of it.
  }];

  let description = [{
    Result Type must be the same type as Composite.

    Object is the object to use as the modified part.

    Composite is the composite to copy all but the modified part from.

    Indexes walk the type hierarchy of Composite to the desired depth,
    potentially down to component granularity, to select the part to modify.
    All indexes must be in bounds. All composite constituents use zero-based
    numbering, as described by their OpType… instruction. The type of the
    part selected to modify must match the type of Object.

    ### Custom assembly form

    ``` {.ebnf}
    composite-insert-op ::= ssa-id `=` `spv.CompositeInsert` ssa-use, ssa-use
                            `[` integer-literal (',' integer-literal)* `]`
                            `:` object-type `into` composite-type
    ```

    For example:

    ```
    %0 = spv.CompositeInsert %object, %composite[1 : i32] : f32 into !spv.array<4xf32>
    ```
  }];

  let arguments = (ins
    SPV_Type:$object,
    SPV_Composite:$composite,
    I32ArrayAttr:$indices
  );

  let results = (outs
    SPV_Composite:$result
  );
}

#endif // SPIRV_COMPOSITE_OPS
