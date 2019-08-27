//===-- SPIRVControlFlowOps.td - SPIR-V Control Flow Ops ---*- tablegen -*-===//
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
// This file contains control flow ops for the SPIR-V dialect. It corresponds
// to "3.32.17. Control-Flow Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifdef SPIRV_CONTROLFLOW_OPS
#else
#define SPIRV_CONTROLFLOW_OPS

#ifdef SPIRV_BASE
#else
include "mlir/SPIRV/SPIRVBase.td"
#endif // SPIRV_BASE

// -----

def SPV_ReturnOp : SPV_Op<"Return", [InFunctionScope, Terminator]> {
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
}

// -----

def SPV_ReturnValueOp : SPV_Op<"ReturnValue", [InFunctionScope, Terminator]> {
  let summary = "Return a value from a function.";

  let description = [{
    Value is the value returned, by copy, and must match the Return Type
    operand of the OpTypeFunction type of the OpFunction body this return
    instruction is in.

    This instruction must be the last instruction in a block.

    ### Custom assembly form

    ``` {.ebnf}
    return-value-op ::= `spv.ReturnValue` ssa-use `:` spirv-type
    ```

    For example:

    ```
    spv.ReturnValue %0 : f32
    ```
  }];

  let arguments = (ins
    SPV_Type:$value
  );

  let results = (outs);
}

#endif // SPIRV_CONTROLFLOW_OPS
