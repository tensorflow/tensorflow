//===-- SPIRVNonUniformOps.td - MLIR SPIR-V NonUniform Ops -*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains non-uniform ops for the SPIR-V dialect. It corresponds to
// "3.32.24. Non-Uniform Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_NON_UNIFORM_OPS
#define SPIRV_NON_UNIFORM_OPS

// -----

def SPV_GroupNonUniformBallotOp : SPV_Op<"GroupNonUniformBallot", []> {
  let summary = [{
    Returns a bitfield value combining the Predicate value from all
    invocations in the group that execute the same dynamic instance of this
    instruction. The bit is set to one if the corresponding invocation is
    active and the Predicate for that invocation evaluated to true;
    otherwise, it is set to zero.
  }];

  let description = [{
    Result Type  must be a vector of four components of integer type scalar,
    whose Signedness operand is 0.

    Result is a set of bitfields where the first invocation is represented
    in the lowest bit of the first vector component and the last (up to the
    size of the group) is the higher bit number of the last bitmask needed
    to represent all bits of the group invocations.

    Execution must be Workgroup or Subgroup Scope.

    Predicate must be a Boolean type.

    ### Custom assembly form

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    non-uniform-ballot-op ::= ssa-id `=` `spv.GroupNonUniformBallot` scope
                              ssa-use `:` `vector` `<` 4 `x` `integer-type` `>`
    ```

    For example:

    ```
    %0 = spv.GroupNonUniformBallot "SubGroup" %predicate : vector<4xi32>
    ```
  }];

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope,
    SPV_Bool:$predicate
  );

  let results = (outs
    SPV_IntVec4:$result
  );
}

// -----

#endif // SPIRV_NON_UNIFORM_OPS

