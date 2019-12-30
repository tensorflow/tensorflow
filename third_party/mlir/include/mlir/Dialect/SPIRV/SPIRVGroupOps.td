//===-- SPIRVGroupOps.td - MLIR SPIR-V (Sub)Group Ops ------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains group and subgroup ops for the SPIR-V dialect. It
// corresponds to "3.32.21. Group and Subgroup Instructions" of the SPIR-V
// specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_GROUP_OPS
#define SPIRV_GROUP_OPS

// -----

def SPV_SubgroupBallotKHROp : SPV_Op<"SubgroupBallotKHR", []> {
  let summary = "See extension SPV_KHR_shader_ballot";

  let description = [{
    Computes a bitfield value combining the Predicate value from all invocations
    in the current Subgroup that execute the same dynamic instance of this
    instruction. The bit is set to one if the corresponding invocation is active
    and the predicate is evaluated to true; otherwise, it is set to zero.

    Predicate must be a Boolean type.

    Result Type must be a 4 component vector of 32 bit integer types.

    Result is a set of bitfields where the first invocation is represented in bit
    0 of the first vector component and the last (up to SubgroupSize) is the
    higher bit number of the last bitmask needed to represent all bits of the
    subgroup invocations.

    ### Custom assembly form

    ```
    subgroup-ballot-op ::= ssa-id `=` `spv.SubgroupBallotKHR`
                                ssa-use `:` `vector` `<` 4 `x` `i32` `>`
    ```

    For example:

    ```
    %0 = spv.SubgroupBallotKHR %predicate : vector<4xi32>
    ```
  }];

  let arguments = (ins
    SPV_Bool:$predicate
  );

  let results = (outs
    SPV_I32Vec4:$result
  );

  let verifier = [{ return success(); }];
}

// -----

#endif // SPIRV_GROUP_OPS
