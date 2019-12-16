//===-- SPIRVAtomicOps.td - MLIR SPIR-V Atomic Ops ---------*- tablegen -*-===//
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
// This file contains atomic ops for the SPIR-V dialect. It corresponds to
// "3.32.18. Atomic Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_ATOMIC_OPS
#define SPIRV_ATOMIC_OPS

// -----

def SPV_AtomicCompareExchangeWeakOp : SPV_Op<"AtomicCompareExchangeWeak", []> {
  let summary = "Deprecated (use OpAtomicCompareExchange).";

  let description = [{
    Has the same semantics as OpAtomicCompareExchange.

    Memory must be a valid memory Scope.

    ### Custom assembly form

    ```
    scope ::= `"CrossDevice"` | `"Device"` | `"Workgroup"` | ...

    memory-semantics ::= `"None"` | `"Acquire"` | "Release"` | ...

    atomic-compare-exchange-weak-op ::=
        `spv.AtomicCompareExchangeWeak` scope memory-semantics memory-semantics
                                        ssa-use `,` ssa-use `,` ssa-use
                                        `:` spv-pointer-type
    ```

    For example:

    ```
    %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Acquire" "None"
                                       %pointer, %value, %comparator
                                       : !spv.ptr<i32, WorkGroup>
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$equal_semantics,
    SPV_MemorySemanticsAttr:$unequal_semantics,
    SPV_Integer:$value,
    SPV_Integer:$comparator
  );

  let results = (outs
    SPV_Integer:$result
  );
}

// -----

#endif // SPIRV_ATOMIC_OPS
