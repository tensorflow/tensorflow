/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_DYNAMIC_SLICING_UTILS_H_
#define XLA_SERVICE_GPU_DYNAMIC_SLICING_UTILS_H_

#include "absl/container/inlined_vector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/util.h"

namespace xla::gpu {

// Each user of `instr` that goes into a DUS will have an entry in the returned
// vector.
// Each entry contains the sliced paths for that user, i.e. the sequence of ops
// following the dataflow from the user itself to the DUS (included).
absl::InlinedVector<absl::InlinedVector<HloInstruction*, 2>, 4>
GetSlicedUserPaths(const HloInstruction& instr, const CallGraph& call_graph,
                   HloPredicate is_noop =
                       HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                                        HloOpcode::kGetTupleElement>,
                   bool check_alignment = true);

absl::InlinedVector<HloInstruction*, 8> GetSlicedOperandPaths(
    const HloInstruction& instr, const CallGraph& call_graph,
    HloPredicate is_noop = HloPredicateIsOp<
        HloOpcode::kBitcast, HloOpcode::kTuple, HloOpcode::kGetTupleElement>,
    bool check_alignment = true);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_DYNAMIC_SLICING_UTILS_H_
