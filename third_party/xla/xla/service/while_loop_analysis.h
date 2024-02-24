/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
#define XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Returns the precise trip count of the loop if it's statically known,
// nullopt otherwise.
//
// max_brute_force_iters limits the number of steps that are evaluated while
// trying to brute force a loop trip count. trip counts larger than
// max_brute_force_iters may be returned if we can pattern-match the loop
// condition.
std::optional<int64_t> ComputeWhileLoopTripCount(
    const HloInstruction *while_op, int64_t max_brute_force_iters = 128);

// Returns an upper bound on the trip count of the loop if it's statically
// known, nullopt otherwise.
std::optional<int64_t> ComputeWhileLoopTripCountUpperBound(
    const HloInstruction *while_op);

// The below function identifies a subset of all possible auxiliary
// induction variables (AIV). Specifically, candidates are gtes, e.g.,
// gte(param0, N)
std::vector<const HloInstruction *> GetAuxiliaryLoopInductionVars(
    const HloInstruction *while_op);
// Returns the tuple index of the loop induction variable if there is such an
// induction variable detected. Otherwise returns nullopt.
std::optional<int64_t> GetLoopInductionVarTupleIdx(
    const HloInstruction *while_op);

// Checks the following conditions:
//  - `i`, the induction varaiable, is initialized to a scalar constant K
//    (namely, `indvar_init`),
//  - the while condition does `i < N` or `i <= N` (where N is a know constant)
//  - the while body does `i++`.
// If so, it's trivial to compute the loop bound as `N - k` or `N - k + 1`,
// respectively.
std::optional<int64_t> MatchTrivialLoopTripCount(const HloInstruction *while_op,
                                                 int64_t indvar_tuple_idx,
                                                 const Literal &indvar_init);
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_ANALYSIS_H_
