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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_IOPDDL_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_IOPDDL_H_

#include <cstdint>
#include <vector>

#include "xla/hlo/experimental/auto_sharding/auto_sharding.pb.h"
#include "xla/hlo/experimental/auto_sharding/iopddl.h"

namespace xla {
namespace spmd {

constexpr int64_t kInfinityInt = 1e18;

iopddl::Cost ConvertCost(double cost);

iopddl::Problem ConvertToProblem(const AutoShardingSolverRequest& request,
                                 bool use_follower_constraints = true);

AutoShardingSolverRequest ConvertToSolverRequest(
    const iopddl::Problem& problem);

std::vector<int64_t> GetFollowers(const iopddl::Problem& problem);

std::vector<iopddl::Edge> GetAliases(const iopddl::Problem& problem);

std::vector<iopddl::Edge> GetDeduplicatedEdges(const iopddl::Problem& problem);

// Returns true if the edge contains some infinite costs, but no other non-zero
// costs (these represent "aliases" that require special treatment in the
// solver).
bool IsEdgeAlias(const iopddl::Edge& edge);

void RandomizeCosts(iopddl::Problem& problem);

}  // namespace spmd
}  // namespace xla

#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_IOPDDL_H_
