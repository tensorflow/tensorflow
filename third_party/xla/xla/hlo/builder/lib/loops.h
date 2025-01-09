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

#ifndef XLA_HLO_BUILDER_LIB_LOOPS_H_
#define XLA_HLO_BUILDER_LIB_LOOPS_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Function that builds a loop condition. Takes as input a sequence of input
// values, and returns a boolean value representing if the condition succeeds.
typedef std::function<absl::StatusOr<XlaOp>(absl::Span<const XlaOp>,
                                            XlaBuilder*)>
    WhileLoopHelperConditionFunction;

// Function that builds a loop body. Takes as input a sequence of input values
// and returns a sequence of output values.
typedef std::function<absl::StatusOr<std::vector<XlaOp>>(
    absl::Span<const XlaOp>, XlaBuilder*)>
    WhileLoopHelperBodyFunction;

// Helper function for building an XLA while loop, where the values carried by
// the loop are a tuple of values, e.g., (a, b, c):
// while(
//   condition: (a, b, c) -> bool,
//   body: (a, b, c) -> (a, b, c)
//   init: (a, b, c)
// )
// 'name' is a descriptive name for the loop.
absl::StatusOr<std::vector<XlaOp>> WhileLoopHelper(
    const WhileLoopHelperConditionFunction& condition_function,
    const WhileLoopHelperBodyFunction& body_function,
    absl::Span<const XlaOp> initial_values, absl::string_view name,
    XlaBuilder* builder);

// Builds an XLA loop that repeats a computation `num_iterations` times.
//
// The body function (ForEachIndexBodyFunction) takes as input a pair of
// (current iteration number, loop-carried values), and returns an updated
// vector of the loop-carried values.
typedef std::function<absl::StatusOr<std::vector<XlaOp>>(
    XlaOp, absl::Span<const XlaOp>, XlaBuilder*)>
    ForEachIndexBodyFunction;

absl::StatusOr<std::vector<XlaOp>> ForEachIndex(
    int64_t num_iterations, PrimitiveType num_iterations_type,
    const ForEachIndexBodyFunction& body_function,
    absl::Span<const XlaOp> initial_values, absl::string_view name,
    XlaBuilder* builder);

}  // namespace xla

#endif  // XLA_HLO_BUILDER_LIB_LOOPS_H_
