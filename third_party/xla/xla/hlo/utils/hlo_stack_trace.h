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

#ifndef XLA_HLO_UTILS_HLO_STACK_TRACE_H_
#define XLA_HLO_UTILS_HLO_STACK_TRACE_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_value.h"

namespace xla {

// Generates a human-readable report that breaks down memory usage by HLO stack
// traces.
//
// This function takes a list of HLO buffer allocations, constructs a
// hierarchical tree representing the call stacks responsible for those
// allocations, and formats this tree into a string. The report shows memory
// usage aggregated at each level of the stack, helping to identify which
// operations or computations contribute most to peak memory usage.
std::string FormatStackTraceBreakdown(
    const std::vector<std::pair<int64_t, const HloValue*>>& sized_buffers,
    const HloModule* module);

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_STACK_TRACE_H_
