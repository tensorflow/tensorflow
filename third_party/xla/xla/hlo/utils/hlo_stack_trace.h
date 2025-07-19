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

// Formats a stack breakdown from a list of (buffer size, value) pairs.
// module must not be null.
std::string FormatStackTraceBreakdown(
    const std::vector<std::pair<int64_t, const HloValue*>>& sized_buffers,
    const HloModule* module);

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_STACK_TRACE_H_
