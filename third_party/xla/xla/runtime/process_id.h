/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_PROCESS_ID_H_
#define XLA_RUNTIME_PROCESS_ID_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// Strongly-typed integer type for identifying processes in a distributed
// execution. Processes can belong to the same physical host, or can be
// distributed over multiple nodes.
TSL_LIB_GTL_DEFINE_INT_TYPE(ProcessId, int32_t);

template <typename Sink>
void AbslStringify(Sink& sink, ProcessId id) {
  absl::Format(&sink, "%d", id.value());
}

// StrJoin for processes that shortens long list of processes for readability.
//
// It is not uncommon to see in XLA a list of processes with more than 1k
// of entries. We don't need to print them all to get a human readable list
// of processes for logging and debugging.
inline std::string HumanReadableProcesses(absl::Span<const ProcessId> processes,
                                          absl::string_view separator = ",",
                                          size_t first = 10, size_t last = 4) {
  if (processes.size() > first + last) {
    return absl::StrCat(absl::StrJoin(processes.first(first), separator), "...",
                        absl::StrJoin(processes.last(last), separator));
  }
  return absl::StrJoin(processes, separator);
}

}  // namespace xla

#endif  // XLA_RUNTIME_PROCESS_ID_H_
