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

#include <cstdint>

#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// Strongly-typed integer type for identifying processes in a distributed
// execution. Processes can belong to the same physical host, or can be
// distributed over multiple nodes.
TSL_LIB_GTL_DEFINE_INT_TYPE(ProcessId, int64_t);

template <typename Sink>
void AbslStringify(Sink& sink, ProcessId id) {
  absl::Format(&sink, "%d", id.value());
}

}  // namespace xla

#endif  // XLA_RUNTIME_PROCESS_ID_H_
