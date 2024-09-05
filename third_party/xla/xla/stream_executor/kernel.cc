/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/kernel.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace stream_executor {

std::optional<int64_t> KernelMetadata::registers_per_thread() const {
  return registers_per_thread_;
}

std::optional<int64_t> KernelMetadata::shared_memory_bytes() const {
  return shared_memory_bytes_;
}

void KernelMetadata::set_registers_per_thread(int registers_per_thread) {
  registers_per_thread_ = registers_per_thread;
}

void KernelMetadata::set_shared_memory_bytes(int shared_memory_bytes) {
  shared_memory_bytes_ = shared_memory_bytes;
}

//===----------------------------------------------------------------------===//
// Kernel
//===----------------------------------------------------------------------===//

void Kernel::set_name(absl::string_view name) { name_ = std::string(name); }

}  // namespace stream_executor
