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

#ifndef XLA_CORE_COLLECTIVES_SYMMETRIC_MEMORY_H_
#define XLA_CORE_COLLECTIVES_SYMMETRIC_MEMORY_H_

#include <string>

#include "absl/strings/str_format.h"
#include "xla/stream_executor/kernel_args.h"

namespace xla {

// Symmetric memory allows memory allocations from different devices to be
// grouped into a symmetric memory allocation, where each device in a collective
// clique can access peer memory through the symmetric memory handle.
class SymmetricMemory {
 public:
  virtual ~SymmetricMemory() = default;
  virtual std::string ToString() const = 0;

  // A packed kernel argument type for passing symmetric memory to device
  // kernels (a platform-specific POD data type, happens to be a pointer).
  using PackedKernelArg = void*;
  virtual PackedKernelArg PackKernelArg() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymmetricMemory& mem) {
    absl::Format(&sink, "%s", mem.ToString());
  }
};

}  // namespace xla

namespace stream_executor {
template <>
struct KernelArgPacking<xla::SymmetricMemory*> {
  using Type = xla::SymmetricMemory::PackedKernelArg;
  static Type Pack(xla::SymmetricMemory* mem) { return mem->PackKernelArg(); }
};
}  // namespace stream_executor

#endif  // XLA_CORE_COLLECTIVES_SYMMETRIC_MEMORY_H_
