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

#ifndef XLA_PYTHON_IFRT_MPMD_EXECUTABLE_H_
#define XLA_PYTHON_IFRT_MPMD_EXECUTABLE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"

namespace xla {
namespace ifrt {

// Wraps a computation that has been fully compiled and loaded for MPMD
// execution.
class MpmdLoadedExecutable
    : public llvm::RTTIExtends<MpmdLoadedExecutable, LoadedExecutable> {
 public:
  // MPMD-specific interfaces.

  // Returns a mapping between atom program name and addressable devices.
  virtual absl::StatusOr<
      absl::flat_hash_map<std::string, absl::Span<Device* const>>>
  GetMpmdAddressableDevices() const = 0;

  // Returns a mapping between atom program name and CompiledMemoryStats.
  virtual absl::StatusOr<absl::flat_hash_map<std::string, CompiledMemoryStats>>
  GetMpmdCompiledMemoryStats() const = 0;

  // Returns a mapping between atom program name and a vector of HLO modules,
  // with an entry for each partition.
  virtual absl::StatusOr<
      absl::flat_hash_map<std::string, std::vector<std::shared_ptr<HloModule>>>>
  GetMpmdHloModules() const = 0;

  // Returns a mapping between atom program name and map of cost properties.
  virtual absl::StatusOr<absl::flat_hash_map<std::string, AttributeMap>>
  GetMpmdCostAnalysis() const = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MPMD_EXECUTABLE_H_
