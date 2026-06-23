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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_UTIL_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_UTIL_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

struct BdfComponents {
  uint64_t domain;
  uint64_t bus;
  uint64_t device;
  uint64_t function;
};

// Returns true if rocm_smi was successfully initialized.
bool InitRocmSmi();

// Parses a PCI bus ID string (e.g., "0000:41:00.0") into its BDF components.
// Returns std::nullopt on parse failure.
std::optional<BdfComponents> ParseBdf(absl::string_view pci_bus_id);

// Finds the rocm_smi device index that matches the given PCI bus ID.
// Returns std::nullopt if not found.
std::optional<uint32_t> FindDeviceIndex(const BdfComponents& target_bdf);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_SMI_UTIL_H_
