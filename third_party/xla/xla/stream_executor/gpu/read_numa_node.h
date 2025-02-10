/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_READ_NUMA_NODE_H_
#define XLA_STREAM_EXECUTOR_GPU_READ_NUMA_NODE_H_

#include <optional>

#include "absl/strings/string_view.h"

namespace stream_executor::gpu {

// Attempts to read the NUMA node corresponding to the GPU device's PCI bus out
// of SysFS. Returns an empty optional if no value could be determined, returns
// tsl::port::kNUMANoAffinity if the kernel reports a negative value.
std::optional<int> ReadNumaNode(absl::string_view pci_bus_id,
                                int device_ordinal);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_READ_NUMA_NODE_H_
