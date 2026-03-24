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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_STATS_H_
#define XLA_STREAM_EXECUTOR_KERNEL_STATS_H_

#include <string>

#include "absl/container/flat_hash_map.h"

// Stats about a single kernel.
struct KernelStats {
  // The number of spilled register bytes in the kernel for stores.
  int store_bytes_spilled = 0;
  // The number of spilled register bytes in the kernel for loads.
  int load_bytes_spilled = 0;
};

// Map from a function/kernel name to its kernel stats.
using ModuleStats = absl::flat_hash_map<std::string, KernelStats>;

#endif  // XLA_STREAM_EXECUTOR_KERNEL_STATS_H_
