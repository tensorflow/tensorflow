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

#ifndef XLA_PYTHON_IFRT_IR_UTILS_H_
#define XLA_PYTHON_IFRT_IR_UTILS_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/memory.h"

namespace xla {
namespace ifrt {

// Returns a DeviceList for the given device ids.
absl::StatusOr<DeviceListRef> LookUpDevices(Client* client,
                                            absl::Span<const DeviceId> ids);

// Converts an XLA computation to an `xla::ifrt::HloProgram`, and applies input
// donation and memory kind attributes to the input and output. The generated
// MLIR module will have flattened (non XLA tuple) parameters and results.
absl::StatusOr<std::unique_ptr<HloProgram>> XlaComputationToHloProgram(
    const xla::XlaComputation& xla_computation,
    absl::Span<const int64_t> donated_input_indices,
    absl::Span<const MemoryKind> arg_memory_kinds,
    absl::Span<const MemoryKind> result_memory_kinds);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_UTILS_H_
