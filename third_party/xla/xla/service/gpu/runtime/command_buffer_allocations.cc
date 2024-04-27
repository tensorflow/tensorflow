/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/command_buffer_allocations.h"

#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/service/buffer_assignment.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::gpu {

absl::StatusOr<se::DeviceMemoryBase> CommandBufferAllocations::GetDeviceAddress(
    BufferAllocation::Index index) const {
  auto base = allocs_.find(index);
  if (base == allocs_.end()) {
    return absl::InternalError(absl::StrCat("Command buffer allocation #",
                                            index, " was not allocated"));
  }
  return allocs_.at(index);
}

absl::Status CommandBufferAllocations::AddAllocation(
    BufferAllocation::Index index, se::DeviceMemoryBase memory) {
  VLOG(2) << "Add comand buffer allocation: index=" << index
          << "; ptr=" << memory.opaque();

  auto emplaced = allocs_.try_emplace(index, std::move(memory));
  if (emplaced.second == false) {
    return absl::InternalError(absl::StrCat("Command buffer allocation #",
                                            index, " was already allocated"));
  }
  return absl::OkStatus();
}

absl::Status CommandBufferAllocations::EraseAllocation(
    BufferAllocation::Index index) {
  VLOG(2) << "Erase comand buffer allocation: index=" << index;

  if (allocs_.erase(index) == 0) {
    return absl::InternalError(absl::StrCat("Command buffer allocation #",
                                            index, " was not allocated"));
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu
