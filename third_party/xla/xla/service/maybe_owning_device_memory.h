/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MAYBE_OWNING_DEVICE_MEMORY_H_
#define XLA_SERVICE_MAYBE_OWNING_DEVICE_MEMORY_H_

#include <optional>
#include <utility>

#include "absl/types/variant.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"

namespace xla {

// MaybeOwningDeviceMemory represents either an owned or unowned device memory.
// Like std::variant<se::OwningDeviceMemory, DeviceMemory>. When the object goes
// output of scope, it will free the underlying memory if it owns it.
class MaybeOwningDeviceMemory {
 public:
  MaybeOwningDeviceMemory() = default;
  explicit MaybeOwningDeviceMemory(tensorflow::se::OwningDeviceMemory owned)
      : mem_(std::move(owned)) {}
  explicit MaybeOwningDeviceMemory(tensorflow::se::DeviceMemoryBase unowned)
      : mem_(unowned) {}
  MaybeOwningDeviceMemory(MaybeOwningDeviceMemory&&) = default;
  ~MaybeOwningDeviceMemory() = default;

  MaybeOwningDeviceMemory& operator=(tensorflow::se::DeviceMemoryBase unowned) {
    mem_ = unowned;
    return *this;
  }

  MaybeOwningDeviceMemory& operator=(tensorflow::se::OwningDeviceMemory owned) {
    mem_ = std::move(owned);
    return *this;
  }

  MaybeOwningDeviceMemory& operator=(MaybeOwningDeviceMemory&&) = default;

  // Fetches the underlying DeviceMemoryBase from a MaybeOwningDeviceMemory. The
  // caller of this function is *not* responsible for freeing the memory.
  tensorflow::se::DeviceMemoryBase AsDeviceMemoryBase() const;

  // Release the tensorflow::se::OwningDeviceMemory without freeing it, and
  // moves the ownership of the memory buffer from the object to the caller.
  //
  // A nullopt is returned if the HasOwnership() == false;
  std::optional<tensorflow::se::OwningDeviceMemory> Release();

  // If the device memory is owned, returns a pointer to the internal
  // OwningDeviceMemory, otherwise nullptr is returned.
  const tensorflow::se::OwningDeviceMemory* AsOwningDeviceMemory() const;

  // Returns true if the device_memory has ownership over underlying memory.
  bool HasOwnership() const;

 private:
  std::variant<tensorflow::se::OwningDeviceMemory,
               tensorflow::se::DeviceMemoryBase>
      mem_;
};

}  // namespace xla

#endif  // XLA_SERVICE_MAYBE_OWNING_DEVICE_MEMORY_H_
