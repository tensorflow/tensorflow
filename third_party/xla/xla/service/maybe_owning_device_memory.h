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

#ifndef XLA_SERVICE_MAYBE_OWNING_DEVICE_MEMORY_H_
#define XLA_SERVICE_MAYBE_OWNING_DEVICE_MEMORY_H_

#include "absl/base/macros.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/stream_executor/device_address.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_address_allocator.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_memory.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_memory_allocator.h"  // IWYU pragma: keep

namespace xla {

using MaybeOwningDeviceMemory ABSL_DEPRECATE_AND_INLINE() =
    MaybeOwningDeviceAddress;

}

#endif  // XLA_SERVICE_MAYBE_OWNING_DEVICE_MEMORY_H_
