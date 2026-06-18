/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/client.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

char Client::ID = 0;

absl::StatusOr<CustomLayoutRef> Client::GetDefaultLayout(
    DType dtype, absl::Span<const int64_t> shard_dims, Device* device,
    xla::ifrt::MemoryKind memory_kind) const {
  return GetDefaultLayout(dtype, Shape(shard_dims),
                          SingleDeviceSharding::Create(device, memory_kind));
}

}  // namespace ifrt
}  // namespace xla
