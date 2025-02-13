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
#include "xla/python/transfer/test_pattern.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/test.h"

namespace aux::tests {

std::vector<int32_t> CreateTestPattern(size_t offset, size_t length) {
  std::vector<int32_t> out(length);
  for (int32_t i = 0; i < length; ++i) {
    out[i] = i + offset;
  }
  return out;
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> CopyTestPatternToDevice(
    xla::ifrt::Client* client, xla::ifrt::Device* dest_device,
    const std::vector<int32_t>& pattern) {
  return client->MakeArrayFromHostBuffer(
      pattern.data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({static_cast<int64_t>(pattern.size())}), std::nullopt,
      xla::ifrt::SingleDeviceSharding::Create(dest_device,
                                              xla::ifrt::MemoryKind()),
      xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall, [] {});
}

}  // namespace aux::tests
