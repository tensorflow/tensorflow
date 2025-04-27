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

#include "xla/pjrt/raw_buffer.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

std::vector<RegisterRawBufferFactory::FactoryFuncT>& GetFactoryFuncs() {
  static auto* const funcs =
      new std::vector<RegisterRawBufferFactory::FactoryFuncT>;
  return *funcs;
}

PjRtFuture<> CommonPjRtRawBuffer::CopyRawHostToDevice(const void* src,
                                                      int64_t offset,
                                                      int64_t transfer_size) {
  auto event = CopyRawHostToDeviceAndReturnEvent(src, offset, transfer_size);
  if (!event.ok()) {
    return PjRtFuture<>(event.status());
  }
  return (*event)->GetReadyFuture();
}

absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>
PjRtRawBuffer::CreateRawAliasOfBuffer(PjRtBuffer* buffer) {
  for (auto* func : GetFactoryFuncs()) {
    auto res = (*func)(buffer);
    if (res.has_value()) {
      return *res;
    }
  }
  if (buffer == nullptr) {
    return absl::InvalidArgumentError("Cannot create view of null buffer.");
  }
  return absl::UnimplementedError(
      absl::StrCat("CreateRawAliasOfBuffer not implemented for: ",
                   buffer->client()->platform_version()));
}

RegisterRawBufferFactory::RegisterRawBufferFactory(
    RegisterRawBufferFactory::FactoryFuncT func) {
  GetFactoryFuncs().push_back(func);
}

}  // namespace xla
