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

#include "xla/service/gpu/runtime/memset_thunk.h"

#include "absl/status/status.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

absl::Status MemzeroThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::DeviceMemoryBase dest_data =
      params.buffer_allocations->GetDeviceAddress(dest_);
  return params.stream->MemZero(&dest_data, dest_data.size());
}

absl::Status Memset32BitValueThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceMemoryBase dest_data =
      params.buffer_allocations->GetDeviceAddress(dest_);
  return params.stream->Memset32(&dest_data, value_, dest_data.size());
}

}  // namespace gpu
}  // namespace xla
