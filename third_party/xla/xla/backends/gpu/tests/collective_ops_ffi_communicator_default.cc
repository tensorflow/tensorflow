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

#include <cstdint>

#include "absl/status/status.h"
#include "xla/ffi/api/collectives_c_api.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace xla::gpu {

absl::Status CommunicatorAllReduceU32(stream_executor::Stream*,
                                      XLA_FFI_Communicator*, const void*, void*,
                                      int64_t) {
  return absl::UnimplementedError(
      "Communicator all-reduce is not implemented for this platform");
}

}  // namespace xla::gpu
