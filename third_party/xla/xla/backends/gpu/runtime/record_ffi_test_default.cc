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

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

absl::StatusOr<void*> LoadNativeFunctionPtr(stream_executor::StreamExecutor*,
                                            absl::Span<const uint8_t>,
                                            absl::string_view) {
  return absl::UnimplementedError(
      "Native function pointer loading is not supported on this platform.");
}

}  // namespace xla::gpu
