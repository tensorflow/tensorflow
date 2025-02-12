/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/stream_finder.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

absl::StatusOr<Stream*> FindStream(Platform* platform, void* gpu_stream) {
  int number_devices = platform->VisibleDeviceCount();
  for (int i = 0; i < number_devices; ++i) {
    auto stream_executor = platform->FindExisting(i);
    if (!stream_executor.ok()) {
      continue;
    }
    Stream* found_stream = nullptr;
    if ((found_stream = (*stream_executor)->FindAllocatedStream(gpu_stream)) !=
        nullptr) {
      return found_stream;
    }
  }
  return absl::NotFoundError("Stream not found");
}

}  // namespace stream_executor
