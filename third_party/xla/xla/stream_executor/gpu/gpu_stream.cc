/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/gpu_stream.h"

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {
namespace gpu {

GpuStreamHandle AsGpuStreamValue(Stream* stream) {
  DCHECK(stream != nullptr);
  return absl::bit_cast<GpuStreamHandle>(
      stream->platform_specific_handle().stream);
}

}  // namespace gpu
}  // namespace stream_executor
