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

#include "xla/backends/gpu/collectives/gpu_collectives.h"

#include <cstdint>

namespace xla::gpu {

CollectiveStreamId GetCollectiveStreamId(bool is_async,
                                         AsyncStreamKind stream_kind) {
  // TODO(ezhulenev): This implementation does not look correct as stream IDs
  // are not really unique. Figure out if it's the case and fix either the code
  // or the documentation.
  int64_t stream_id = static_cast<int64_t>(stream_kind);
  return CollectiveStreamId(is_async ? stream_id + 1 : 0);
}

}  // namespace xla::gpu
