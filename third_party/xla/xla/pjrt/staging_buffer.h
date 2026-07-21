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

#ifndef XLA_PJRT_STAGING_BUFFER_H_
#define XLA_PJRT_STAGING_BUFFER_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

// A buffer allocated for the purposes of transferring to and from the device.
// This is used as a staging area for linearize and delinearize operations.
// TODO(parkers): consider merging with PjRtChunk.
class PjRtStagingBuffer {
 public:
  virtual ~PjRtStagingBuffer();

  virtual absl::Span<uint8_t> data() = 0;

  virtual absl::Span<const uint8_t> const_data() const = 0;

  // Creates a PjRtStagingBuffer that wraps an already allocated memory span,
  // and runs `on_done` when the buffer is destroyed.
  static tsl::AsyncValueRef<PjRtStagingBuffer> Create(
      absl::Span<uint8_t> span, absl::AnyInvocable<void() &&> on_done);
};

}  // namespace xla

#endif  // XLA_PJRT_STAGING_BUFFER_H_
