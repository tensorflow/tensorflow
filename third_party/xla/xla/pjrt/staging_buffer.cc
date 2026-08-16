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

#include "xla/pjrt/staging_buffer.h"

#include <cstdint>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

PjRtStagingBuffer::~PjRtStagingBuffer() = default;

namespace {

class CallbackStagingBuffer : public PjRtStagingBuffer {
 public:
  CallbackStagingBuffer(absl::Span<uint8_t> data,
                        absl::AnyInvocable<void() &&> on_done)
      : data_(data), on_done_(std::move(on_done)) {}

  ~CallbackStagingBuffer() override {
    if (on_done_) {
      std::move(on_done_)();
    }
  }

  absl::Span<uint8_t> data() override { return data_; }
  absl::Span<const uint8_t> const_data() const override { return data_; }

 private:
  absl::Span<uint8_t> data_;
  absl::AnyInvocable<void() &&> on_done_;
};

}  // namespace

/*static*/ tsl::AsyncValueRef<PjRtStagingBuffer> PjRtStagingBuffer::Create(
    absl::Span<uint8_t> span, absl::AnyInvocable<void() &&> on_done) {
  auto staging_buffer = tsl::MakeAvailableAsyncValueRef<CallbackStagingBuffer>(
      span, std::move(on_done));
  return tsl::AsyncValueRef<PjRtStagingBuffer>(std::move(staging_buffer));
}

}  // namespace xla
