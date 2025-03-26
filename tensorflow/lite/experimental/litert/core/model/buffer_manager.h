// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_BUFFER_MANAGER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_BUFFER_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::internal {

// Extra info about how the buffer is handled during load or serialization.
struct BufferContext {
  using Ref = std::reference_wrapper<BufferContext>;

  // Whether the buffer should be appended to the flatbuffer during
  // serialization.
  bool should_append = false;
};

// Container type for efficiently holding data buffers used by the model. These
// buffers may be owned or non-owned by the model. Uses id based indexing.
class BufferManager {
 public:
  using Ptr = std::unique_ptr<BufferManager>;

  // Unique identifier for a buffer. 0 is reserved for empty buffers.
  using BufferId = uint32_t;
  static constexpr BufferId kEmptyBufferId = 0;

  // Register a buffer that is not owned by the model. Caller must ensure the
  // buffer outlives the model.
  BufferId RegisterNonOwnedBuffer(
      BufferRef<uint8_t> buffer,
      std::optional<BufferContext> context = std::nullopt) {
    auto&& ctx = context.has_value() ? std::move(*context) : BufferContext{};
    buffers_.emplace_back(BufferWithContext(buffer, std::move(ctx)));
    return buffers_.size() - 1;
  }

  // Register a buffer that is owned by the model.
  BufferId RegisterOwnedBuffer(
      OwningBufferRef<uint8_t>&& buffer,
      std::optional<BufferContext> context = std::nullopt) {
    auto&& ctx = context.has_value() ? std::move(*context) : BufferContext{};
    buffers_.emplace_back(BufferWithContext(buffer, std::move(ctx)));
    return buffers_.size() - 1;
  }

  // Get a view of the buffer at the given id.
  Expected<BufferRef<uint8_t>> GetBuffer(BufferId id) {
    if (id >= buffers_.size()) {
      return Error(kLiteRtStatusErrorIndexOOB);
    }
    return GetView(buffers_[id].first);
  }

  // Get the context of the buffer at the given id.
  Expected<BufferContext::Ref> GetContext(BufferId id) {
    if (id >= buffers_.size()) {
      return Error(kLiteRtStatusErrorIndexOOB);
    }
    return std::ref(buffers_[id].second);
  }

  // Number of buffers. Ids will be 0 <-> num - 1.
  size_t NumBuffers() const { return buffers_.size(); }

  BufferManager() {
    // Zero is reserved for empty buffers.
    buffers_.emplace_back(
        BufferWithContext(BufferRef<uint8_t>(), BufferContext{}));
  }
  BufferManager(const BufferManager&) = delete;
  BufferManager& operator=(const BufferManager&) = delete;
  BufferManager(BufferManager&& other) = default;
  BufferManager& operator=(BufferManager&& other) = default;

 private:
  using BufferType = std::variant<BufferRef<uint8_t>, OwningBufferRef<uint8_t>>;
  using BufferWithContext = std::pair<BufferType, BufferContext>;

  static BufferRef<uint8_t> GetView(const BufferType& buffer) {
    BufferRef<uint8_t> res;
    std::visit([&res](auto&& arg) { res = arg; }, buffer);
    return res;
  }

  std::vector<BufferWithContext> buffers_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_BUFFER_MANAGER_H_
