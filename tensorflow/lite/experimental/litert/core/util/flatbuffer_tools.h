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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_FLATBUFFER_TOOLS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_FLATBUFFER_TOOLS_H_

#include <cstdint>
#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

using TflTensor = ::tflite::TensorT;
using TflOp = ::tflite::OperatorT;
using TflBuffer = ::tflite::BufferT;
using TflModel = ::tflite::ModelT;
using TflOpCode = ::tflite::BuiltinOperator;

using TflBufferPtr = std::unique_ptr<TflBuffer>;
using TflModelPtr = std::unique_ptr<TflModel>;

// Flatbuffer bytes util.

// Convenience method to get string view from native flatbuffer chars.
absl::string_view FbBufToStr(const uint8_t* fb_data, size_t size);

// Span version.
absl::string_view FbBufToStr(absl::Span<const uint8_t> fb_buf);

// Convenience method to get mutable signed char span from native flatbuffer
// chars.
absl::Span<char> FbBufToStr(uint8_t* fb_data, size_t size);

// Span to span version.
absl::Span<char> FbBufToStr(absl::Span<uint8_t> fb_buf);

// Flatbuffer verifiers.

// Verifies given serialized flatbuffer
bool VerifyFlatbuffer(const uint8_t* buf, size_t buf_size);

// Override of above with view input.
bool VerifyFlatbuffer(absl::Span<const uint8_t> buf);

// Flatbuffer model api helpers.

// Get the metadata buffer under given key if it exists.
Expected<BufferRef<uint8_t>> GetMetadata(absl::string_view key,
                                         const TflModel& model);

// Get the metadata buffer under given key if it exists that can be written to.
Expected<MutableBufferRef<uint8_t>> GetMutableMetadata(absl::string_view key,
                                                       TflModel& model);

// Push the given metadata to the given key if the key does not already exist.
LiteRtStatus PushMetadata(absl::string_view key, TflModel& model,
                          BufferRef<uint8_t> metadata);

// Get the buffer object at the given index if it exists.
Expected<BufferRef<uint8_t>> GetTflBuffer(const TflModel& tfl_model,
                                          uint32_t buffer_ind);

// Get the buffer object at the given index if it exists that can be written to.
Expected<MutableBufferRef<uint8_t>> GetMutableTflBuffer(TflModel& tfl_model,
                                                        uint32_t buffer_ind);

// Move and take ownership of the buffer object at given index if it exists.
Expected<TflBufferPtr> TakeBuffer(TflModel& tfl_model, uint32_t buffer_ind);

// Add a new buffer to the tflite model, returning its index.
Expected<uint32_t> PushTflBuffer(TflModel& tfl_model,
                                 BufferRef<uint8_t> buffer);

// Get the op code from the model at the given index if it exists.
Expected<TflOpCode> GetTflOpCode(const TflModel& tfl_model,
                                 uint32_t op_code_ind);

// Make a tfl allocation from buffer.
::tflite::Allocation::Ptr MakeAllocation(BufferRef<uint8_t> buf);

// Wrapper around a tflite model buffer.
class FlatbufferWrapper {
 public:
  using Ptr = std::unique_ptr<FlatbufferWrapper>;

  // Load flatbuffer from file.
  static Expected<Ptr> CreateFromTflFile(absl::string_view path);

  // Load flatbuffer from allocated buffer that will be copied.
  static Expected<Ptr> CreateFromBuffer(BufferRef<uint8_t> buffer);

  // Load flatbuffer from allocated buffer and take ownership.
  static Expected<Ptr> CreateFromBuffer(OwningBufferRef<uint8_t>&& buffer);

  // Underlying buffer.
  BufferRef<uint8_t> Buf() const {
    return BufferRef<uint8_t>(alloc_->base(), alloc_->bytes());
  }

  // Underlying model object.
  const ::tflite::FlatBufferModel& FlatbufferModel() const {
    return *fb_model_;
  }

  // Unpacked version of underlying model object.
  const TflModel& UnpackedModel() const { return *unpacked_; }
  TflModel& UnpackedModel() { return *unpacked_; }

 private:
  FlatbufferWrapper(::tflite::FlatBufferModel::Ptr fb_model,
                    ::tflite::Allocation::Ptr alloc,
                    OwningBufferRef<uint8_t>&& model_buf)
      : fb_model_(std::move(fb_model)),
        alloc_(std::move(alloc)),
        model_buf_(std::forward<OwningBufferRef<uint8_t>>(model_buf)),
        unpacked_(TflModelPtr(fb_model_->GetModel()->UnPack())) {}

  ::tflite::FlatBufferModel::Ptr fb_model_;
  ::tflite::Allocation::Ptr alloc_;
  OwningBufferRef<uint8_t> model_buf_;
  TflModelPtr unpacked_;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_FLATBUFFER_TOOLS_H_
