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

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_consts.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

// Flatbuffer IR

using TflTensor = ::tflite::TensorT;
using TflOp = ::tflite::OperatorT;
using TflBuffer = ::tflite::BufferT;
using TflSubgraph = ::tflite::SubGraphT;
using TflModel = ::tflite::ModelT;
using TflOpCodeEnum = ::tflite::BuiltinOperator;
using TflOpCode = ::tflite::OperatorCodeT;
using TflQuantization = ::tflite::QuantizationParametersT;
using TflElementType = ::tflite::TensorType;
using TflOptions = ::tflite::BuiltinOptionsUnion;
using TflSignature = ::tflite::SignatureDefT;
using TflMetadata = ::tflite::MetadataT;

using TflBufferPtr = std::unique_ptr<TflBuffer>;
using TflModelPtr = std::unique_ptr<TflModel>;
using TflQuantizationPtr = std::unique_ptr<TflQuantization>;
using TflOpCodePtr = std::unique_ptr<TflOpCode>;
using TflSubgraphPtr = std::unique_ptr<TflSubgraph>;
using TflTensorPtr = std::unique_ptr<TflTensor>;
using TflOpPtr = std::unique_ptr<TflOp>;
using TflSignaturePtr = std::unique_ptr<TflSignature>;
using TflMetadataPtr = std::unique_ptr<TflMetadata>;

// Code and verion.
using TflOpCodeDetail = std::pair<TflOpCodeEnum, int32_t>;

// Zero-point, scale.
using TflPerTensorQParams = std::pair<int64_t, float>;

// Quantized dim, num channels, zero-points, scales.
using TflPerChannelQParams =
    std::tuple<int32_t, size_t, std::vector<int64_t>, std::vector<float>>;

// Mirror of all the tensor type related fields in flatbuffer tensor definition.
struct TflShapeInfo {
  // Fixed or dynamic rank.
  bool has_rank;

  // Basic shape, all elements are non-negative (even if this is a dynamic
  // shape).
  absl::InlinedVector<int32_t, kExpectedMaxTensorRank> shape;

  // Dynamic dyn info. If this is not empty, then its length is equal to shape.
  // If i is a dyn dim, then shape[i] == 1 and shape_signature[i] < 0. Otherwise
  // shape_signature[i] == shape[i].
  absl::InlinedVector<int32_t, kExpectedMaxTensorRank> shape_signature;

  // Convert from a single dims array. Will detect if array is static/dynamic
  // and populate fields accordingly.
  explicit TflShapeInfo(absl::Span<const int32_t> shape_data) : has_rank(true) {
    bool is_dyn = false;
    shape.reserve(shape_data.size());
    shape_signature.reserve(shape_data.size());
    for (auto d : shape_data) {
      if (d >= 0) {
        shape.push_back(d);
        shape_signature.push_back(d);
      } else {
        is_dyn = true;
        shape.push_back(1);
        shape_signature.push_back(-1);
      }
    }
    if (!is_dyn) {
      shape_signature.clear();
    }
  }

  // Convert from tensor.
  explicit TflShapeInfo(const TflTensor& tfl_tensor)
      : has_rank(tfl_tensor.has_rank),
        shape(tfl_tensor.shape.begin(), tfl_tensor.shape.end()),
        shape_signature(tfl_tensor.shape_signature.begin(),
                        tfl_tensor.shape_signature.end()) {}
};

using TflTensorType = std::pair<TflElementType, TflShapeInfo>;

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

// TFL flatbuffer IR helpers.

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

// Get a non-owning view of tfl buffer if it exists.
Expected<const TflBuffer*> GetBuffer(const TflModel& tfl_model,
                                     uint32_t buffer_ind);

// Move and take ownership of the buffer object at given index if it exists.
Expected<TflBufferPtr> TakeBuffer(TflModel& tfl_model, uint32_t buffer_ind);

// Add a new buffer to the tflite model, returning its index.
Expected<uint32_t> PushTflBuffer(TflModel& tfl_model,
                                 BufferRef<uint8_t> buffer);

// Make a tflite buffer from data.
template <class T>
TflBufferPtr MakeTflBuffer(std::initializer_list<T> data) {
  auto res = std::make_unique<TflBuffer>();
  const auto byte_size = data.size() * sizeof(T);
  res->data.resize(byte_size);
  for (auto it = data.begin(); it != data.end(); ++it) {
    auto* write_to =
        reinterpret_cast<T*>(res->data.data()) + (it - data.begin());
    *write_to = *it;
  }
  res->size = res->data.size();
  res->offset = 0;
  return res;
}

// Get the op code from the model at the given index if it exists.
Expected<TflOpCodeEnum> GetTflOpCode(const TflModel& tfl_model,
                                     uint32_t op_code_ind);

// Is tensor fixed rank, with possible dynamic dims.
bool IsRankedTensorType(const TflShapeInfo& tfl_shape);

// Is ranked tensor type with static shape.
bool IsStaticTensorType(const TflShapeInfo& tfl_shape);

// Get static shape info if given is indeed a static shape.
Expected<absl::Span<const int32_t>> AsStaticShape(
    const TflShapeInfo& tfl_shape);

// Get ranked dynamic shape info if given is indeed a ranked. Still works with
// static shapes.
Expected<absl::Span<const int32_t>> AsDynamicShape(
    const TflShapeInfo& tfl_shape);

// Is the tensor quantized.
bool IsQuantized(const TflQuantization* tfl_quantization);

// Is the tensor per-tensor quantized.
bool IsPerTensorQuantized(const TflQuantization* tfl_quantization);

// Is the tensor per-channel quantized.
bool IsPerChannelQuantized(const TflQuantization* tfl_quantization);

// Is the tensor block-wise quantized.
bool IsBlockWiseQuantized(const TflQuantization* tfl_quantization);

// Does tensor have custom quantization.
bool IsCustomQuantized(const TflQuantization* tfl_quantization);

// Get the per-tensor tensor q-params if given tensor has them.
Expected<TflPerTensorQParams> AsPerTensorQparams(
    const TflQuantization* tfl_quantization);

// Get the per-channel tensor q-params if given tensor has them.
Expected<TflPerChannelQParams> AsPerChannelQparams(
    const TflQuantization* tfl_quantization);

// Flatbuffer management helpers.

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

  // Unpack the contained flatbuffer.
  TflModelPtr Unpack() const {
    return TflModelPtr(fb_model_->GetModel()->UnPack());
  }

 private:
  FlatbufferWrapper(::tflite::FlatBufferModel::Ptr fb_model,
                    ::tflite::Allocation::Ptr alloc,
                    OwningBufferRef<uint8_t>&& model_buf)
      : fb_model_(std::move(fb_model)),
        alloc_(std::move(alloc)),
        model_buf_(std::forward<OwningBufferRef<uint8_t>>(model_buf)) {}

  ::tflite::FlatBufferModel::Ptr fb_model_;
  ::tflite::Allocation::Ptr alloc_;
  OwningBufferRef<uint8_t> model_buf_;
};

// Re-serialize the unpacked model from flatbuffer wrapper.
OwningBufferRef<uint8_t> SerializeFlatbuffer(
    const FlatbufferWrapper& flatbuffer);
OwningBufferRef<uint8_t> SerializeFlatbuffer(const TflModel& tfl_model);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_UTIL_FLATBUFFER_TOOLS_H_
