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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_consts.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"

namespace litert {

// Type for tensors with known dimensions. C++ equivalent to
// LiteRtRankedTensorType.
class RankedTensorType {
 public:
  RankedTensorType(enum ElementType element_type, class Layout&& layout)
      : element_type_(element_type), layout_(std::move(layout)) {}
  explicit RankedTensorType(const LiteRtRankedTensorType& type)
      : element_type_(static_cast<enum ElementType>(type.element_type)),
        layout_(type.layout) {}

  explicit operator LiteRtRankedTensorType() const {
    return LiteRtRankedTensorType{
        /*.element_type=*/static_cast<LiteRtElementType>(element_type_),
        /*layout=*/static_cast<LiteRtLayout>(layout_),
    };
  }

  bool operator==(const RankedTensorType& other) const {
    return ElementType() == other.ElementType() && Layout() == other.Layout();
  }

  enum ElementType ElementType() const { return element_type_; }

  const class Layout& Layout() const { return layout_; }

 private:
  enum ElementType element_type_;
  class Layout layout_;
};

// Tensor weights. C++ equivalent of LiteRtWeights.
class Weights : public internal::NonOwnedHandle<LiteRtWeights> {
 public:
  explicit Weights(LiteRtWeights weights)
      : internal::NonOwnedHandle<LiteRtWeights>(weights) {}

  absl::Span<const uint8_t> Bytes() const {
    size_t size;
    const void* addr;
    internal::AssertOk(LiteRtGetWeightsBytes, Get(), &addr, &size);
    return absl::MakeSpan(static_cast<const uint8_t*>(addr), size);
  }
};

// Tensor. C++ equivalent of LiteRtTensor.
class Tensor : public internal::NonOwnedHandle<LiteRtTensor> {
 public:
  explicit Tensor(LiteRtTensor tensor)
      : internal::NonOwnedHandle<LiteRtTensor>(tensor) {}

  enum ElementType ElementType() const {
    if (TypeId() == kLiteRtUnrankedTensorType) {
      return static_cast<enum ElementType>(UnrankedTensorType()->element_type);
    } else {
      return RankedTensorType()->ElementType();
    }
  }

  LiteRtTensorTypeId TypeId() const {
    LiteRtTensorTypeId type_id;
    internal::AssertOk(LiteRtGetTensorTypeId, Get(), &type_id);
    return type_id;
  }

  Expected<LiteRtUnrankedTensorType> UnrankedTensorType() const {
    if (TypeId() != kLiteRtUnrankedTensorType) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Not an unranked invalid tensor");
    }
    LiteRtUnrankedTensorType unranked_tensor_type;
    internal::AssertOk(LiteRtGetUnrankedTensorType, Get(),
                       &unranked_tensor_type);
    return unranked_tensor_type;
  }

  Expected<class RankedTensorType> RankedTensorType() const {
    if (TypeId() != kLiteRtRankedTensorType) {
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Not a ranked tensor type");
    }
    LiteRtRankedTensorType ranked_tensor_type;
    internal::AssertOk(LiteRtGetRankedTensorType, Get(), &ranked_tensor_type);
    return litert::RankedTensorType(ranked_tensor_type);
  }

  LiteRtQuantizationTypeId QTypeId() const {
    LiteRtQuantizationTypeId q_type_id;
    internal::AssertOk(LiteRtGetQuantizationTypeId, Get(), &q_type_id);
    return q_type_id;
  }

  bool HasQuantization() const { return QTypeId() != kLiteRtQuantizationNone; }

  LiteRtQuantizationPerTensor PerTensorQuantization() const {
    internal::AssertEq([&]() { return QTypeId(); },
                       kLiteRtQuantizationPerTensor);
    LiteRtQuantizationPerTensor per_tensor_quantization;
    internal::AssertOk(LiteRtGetPerTensorQuantization, Get(),
                       &per_tensor_quantization);
    return per_tensor_quantization;
  }

  LiteRtQuantizationPerChannel PerChannelQuantization() const {
    internal::AssertEq([&]() { return QTypeId(); },
                       kLiteRtQuantizationPerChannel);
    LiteRtQuantizationPerChannel per_channel_quantization;
    internal::AssertOk(LiteRtGetPerChannelQuantization, Get(),
                       &per_channel_quantization);
    return per_channel_quantization;
  }

  bool HasWeights() const {
    auto weights = Weights();
    return !weights.Bytes().empty();
  }

  class Weights Weights() const {
    LiteRtWeights weights;
    internal::AssertOk(LiteRtGetTensorWeights, Get(), &weights);
    return litert::Weights(weights);
  }

  absl::string_view Name() const {
    const char* name;
    internal::AssertOk(LiteRtGetTensorName, Get(), &name);
    return absl::string_view(name);
  }

  struct TensorUse;
  using TensorUses =
      absl::InlinedVector<TensorUse, kExpectedMaxNumOfTensorUses>;

  TensorUses Uses() const;

  template <typename T>
  Expected<absl::Span<const T>> WeightsData() const {
    auto ranked_tensor_type = RankedTensorType();
    if (!ranked_tensor_type) {
      return ranked_tensor_type.Error();
    }

    const enum ElementType ty = ranked_tensor_type->ElementType();
    if (ty != GetElementType<T>()) {
      return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
    }

    if (!HasWeights()) {
      return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    const absl::Span<const uint8_t> weights = Weights().Bytes();

    auto num_elements = ranked_tensor_type->Layout().NumElements();
    if (!num_elements.has_value()) {
      return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
    }
    auto byte_width = GetByteWidth(ty);
    if (!byte_width.has_value()) {
      return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
    }

    if (byte_width.value() * num_elements.value() != weights.size()) {
      return litert::Unexpected(kLiteRtStatusErrorInvalidArgument);
    }

    return absl::MakeConstSpan(reinterpret_cast<const T*>(weights.data()),
                               num_elements.value());
  }

  std::optional<LiteRtTensorDefiningOp> DefiningOp() const {
    bool has_defining_op;
    LiteRtTensorDefiningOp defining_op;
    internal::AssertOk(LiteRtGetTensorDefiningOp, Get(), &has_defining_op,
                       &defining_op);
    if (has_defining_op) {
      return defining_op;
    } else {
      return std::nullopt;
    }
  }

  bool IsSubgraphOutput() const;
  bool IsSubgraphInput() const;
  bool IsConstant() const;
};

using OpInputs = absl::InlinedVector<Tensor, kExpectedMaxNumOfOpInputs>;
using OpOutputs = absl::InlinedVector<Tensor, kExpectedMaxNumOfOpOutputs>;

// Operator. C++ equivalent of LiteRtOp.
class Op : public internal::NonOwnedHandle<LiteRtOp> {
 public:
  explicit Op(LiteRtOp op) : internal::NonOwnedHandle<LiteRtOp>(op) {}

  LiteRtOpCode Code() const {
    LiteRtOpCode opcode;
    internal::AssertOk(LiteRtGetOpCode, Get(), &opcode);
    return opcode;
  }

  OpInputs Inputs() const;
  OpOutputs Outputs() const;
};

struct Tensor::TensorUse {
  Op user;
  LiteRtParamIndex user_arg_ind;
};

using SubgraphInputs =
    absl::InlinedVector<Tensor, kExpectedMaxNumOfSubgraphInputs>;
using SubgraphOutputs =
    absl::InlinedVector<Tensor, kExpectedMaxNumOfSubgraphOutputs>;

// Model subgraph. C++ equivalent of LiteRtSubgraph.
class Subgraph : public internal::NonOwnedHandle<LiteRtSubgraph> {
 public:
  explicit Subgraph(LiteRtSubgraph subgraph)
      : internal::NonOwnedHandle<LiteRtSubgraph>(subgraph) {}

  SubgraphInputs Inputs() const;
  SubgraphOutputs Outputs() const;
  std::vector<Op> Ops() const;

  // Returns the input tensor with the given input signature name.
  Expected<Tensor> Input(absl::string_view name) const;

  // Returns the output tensor with the given output signature name.
  Expected<Tensor> Output(absl::string_view name) const;
};

// Model signature. C++ equivalent of LiteRtSignature.
class Signature : public internal::NonOwnedHandle<LiteRtSignature> {
 public:
  explicit Signature(LiteRtSignature signature)
      : internal::NonOwnedHandle<LiteRtSignature>(signature) {}

  absl::string_view Key() const {
    const char* key;
    internal::AssertOk(LiteRtGetSignatureKey, Get(), &key);
    return key;
  }

  LiteRtSubgraph Subgraph() const {
    LiteRtSubgraph subgraph;
    internal::AssertOk(LiteRtGetSignatureSubgraph, Get(), &subgraph);
    return subgraph;
  }

  std::vector<absl::string_view> InputNames() const {
    LiteRtParamIndex num_inputs;
    internal::AssertOk(LiteRtGetNumSignatureInputs, Get(), &num_inputs);
    std::vector<absl::string_view> input_names;
    input_names.reserve(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      const char* input_name;
      internal::AssertOk(LiteRtGetSignatureInputName, Get(), i, &input_name);
      input_names.push_back(input_name);
    }
    return input_names;
  }

  std::vector<absl::string_view> OutputNames() const {
    LiteRtParamIndex num_outputs;
    internal::AssertOk(LiteRtGetNumSignatureOutputs, Get(), &num_outputs);
    std::vector<absl::string_view> output_names;
    output_names.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      const char* output_name;
      internal::AssertOk(LiteRtGetSignatureOutputName, Get(), i, &output_name);
      output_names.push_back(output_name);
    }
    return output_names;
  }
};

// Model. C++ equivalent of LiteRtModel.
class Model : public internal::Handle<LiteRtModel, LiteRtDestroyModel> {
 public:
  Model() = default;

  static Model CreateFromOwnedHandle(LiteRtModel model) {
    return Model(model, /*owned=*/true);
  }

  static Model CreateFromNonOwnedHandle(LiteRtModel model) {
    return Model(model, /*owned=*/false);
  }

  static Expected<Model> CreateFromFile(const std::string& filename) {
    LiteRtModel model;
    if (auto status = LiteRtCreateModelFromFile(filename.c_str(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from file");
    }
    return CreateFromOwnedHandle(model);
  }

  static Expected<Model> CreateFromBuffer(BufferRef<uint8_t> buffer) {
    LiteRtModel model;
    if (auto status =
            LiteRtCreateModelFromBuffer(buffer.Data(), buffer.Size(), &model);
        status != kLiteRtStatusOk) {
      return Unexpected(status, "Failed to load model from buffer");
    }
    return CreateFromOwnedHandle(model);
  }

  Expected<absl::Span<const uint8_t>> Metadata(
      const std::string& metadata_key) const {
    const void* buffer;
    size_t buffer_size;
    if (LiteRtGetModelMetadata(Get(), metadata_key.data(), &buffer,
                               &buffer_size) != kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Metadata key not found");
    }
    return absl::MakeSpan(static_cast<const uint8_t*>(buffer), buffer_size);
  }

  Expected<class Subgraph> MainSubgraph() {
    LiteRtParamIndex main_subgraph_index;
    internal::AssertOk(LiteRtGetMainModelSubgraphIndex, Get(),
                       &main_subgraph_index);
    return this->Subgraph(main_subgraph_index);
  }

  size_t NumSubgraphs() const {
    LiteRtParamIndex num_subgraphs;
    internal::AssertOk(LiteRtGetNumModelSubgraphs, Get(), &num_subgraphs);
    return num_subgraphs;
  }

  Expected<class Subgraph> Subgraph(size_t subgraph_index) {
    LiteRtSubgraph subgraph;
    if (LiteRtGetModelSubgraph(Get(), subgraph_index, &subgraph) !=
        kLiteRtStatusOk) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Subgraph not found");
    }
    return litert::Subgraph(subgraph);
  }

  Expected<class Subgraph> Subgraph(absl::string_view signature_key) const {
    auto signature = FindSignature(signature_key);
    if (!signature) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
    }
    return litert::Subgraph(signature->Subgraph());
  }

  size_t GetNumSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    return num_signatures;
  }

  // Returns the list of signatures defined in the model.
  Expected<std::vector<class Signature>> GetSignatures() const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    std::vector<class Signature> signatures;
    signatures.reserve(num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      Signature signature(lite_rt_signature);
      signatures.push_back(std::move(signature));
    }
    return std::move(signatures);
  }

  // Returns the signature at the given index.
  Expected<class Signature> GetSignature(size_t signature_index) const {
    LiteRtSignature lite_rt_signature;
    internal::AssertOk(LiteRtGetModelSignature, Get(), signature_index,
                       &lite_rt_signature);
    return Signature(lite_rt_signature);
  }

  // Returns the signature index for the given signature key.
  Expected<size_t> GetSignatureIndex(absl::string_view signature_key) const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      const char* key_cstr;
      internal::AssertOk(LiteRtGetSignatureKey, lite_rt_signature, &key_cstr);
      if (absl::string_view(key_cstr) == signature_key) {
        return i;
      }
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  // Returns the Signature object for the given signature key.
  Expected<class Signature> FindSignature(
      absl::string_view signature_key) const {
    LiteRtParamIndex num_signatures;
    internal::AssertOk(LiteRtGetNumModelSignatures, Get(), &num_signatures);
    for (int i = 0; i < num_signatures; ++i) {
      LiteRtSignature lite_rt_signature;
      internal::AssertOk(LiteRtGetModelSignature, Get(), i, &lite_rt_signature);
      const char* key_cstr;
      internal::AssertOk(LiteRtGetSignatureKey, lite_rt_signature, &key_cstr);
      if (absl::string_view(key_cstr) == signature_key) {
        return Signature(lite_rt_signature);
      }
    }
    return Unexpected(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  static absl::string_view DefaultSignatureKey() {
    const char* key;
    internal::AssertOk(LiteRtGetDefaultSignatureKey, &key);
    return key;
  }

 private:
  // Parameter `owned` indicates if the created TensorBuffer object should take
  // ownership of the provided `tensor_buffer` handle.
  Model(LiteRtModel model, bool owned)
      : internal::Handle<LiteRtModel, LiteRtDestroyModel>(model, owned) {}
};

struct SerializationOptions {
  static LiteRtModelSerializationOptions Defaults() {
    return LiteRtModelSerializationOptions{};
  }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_H_
