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
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_handle.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"

namespace litert {

// Data type of tensor elements. C++ equivalent to LiteRtElementType.
enum class ElementType {
  None = kLiteRtElementTypeNone,
  Bool = kLiteRtElementTypeBool,
  Int4 = kLiteRtElementTypeInt4,
  Int8 = kLiteRtElementTypeInt8,
  Int16 = kLiteRtElementTypeInt16,
  Int32 = kLiteRtElementTypeInt32,
  Int64 = kLiteRtElementTypeInt64,
  UInt8 = kLiteRtElementTypeUInt8,
  UInt16 = kLiteRtElementTypeUInt16,
  UInt32 = kLiteRtElementTypeUInt32,
  UInt64 = kLiteRtElementTypeUInt64,
  Float16 = kLiteRtElementTypeFloat16,
  BFloat16 = kLiteRtElementTypeBFloat16,
  Float32 = kLiteRtElementTypeFloat32,
  Float64 = kLiteRtElementTypeFloat64,
  Complex64 = kLiteRtElementTypeComplex64,
  Complex128 = kLiteRtElementTypeComplex128,
  TfResource = kLiteRtElementTypeTfResource,
  TfString = kLiteRtElementTypeTfString,
  TfVariant = kLiteRtElementTypeTfVariant,
};

// Tensor layout. C++ equivalent to LiteRtLayout.
class Layout {
 public:
  explicit Layout(std::vector<int32_t>&& dimensions,
                  std::vector<uint32_t>&& strides = std::vector<uint32_t>())
      : dimensions_(std::move(dimensions)), strides_(std::move(strides)) {}

  explicit Layout(const LiteRtLayout& layout)
      : dimensions_(layout.dimensions, layout.dimensions + layout.rank) {
    if (layout.strides) {
      strides_.reserve(layout.rank);
      std::copy(layout.strides, layout.strides + layout.rank,
                std::back_inserter(strides_));
    }
  }

  explicit operator LiteRtLayout() const {
    return LiteRtLayout{
        /*.rank=*/Rank(),
        /*.dimensions=*/dimensions_.data(),
        /*.strides=*/(HasStrides() ? strides_.data() : nullptr),
    };
  }

  bool operator==(const Layout& other) const {
    return dimensions_ == other.dimensions_ && strides_ == other.strides_;
  }

  uint32_t Rank() const { return dimensions_.size(); }

  absl::Span<const int32_t> Dimensions() const {
    return absl::MakeSpan(dimensions_.data(), dimensions_.size());
  }

  bool HasStrides() const { return !strides_.empty(); }

  absl::Span<const uint32_t> Strides() const {
    const uint32_t* data = HasStrides() ? strides_.data() : nullptr;
    auto size = HasStrides() ? Rank() : 0;
    return absl::MakeSpan(data, size);
  }

 private:
  std::vector<int32_t> dimensions_;
  std::vector<uint32_t> strides_;
};

// Type for tensors with known dimensions. C++ equivalent to
// LiteRtRankedTensorType.
class RankedTensorType {
 public:
  RankedTensorType(ElementType element_type, Layout&& layout)
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

  ElementType ElementType() const { return element_type_; }

  const Layout& Layout() const { return layout_; }

 private:
  enum ElementType element_type_;
  class Layout layout_;
};

// Tensor weights. C++ equivalent of LiteRtWeights.
class Weights : public internal::NonOwnedHandle<LiteRtWeights> {
 public:
  Weights() = default;
  explicit Weights(LiteRtWeights weights)
      : internal::NonOwnedHandle<LiteRtWeights>(weights) {}

  absl::Span<const uint8_t> Bytes() const {
    size_t size;
    const void* addr;
    litert::internal::AssertGet(LiteRtGetWeightsBytes, Get(), &addr, &size);
    return absl::MakeSpan(static_cast<const uint8_t*>(addr), size);
  }
};

// Tensor. C++ equivalent of LiteRtTensor.
class Tensor : public internal::NonOwnedHandle<LiteRtTensor> {
 public:
  Tensor() = default;
  explicit Tensor(LiteRtTensor tensor)
      : internal::NonOwnedHandle<LiteRtTensor>(tensor) {}

  LiteRtTensorTypeId TypeId() const {
    LiteRtTensorTypeId type_id;
    litert::internal::AssertGet(LiteRtGetTensorTypeId, Get(), &type_id);
    return type_id;
  }

  LiteRtUnrankedTensorType UnrankedTensorType() const {
    LiteRtUnrankedTensorType unranked_tensor_type;
    litert::internal::AssertGet(LiteRtGetUnrankedTensorType, Get(),
                                &unranked_tensor_type);
    return unranked_tensor_type;
  }

  RankedTensorType RankedTensorType() const {
    LiteRtRankedTensorType ranked_tensor_type;
    litert::internal::AssertGet(LiteRtGetRankedTensorType, Get(),
                                &ranked_tensor_type);
    return litert::RankedTensorType(ranked_tensor_type);
  }

  bool HasWeights() const {
    auto weights = Weights();
    return !weights.Bytes().empty();
  }

  Weights Weights() const {
    LiteRtWeights weights;
    litert::internal::AssertGet(LiteRtGetTensorWeights, Get(), &weights);
    return litert::Weights(weights);
  }

  void Uses(absl::Span<LiteRtOp>& uses,
            absl::Span<LiteRtParamIndex>& user_arg_indices) const {
    LiteRtParamIndex num_uses;
    LiteRtOpArray users;
    LiteRtParamIndex* user_arg_inds;
    litert::internal::AssertGet(LiteRtGetTensorUses, Get(), &num_uses, &users,
                                &user_arg_inds);
    uses = absl::MakeSpan(users, num_uses);
    user_arg_indices = absl::MakeSpan(user_arg_inds, num_uses);
  }

  std::optional<LiteRtTensorDefiningOp> DefiningOp() const {
    bool has_defining_op;
    LiteRtTensorDefiningOp defining_op;
    litert::internal::AssertGet(LiteRtGetTensorDefiningOp, Get(),
                                &has_defining_op, &defining_op);
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

// Operator. C++ equivalent of LiteRtOp.
class Op : public internal::NonOwnedHandle<LiteRtOp> {
 public:
  Op() = default;
  explicit Op(LiteRtOp op) : internal::NonOwnedHandle<LiteRtOp>(op) {}

  LiteRtOpCode Code() const {
    LiteRtOpCode opcode;
    litert::internal::AssertGet(LiteRtGetOpCode, Get(), &opcode);
    return opcode;
  }

  absl::Span<LiteRtTensor> Inputs() const {
    LiteRtParamIndex num_inputs;
    LiteRtTensorArray inputs;
    litert::internal::AssertGet(LiteRtGetOpInputs, Get(), &num_inputs, &inputs);
    return absl::MakeSpan(inputs, num_inputs);
  }

  absl::Span<LiteRtTensor> Outputs() const {
    LiteRtParamIndex num_outputs;
    LiteRtTensorArray outputs;
    litert::internal::AssertGet(LiteRtGetOpOutputs, Get(), &num_outputs,
                                &outputs);
    return absl::MakeSpan(outputs, num_outputs);
  }
};

// Model subgraph. C++ equivalent of LiteRtSubgraph.
class Subgraph : public internal::NonOwnedHandle<LiteRtSubgraph> {
 public:
  Subgraph() = default;
  explicit Subgraph(LiteRtSubgraph subgraph)
      : internal::NonOwnedHandle<LiteRtSubgraph>(subgraph) {}

  absl::Span<LiteRtTensor> Inputs() const {
    LiteRtParamIndex num_inputs;
    LiteRtTensorArray inputs;
    litert::internal::AssertGet(LiteRtGetSubgraphInputs, Get(), &num_inputs,
                                &inputs);
    return absl::MakeSpan(inputs, num_inputs);
  }

  absl::Span<LiteRtTensor> Outputs() const {
    LiteRtParamIndex num_outputs;
    LiteRtTensorArray outputs;
    litert::internal::AssertGet(LiteRtGetSubgraphOutputs, Get(), &num_outputs,
                                &outputs);
    return absl::MakeSpan(outputs, num_outputs);
  }

  absl::Span<LiteRtOp> Ops() const {
    LiteRtParamIndex num_ops;
    LiteRtOpArray ops;
    litert::internal::AssertGet(LiteRtGetSubgraphOps, Get(), &num_ops, &ops);
    return absl::MakeSpan(ops, num_ops);
  }
};

// Model. C++ equivalent of LiteRtModel.
class Model : public internal::Handle<LiteRtModel, LiteRtModelDestroy> {
 public:
  Model() = default;

  static Model CreateFromOwnedHandle(LiteRtModel model) {
    return Model(model, /*owned=*/true);
  }

  static Model CreateFromNonOwnedHandle(LiteRtModel model) {
    return Model(model, /*owned=*/false);
  }

  absl::StatusOr<absl::Span<const uint8_t>> Metadata(
      const std::string& metadata_key) const {
    const void* buffer;
    size_t buffer_size;
    if (LiteRtGetModelMetadata(Get(), metadata_key.data(), &buffer,
                               &buffer_size) != kLiteRtStatusOk) {
      return absl::NotFoundError("Metadata key not found");
    }
    return absl::MakeSpan(static_cast<const uint8_t*>(buffer), buffer_size);
  }

  absl::StatusOr<Subgraph> MainSubgraph() {
    LiteRtParamIndex main_subgraph_index;
    litert::internal::AssertGet(LiteRtGetMainModelSubgraphIndex, Get(),
                                &main_subgraph_index);
    return this->Subgraph(main_subgraph_index);
  }

  size_t NumSubgraphs() const {
    LiteRtParamIndex num_subgraphs;
    litert::internal::AssertGet(LiteRtGetNumModelSubgraphs, Get(),
                                &num_subgraphs);
    return num_subgraphs;
  }

  absl::StatusOr<Subgraph> Subgraph(size_t subgraph_index) {
    LiteRtSubgraph subgraph;
    if (LiteRtGetModelSubgraph(Get(), subgraph_index, &subgraph) !=
        kLiteRtStatusOk) {
      return absl::NotFoundError("Subgraph not found");
    }
    return litert::Subgraph(subgraph);
  }

 private:
  // Parameter `owned` indicates if the created TensorBuffer object should take
  // ownership of the provided `tensor_buffer` handle.
  Model(LiteRtModel model, bool owned)
      : internal::Handle<LiteRtModel, LiteRtModelDestroy>(model, owned) {}
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_H_
