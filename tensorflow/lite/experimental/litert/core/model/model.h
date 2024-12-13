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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"  // IWYU pragma: export
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/model/ir_allocator.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

////////////////////////////////////////////////////////////////////////////////
// Internal LiteRtIR
//
// These are the backing definitions for the opaque types in the c api
// (c/litert_model.h).
//
// < STORAGE DETAIL >
//
// Unless deleted as a result of calls c api client, the lifetime of all "IR
// Objects" (definitions of opaque types) are designed to be transitively owned
// by the LiteRtModelT which is generally the longset living object. See various
// "Emplace" methods.
//
// Since c api clients interface with pointers to IR Ojbects, a form of pointer
// stability is desirable. Classes in this file enforce that pointers to IR
// Objects are valid for their entire life time. Thus a c api client may store
// pointers and depend on referential equality of IR Objects thoughout different
// calls. This also facilitates storing edge/parent-references as pointers
// within IR Objects.
//
// Direct copying is generally not allowed for IR Objects since copying
// instances of mutually recursive types is not entirely well-defined.
//
// IR Objects are generally default constructible to facilitate stable storage
// and iterative construction.
//
// < EXPOSING TFLITE SCHEMA >
//
// Direct access to tflite schema types is limited to the "detail" namespace.
// This indicates that encapsulating all the details of the flatbuffer is a WIP.
// Future implementations may use different data forms (new litert serialized
// format, tflite runtime types etc).
//
// < USAGE NOTE >
//
// The classes here contain only simple getters & setters. Care should be taken
// to leave the IR in a valid state when using setters since the graph is
// doubly-linked. Higher-level functionality for correct graph mutation can be
// found in "model_graph.h".
////////////////////////////////////////////////////////////////////////////////

// All tflite schema type usage.
namespace detail {

// OP

// Placeholder for the ind of the dispatch op code added during serialization.
static constexpr auto kDispatchOpCodeTflInd = -1;

void SetTflOpCodeInd(LiteRtOpT& litert_op, int32_t tfl_op_code_ind);

int32_t GetTflOpCodeInd(const LiteRtOpT& litert_op);

template <class Arg>
void SetTflOptions(LiteRtOpT& litert_op, Arg&& arg);

const ::litert::internal::TflOptions& GetTflOptions(const LiteRtOpT& litert_op);

::litert::internal::TflOptions&& TakeTflOptions(LiteRtOpT& litert_op);

// WEIGHT

const ::litert::internal::TflBuffer& GetTflBuffer(
    const LiteRtWeightsT& litert_weights);

litert::internal::TflBufferPtr TakeTflBuffer(LiteRtWeightsT& litert_weights);

void SetTflBuffer(LiteRtWeightsT& litert_weights,
                  litert::internal::TflBufferPtr tfl_buffer);

// MODEL

const std::vector<::litert::internal::TflOpCodePtr>& GetTflOpCodes(
    const LiteRtModelT& litert_model);

template <class Arg>
void SetTflOpCodes(LiteRtModelT& litert_model, Arg&& arg);

std::vector<::litert::internal::TflOpCodePtr>&& TakeTflOpCodes(
    LiteRtModelT& litert_model);

void SetTflInitFlatbuffer(LiteRtModelT& litert_model,
                          ::litert::BufferRef<uint8_t> init_flatbuffer);

::litert::BufferRef<uint8_t> GetTflInitFlatbuffer(
    const LiteRtModelT& litert_model);

}  // namespace detail

//
// Helpers for conceptual unions from C api.
//

// // For requesting opaque data stored within IR.
using BufferProvider = std::function<uint8_t*(size_t size)>;

// TENSOR TYPE

// Detail convenience type for tensor type union.
typedef union {
  LiteRtUnrankedTensorType unranked_tensor_type;
  LiteRtRankedTensorType ranked_tensor_type;
} TensorTypeDetail;

// Union and identifier for tensor types.
using TensorType = std::pair<LiteRtTensorTypeId, TensorTypeDetail>;

// Construct tensor type union as ranked tensor. NOTE: Copies data in `dims`.
TensorType MakeRankedTensorType(LiteRtElementType element_type,
                                absl::Span<const int32_t> dims);

// QUANTIZATION TYPE

// Detail convenience type for quantization type union.
typedef union {
  LiteRtQuantizationPerTensor per_tensor;
  LiteRtQuantizationPerChannel per_channel;
} QuantizationDetail;

// Union and identifier for quantization types.
using Quantization = std::pair<LiteRtQuantizationTypeId, QuantizationDetail>;

// Make default type with quantization info.
inline Quantization MakeEmptyQuantization() {
  return Quantization(kLiteRtQuantizationNone, QuantizationDetail());
}

// Construct quantization type as per tensor.
Quantization MakePerTensorQuantization(float scale, int64_t zero_point);

// Construct quantization type as per channel, requires buffer callback to
// store data.
template <class Scales, class ZeroPoints>
Quantization MakePerChannelQuantization(const Scales& scales,
                                        const ZeroPoints& zero_points,
                                        int32_t quantized_dim,
                                        BufferProvider buffer_provider) {
  const auto size = std::size(scales);
  ABSL_DCHECK_EQ(size, std::size(zero_points));

  Quantization res;
  res.first = kLiteRtQuantizationPerChannel;

  res.second.per_channel.num_channels = size;
  res.second.per_channel.quantized_dimension = quantized_dim;

  const size_t scales_buf_size = size * sizeof(float);
  const size_t zeros_buf_size = size * sizeof(int64_t);
  auto* scales_buf = reinterpret_cast<float*>(buffer_provider(scales_buf_size));
  auto* zeros_buf = reinterpret_cast<int64_t*>(buffer_provider(zeros_buf_size));
  std::copy(std::cbegin(scales), std::cend(scales), scales_buf);
  std::copy(std::cbegin(zero_points), std::cend(zero_points), zeros_buf);

  res.second.per_channel.scales = scales_buf;
  res.second.per_channel.zero_points = zeros_buf;

  return res;
}

//
// Tensor
//

// Constant data associated with a tensor.
class LiteRtWeightsT {
 private:
  using OwnedBuffer = ::litert::OwningBufferRef<uint8_t>;

 public:
  // Underlying data.
  ::litert::BufferRef<uint8_t> Buf() const {
    return ::litert::BufferRef<uint8_t>(tfl_buf_->data.data(),
                                        tfl_buf_->data.size());
  }

  // Set weights via copied data.
  void SetFromBuf(::litert::BufferRef<uint8_t> buf) {
    tfl_buf_->data.assign(buf.Data(), buf.Data() + buf.Size());
  }

  // Set via copied vec.
  void SetFromVec(const std::vector<uint8_t>& vec) { tfl_buf_->data = vec; }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtWeightsT()
      : tfl_buf_(std::make_unique<::litert::internal::TflBuffer>()) {}
  LiteRtWeightsT(const LiteRtWeightsT&) = delete;
  LiteRtWeightsT(LiteRtWeightsT&&) = default;
  LiteRtWeightsT& operator=(const LiteRtWeightsT&) = delete;
  LiteRtWeightsT& operator=(LiteRtWeightsT&&) = default;

  // Friendship for internal tflite details.
  friend const ::litert::internal::TflBuffer& detail::GetTflBuffer(
      const LiteRtWeightsT& litert_weights);

  friend litert::internal::TflBufferPtr detail::TakeTflBuffer(
      LiteRtWeightsT& litert_weights);

  friend void detail::SetTflBuffer(LiteRtWeightsT& litert_weights,
                                   litert::internal::TflBufferPtr tfl_buffer);

 private:
  // TFLITE
  ::litert::internal::TflBufferPtr tfl_buf_;
};

// Fundamental value in a litert program, "edges" in the graph.
class LiteRtTensorT {
 private:
  using UserData = std::unique_ptr<uint8_t[]>;

 public:
  using Ref = std::reference_wrapper<LiteRtTensorT>;
  using Use = std::pair<LiteRtOp, LiteRtParamIndex>;
  using UseVec = std::vector<Use>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtTensorT>;

  // The ops that take this tensor as input.
  const std::vector<LiteRtOp>& Users() const { return users_; }
  std::vector<LiteRtOp>& Users() { return users_; }

  // Which operand index users take this tensor on, respects the ordering of
  // users..
  const std::vector<LiteRtParamIndex>& UserArgInds() const {
    return user_arg_inds_;
  }
  std::vector<LiteRtParamIndex>& UserArgInds() { return user_arg_inds_; }

  // Number of uses, same as number of user arg inds.
  size_t NumUses() const { return users_.size(); }

  // Get the ith use.
  Use GetUse(size_t ind) const {
    return {users_.at(ind), user_arg_inds_.at(ind)};
  }

  // Remove the use at the given index.
  void RemoveUse(size_t ind) {
    users_.erase(users_.begin() + ind);
    user_arg_inds_.erase(user_arg_inds_.begin() + ind);
  }

  // Get the op that outputs this tensor, null if constant or subgraph input.
  LiteRtOp DefiningOp() const { return defining_op_; }

  // Get the output index of the op that defines this tensor, only meaningful
  // if it has a defining op.
  LiteRtParamIndex DefiningOpOutInd() const { return defining_op_out_ind_; }

  // Update the defining op of this tensor. The caller is required to update the
  // given op's output if not already correct.
  void SetDefiningOp(LiteRtOpT& defining_op, LiteRtParamIndex out_ind) {
    defining_op_ = &defining_op;
    defining_op_out_ind_ = out_ind;
  }

  // Set the defining op to none.
  void ClearDefiningOp() {
    defining_op_ = nullptr;
    defining_op_out_ind_ = 0;
  }

  // Any constant data associated with this tensor.
  const LiteRtWeightsT& Weights() const { return weights_; }
  LiteRtWeightsT& Weights() { return weights_; }

  // Authored name associated with this tensor. May be empty.
  absl::string_view Name() const { return name_; }

  // Update the name associated with this tensor.
  void SetName(std::string name) { name_ = std::move(name); }

  // Get quantization information for this tensor.
  const Quantization& Qparams() const { return quantization_; }
  Quantization& Qparams() { return quantization_; }

  // Set quantization information.
  template <class Arg>
  void SetQarams(Arg&& arg) {
    quantization_ = std::forward<Arg>(arg);
  }

  // Get the tensor type of this tensor.
  const TensorType& Type() const { return tensor_type_; }
  TensorType& Type() { return tensor_type_; }

  // Set the tensor type.
  template <class Arg>
  void SetType(Arg&& arg) {
    tensor_type_ = std::forward<Arg>(arg);
  }

  // Get a new buffer that will live as long as this tensor. Used for storing
  // various buffers passed through c-api (dims, quantization etc).
  uint8_t* RequestBuffer(size_t size) {
    user_data_.push_back(std::make_unique<uint8_t[]>(size));
    return user_data_.back().get();
  }

  // Allow for implicit conversion to bufer provider.
  // NOLINTNEXTLINE
  operator BufferProvider() & {
    return [this](auto s) { return this->RequestBuffer(s); };
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtTensorT() = default;
  LiteRtTensorT(const LiteRtTensorT&) = delete;
  LiteRtTensorT(LiteRtTensorT&&) = default;
  LiteRtTensorT& operator=(const LiteRtTensorT&) = delete;
  LiteRtTensorT& operator=(LiteRtTensorT&&) = default;

 private:
  std::vector<LiteRtOp> users_;
  std::vector<LiteRtParamIndex> user_arg_inds_;

  LiteRtOp defining_op_ = nullptr;
  LiteRtParamIndex defining_op_out_ind_;

  LiteRtWeightsT weights_;
  Quantization quantization_;
  TensorType tensor_type_;

  std::string name_;

  std::vector<UserData> user_data_;
};

// Helper to get multiple uses at once.
template <class Inds>
LiteRtTensorT::UseVec GetTensorUses(const LiteRtTensorT& tensor,
                                    const Inds& inds) {
  auto start = std::cbegin(inds);
  auto end = std::cend(inds);
  LiteRtTensorT::UseVec uses(end - start);
  auto get = [&tensor = std::as_const(tensor)](auto i) {
    return tensor.GetUse(i);
  };
  std::transform(start, end, uses.begin(), get);
  return uses;
}

//
// Op
//

// Fundamental unit of compute of a litert program, or "nodes" in the graph.
class LiteRtOpT {
 public:
  using Ref = std::reference_wrapper<LiteRtOpT>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtOpT>;

  // Input tensors for this op.
  const std::vector<LiteRtTensor>& Inputs() const { return inputs_; }
  std::vector<LiteRtTensor>& Inputs() { return inputs_; }

  // Access input at given ind.
  LiteRtTensorT& Input(size_t ind) { return *Inputs().at(ind); }
  const LiteRtTensorT& Input(size_t ind) const { return *Inputs().at(ind); }

  // Number of input tensors.
  size_t NumInputs() const { return inputs_.size(); }

  // Output tensors for this op.
  const std::vector<LiteRtTensor>& Outputs() const { return outputs_; }
  std::vector<LiteRtTensor>& Outputs() { return outputs_; }

  // Number of output tensors.
  size_t NumOutputs() const { return outputs_.size(); }

  // Access output at given ind.
  LiteRtTensorT& Output(size_t ind) { return *Outputs().at(ind); }
  const LiteRtTensorT& Output(size_t ind) const { return *Outputs().at(ind); }

  // Remove the ith entry of input list.
  void RemoveInput(size_t ind) { inputs_.erase(inputs_.begin() + ind); }

  // Remove the ith entry of output list.
  void RemoveOutput(size_t ind) { outputs_.erase(outputs_.begin() + ind); }

  // Get any custom options attached to this op. Empty if there are none.
  litert::BufferRef<uint8_t> CustomOptions() const { return custom_options_; }

  // Attach custom opaque optins to this op.
  template <class... Args>
  void SetCustomOptions(Args&&... args) {
    custom_options_ =
        ::litert::OwningBufferRef<uint8_t>(std::forward<Args>(args)...);
  }

  // Sets the custom options to zero length buffer.
  void ClearCustomOptions() { custom_options_.Reset(); }

  // Get the op code.
  LiteRtOpCode OpCode() const { return litert_op_code_; }

  // Set the op code.
  void SetOpCode(LiteRtOpCode litert_op_code) {
    litert_op_code_ = litert_op_code;
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtOpT() = default;
  LiteRtOpT(const LiteRtOpT&) = delete;
  LiteRtOpT(LiteRtOpT&&) = default;
  LiteRtOpT& operator=(const LiteRtOpT&) = delete;
  LiteRtOpT& operator=(LiteRtOpT&&) = default;

  // Friendship for internal tflite details.
  friend void detail::SetTflOpCodeInd(LiteRtOpT& litert_op,
                                      int32_t tfl_op_code_ind);

  friend int32_t detail::GetTflOpCodeInd(const LiteRtOpT& litert_op);

  template <class Arg>
  friend void detail::SetTflOptions(LiteRtOpT& litert_op, Arg&& arg);

  friend const ::litert::internal::TflOptions& detail::GetTflOptions(
      const LiteRtOpT& litert_op);

  friend ::litert::internal::TflOptions&& detail::TakeTflOptions(
      LiteRtOpT& litert_op);

 private:
  LiteRtOpCode litert_op_code_;

  ::litert::OwningBufferRef<uint8_t> custom_options_;

  std::vector<LiteRtTensor> inputs_;
  std::vector<LiteRtTensor> outputs_;

  // TFLITE
  int32_t tfl_op_code_ind_ = detail::kDispatchOpCodeTflInd;
  ::litert::internal::TflOptions tfl_option_;
};

//
// Subgraph
//

// Fundamental block of a litert program. Manages the storage of all
// ops and tensor within.
class LiteRtSubgraphT {
 public:
  using Ref = std::reference_wrapper<LiteRtSubgraphT>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtSubgraphT>;

  // Get a stable pointer for all of the tensors in this subgraph.
  absl::Span<LiteRtTensor> Tensors() { return tensors_.Elements(); }
  absl::Span<const LiteRtTensor> Tensors() const { return tensors_.Elements(); }

  // Access the tensor at given ind.
  LiteRtTensorT& Tensor(size_t ind) { return *Tensors().at(ind); }
  const LiteRtTensorT& Tensor(size_t ind) const { return *Tensors().at(ind); }

  // Get a stable pointer for all of the ops in this subgraph. Will
  // be a valid toplological order.
  absl::Span<LiteRtOp> Ops() { return ops_.Elements(); }
  absl::Span<const LiteRtOp> Ops() const { return ops_.Elements(); }

  // Access op at the given ind.
  LiteRtOpT& Op(size_t ind) { return *Ops().at(ind); }
  const LiteRtOpT& Op(size_t ind) const { return *Ops().at(ind); }

  // All the subgraph input tensors, these also exist in Tensors.
  const std::vector<LiteRtTensor>& Inputs() const { return inputs_; }
  std::vector<LiteRtTensor>& Inputs() { return inputs_; }

  // Number of inputs tensors.
  size_t NumInputs() const { return inputs_.size(); }

  // Access the subgraph input at given ind.
  LiteRtTensorT& Input(size_t ind) { return *Inputs().at(ind); }
  const LiteRtTensorT& Input(size_t ind) const { return *Inputs().at(ind); }

  // All the subgraph output tensors, these also exist in Tensors.
  const std::vector<LiteRtTensor>& Outputs() const { return outputs_; }
  std::vector<LiteRtTensor>& Outputs() { return outputs_; }

  // Number of outputs tensors.
  size_t NumOutputs() const { return outputs_.size(); }

  // Access the subgraph output at given ind.
  LiteRtTensorT& Output(size_t ind) { return *Outputs().at(ind); }
  const LiteRtTensorT& Output(size_t ind) const { return *Outputs().at(ind); }

  // Clear the entry for the ith input.
  void ClearInput(size_t ind) { inputs_.erase(inputs_.begin() + ind); }

  // Clear the entry for the ith output.
  void ClearOutput(size_t ind) { outputs_.erase(outputs_.begin() + ind); }

  // Construct a new tensor which will be owned by this subgraph and get a
  // reference to it.
  template <class... Args>
  LiteRtTensorT& EmplaceTensor(Args&&... args) {
    return tensors_.EmplaceBack(std::forward<Args>(args)...);
  }

  // Construct a new op which will be owned by this subgraph and get a
  // reference to it.
  template <class... Args>
  LiteRtOpT& EmplaceOp(Args&&... args) {
    return ops_.EmplaceBack(std::forward<Args>(args)...);
  }

  // De-allocates ops that pass given predicate. Returns number of ops removed.
  size_t RemoveOpIf(std::function<bool(const LiteRtOpT& op)> pred) {
    return ops_.RemoveIf(pred);
  }

  // De-allocates tensors that pass given predicate. Returns number of tensors
  // removed.
  size_t RemoveTensorIf(std::function<bool(const LiteRtTensorT& tensor)> pred) {
    return tensors_.RemoveIf(pred);
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtSubgraphT() = default;
  LiteRtSubgraphT(const LiteRtSubgraphT&) = delete;
  LiteRtSubgraphT(LiteRtSubgraphT&&) = default;
  LiteRtSubgraphT& operator=(const LiteRtSubgraphT&) = delete;
  LiteRtSubgraphT& operator=(LiteRtSubgraphT&&) = default;

 private:
  LiteRtTensorT::Alloc tensors_;

  LiteRtOpT::Alloc ops_;

  std::vector<LiteRtTensor> inputs_;
  std::vector<LiteRtTensor> outputs_;
};

//
// Signature
//

class LiteRtSignatureT {
 private:
  using StrVec = std::vector<std::string>;

 public:
  using Ptr = std::unique_ptr<LiteRtSignatureT>;
  using Ref = std::reference_wrapper<LiteRtSignatureT>;
  using Alloc = ::litert::internal::IrAllocator<LiteRtSignatureT>;

  static constexpr absl::string_view kDefaultSignatureKey =
      "<placeholder signature>";

  LiteRtSignatureT(LiteRtSubgraph subgraph, StrVec input_names,
                   StrVec output_names, std::string key)
      : key_(std::move(key)),
        subgraph_(subgraph),
        input_names_(std::move(input_names)),
        output_names_(std::move(output_names)) {}

  // String named inputs for called subgraph.
  const StrVec& InputNames() const { return input_names_; }

  // String named outputs for called subgraph.
  const StrVec& OutputNames() const { return output_names_; }

  // Get the callable subgraph.
  const LiteRtSubgraphT& GetSubgraph() const { return *subgraph_; }
  LiteRtSubgraphT& GetSubgraph() { return *subgraph_; }

  // Name of the callable signature.
  absl::string_view Key() const { return key_; }

  bool operator==(const LiteRtSignatureT& other) const {
    const auto key_eq = key_ == other.key_;
    const auto subgraph_eq = subgraph_ == other.subgraph_;
    const auto input_names_eq = input_names_ == other.input_names_;
    const auto output_names_eq = output_names_ == other.output_names_;
    return key_eq && subgraph_eq && input_names_eq && output_names_eq;
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtSignatureT() = default;
  LiteRtSignatureT(const LiteRtSignatureT&) = delete;
  LiteRtSignatureT(LiteRtSignatureT&&) = default;
  LiteRtSignatureT& operator=(const LiteRtSignatureT&) = delete;
  LiteRtSignatureT& operator=(LiteRtSignatureT&&) = default;

 private:
  std::string key_;

  LiteRtSubgraph subgraph_;

  StrVec input_names_;
  StrVec output_names_;
};

// Make a basic signature from information in the given subgraph. Used with the
// main subgraph when no explicit signatures have been authored.
LiteRtSignatureT MakeDefaultSignature(LiteRtSubgraph subgraph);

//
// Model
//

// Root-level graph object for litert programs. Manages the storage
// of all litert graph objects within.
class LiteRtModelT {
 private:
  using MetadataMap =
      absl::flat_hash_map<std::string, litert::OwningBufferRef<uint8_t>>;

 public:
  using Ref = std::reference_wrapper<LiteRtModelT>;
  using Ptr = std::unique_ptr<LiteRtModelT>;
  using TflOpCodes = std::vector<litert::internal::TflOpCodePtr>;

  // TODO replace this with the index of the default signature.
  static constexpr const size_t kMainSubgraphIndex = 0;

  // OBSERVERS

  // Get a stable pointer for all of the subgraphs within this model.
  absl::Span<LiteRtSubgraph> Subgraphs() { return subgraphs_.Elements(); }
  absl::Span<const LiteRtSubgraph> Subgraphs() const {
    return subgraphs_.Elements();
  }

  // Access subgraph at given ind.
  LiteRtSubgraphT& Subgraph(size_t ind) { return *Subgraphs().at(ind); }
  const LiteRtSubgraphT& Subgraph(size_t ind) const {
    return *Subgraphs().at(ind);
  }

  // Number of subraphs.
  size_t NumSubgraphs() const { return subgraphs_.Elements().size(); }

  // Default entry point of this model.
  const LiteRtSubgraphT* MainSubgraph() const {
    return &Subgraph(kMainSubgraphIndex);
  }
  LiteRtSubgraph MainSubgraph() { return &Subgraph(kMainSubgraphIndex); }

  // Look up signature by key.
  litert::Expected<LiteRtSignatureT::Ref> FindSignature(
      absl::string_view signature_key) const {
    for (LiteRtSignature sig : signatures_.Elements()) {
      if (sig->Key() == signature_key) {
        return std::ref(*sig);
      }
    }
    return ::litert::Error(kLiteRtStatusErrorNotFound, "Signature not found");
  }

  // All signatures registered with this model.
  absl::Span<LiteRtSignature> Signatures() const {
    return signatures_.Elements();
  }

  // Look up metadata by key, getting a view of its buffer as a string
  // if it exists.
  litert::Expected<litert::BufferRef<uint8_t>> FindMetadata(
      absl::string_view key) const {
    if (auto it = metadata_.find(key); it != metadata_.end()) {
      return it->second;
    }
    return ::litert::Error(kLiteRtStatusErrorNotFound);
  }

  // Metadata key-val pair iterator.
  MetadataMap::iterator MetadataBegin() { return metadata_.begin(); }
  MetadataMap::iterator MetadataEnd() { return metadata_.end(); }

  // Remvoe and take ownership of the metadata under given key if it exists.
  litert::Expected<litert::OwningBufferRef<uint8_t>> PopMetadata(
      absl::string_view key) {
    if (auto it = metadata_.find(key); it != metadata_.end()) {
      return metadata_.extract(it).mapped();
    }
    return ::litert::Error(kLiteRtStatusErrorNotFound);
  }

  // BUILDERS

  // Build a new subgraph and get a stable reference to it.
  template <class... Args>
  LiteRtSubgraphT& EmplaceSubgraph(Args&&... args) {
    return subgraphs_.EmplaceBack(std::forward<Args>(args)...);
  }

  // Transfers given subgraphs into this model.
  void TransferSubgraphs(LiteRtSubgraphT::Alloc&& subgraphs) {
    subgraphs_.Transfer(std::move(subgraphs));
  }

  // Cut all by the first `size` subgraphs. Does nothing if given size is
  // greater or equal to current.
  void ResizeSubgraphsDown(size_t size) { subgraphs_.ResizeDown(size); }

  // Adds a new metadata buffer to the model. Fails if it already exists.
  template <class... Args>
  LiteRtStatus PushMetadata(absl::string_view key, Args&&... args) {
    if (metadata_.contains(key)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    metadata_.insert(
        {std::string(key.begin(), key.end()),
         ::litert::OwningBufferRef<uint8_t>(std::forward<Args>(args)...)});
    return kLiteRtStatusOk;
  }

  // Construct a new signature for this model.
  template <class... Args>
  LiteRtSignatureT& EmplaceSignature(Args&&... args) {
    return signatures_.EmplaceBack(std::forward<Args>(args)...);
  }

  // IR is generally, default constructible and movable but not copyable.
  LiteRtModelT() = default;
  LiteRtModelT(const LiteRtModelT&) = delete;
  LiteRtModelT(LiteRtModelT&&) = default;
  LiteRtModelT& operator=(const LiteRtModelT&) = delete;
  LiteRtModelT& operator=(LiteRtModelT&&) = default;

  // Friendship for internal tflite details.
  friend const TflOpCodes& detail::GetTflOpCodes(
      const LiteRtModelT& litert_model);

  template <class Arg>
  friend void detail::SetTflOpCodes(LiteRtModelT& litert_model, Arg&& arg);

  friend TflOpCodes&& detail::TakeTflOpCodes(LiteRtModelT& litert_model);

  friend void detail::SetTflInitFlatbuffer(
      LiteRtModelT& litert_model, ::litert::BufferRef<uint8_t> init_flatbuffer);

  friend ::litert::BufferRef<uint8_t> detail::GetTflInitFlatbuffer(
      const LiteRtModelT& litert_model);

 private:
  LiteRtSubgraphT::Alloc subgraphs_;
  LiteRtSignatureT::Alloc signatures_;

  MetadataMap metadata_;

  // TFLITE
  TflOpCodes tfl_operator_codes_;
  litert::BufferRef<uint8_t> tfl_init_flatbuffer_;
};

// Lookup subgraph by signature name.
::litert::Expected<LiteRtSubgraph> LookupSubgraph(
    const LiteRtModelT& model, absl::string_view signature_key);

//
// Utils
//

// Used for communicating selections of ops.
class LiteRtOpListT {
 public:
  void Push(LiteRtOp op) { ops_.push_back(op); }

  std::vector<LiteRtOp> Vec() const {
    std::vector<LiteRtOp> res;
    res.reserve(ops_.size());
    res.assign(ops_.begin(), ops_.end());
    return res;
  }

 private:
  // Investigate if this is possible with vector (hit some issues).
  std::list<LiteRtOp> ops_;
};

namespace detail {

template <class Arg>
void SetTflOptions(LiteRtOpT& litert_op, Arg&& arg) {
  litert_op.tfl_option_ = std::forward<Arg>(arg);
}

template <class Arg>
void SetTflOpCodes(LiteRtModelT& litert_model, Arg&& arg) {
  litert_model.tfl_operator_codes_ = std::forward<Arg>(arg);
}

}  // namespace detail

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_MODEL_MODEL_H_
