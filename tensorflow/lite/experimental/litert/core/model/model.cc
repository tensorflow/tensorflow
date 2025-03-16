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

#include "tensorflow/lite/experimental/litert/core/model/model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_layout.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

using ::litert::BufferRef;
using ::litert::internal::TflBuffer;
using ::litert::internal::TflBufferPtr;
using ::litert::internal::TflOpCode;
using ::litert::internal::TflOpCodePtr;
using ::litert::internal::TflOptions;
using ::litert::internal::TflOptions2;

std::optional<std::string> GetCustomOpCode(const LiteRtModelT& model,
                                           const LiteRtOpT& op) {
  if (op.OpCode() != kLiteRtOpCodeTflCustom) {
    return {};
  }
  const auto& tfl_op_codes = detail::GetTflOpCodes(model);
  const auto tfl_op_code_ind = detail::GetTflOpCodeInd(op);
  return tfl_op_codes[tfl_op_code_ind]->custom_code;
}

TensorType MakeRankedTensorType(LiteRtElementType element_type,
                                absl::Span<const int32_t> dims) {
  TensorType tensor_type;
  tensor_type.first = kLiteRtRankedTensorType;
  auto& ranked = tensor_type.second.ranked_tensor_type;
  ranked.element_type = element_type;
  ABSL_DCHECK_LE(dims.size(), LITERT_TENSOR_MAX_RANK);
  ranked.layout.rank = dims.size();
  std::copy(dims.begin(), dims.end(), ranked.layout.dimensions);
  // Strides not yet supported.
  ranked.layout.strides = nullptr;
  return tensor_type;
}

Quantization MakePerTensorQuantization(float scale, int64_t zero_point) {
  Quantization quantization;
  quantization.first = kLiteRtQuantizationPerTensor;
  quantization.second.per_tensor.scale = scale;
  quantization.second.per_tensor.zero_point = zero_point;
  return quantization;
}

LiteRtSignatureT MakeDefaultSignature(LiteRtSubgraph subgraph) {
  auto tensor_name = [](auto* tensor) { return std::string(tensor->Name()); };

  auto in_start = subgraph->Inputs().cbegin();
  auto in_end = subgraph->Inputs().cend();
  std::vector<std::string> input_names(subgraph->NumInputs());
  std::transform(in_start, in_end, input_names.begin(), tensor_name);

  auto out_start = subgraph->Outputs().cbegin();
  auto out_end = subgraph->Outputs().cend();
  std::vector<std::string> output_names(subgraph->NumOutputs());
  std::transform(out_start, out_end, output_names.begin(), tensor_name);

  std::string name(LiteRtSignatureT::kDefaultSignatureKey);
  return LiteRtSignatureT(subgraph, std::move(input_names),
                          std::move(output_names), std::move(name));
}

::litert::Expected<LiteRtSubgraph> LookupSubgraph(
    const LiteRtModelT& model, absl::string_view signature_key) {
  auto sig = model.FindSignature(signature_key);
  if (!sig) {
    return sig.Error();
  }
  return &sig->get().GetSubgraph();
}

void LiteRtModelT::TransferSubgraphTo(LiteRtSubgraphT::Alloc& dest,
                                      std::vector<size_t> indices) {
  if (indices.empty()) {
    return;
  }
  std::sort(indices.begin(), indices.end());
  std::vector<int> new_inds(subgraphs_.Size(), 0);
  auto num_removed = 0;
  auto i = indices.begin();
  for (size_t j = 0; j < new_inds.size(); ++j) {
    if (i != indices.end() && *i == j) {
      ++num_removed;
      // Keep track of removed sgs just for dcheck.
      new_inds[j] = -1;
      ++i;
      continue;
    }
    new_inds[j] = j - num_removed;
  }

  ForEachIr(
      this, [&](LiteRtSubgraph subgraph, int32_t subgraph_index, LiteRtOp op) {
        if (op->OpCode() != kLiteRtOpCodeShloComposite) {
          return;
        }
        auto opts = detail::TakeTflOptions2(*op);
        auto& decomp_ind =
            opts.AsStableHLOCompositeOptions()->decomposition_subgraph_index;
        const auto new_ind = new_inds[decomp_ind];

        // This op is either in a removed subgraph or refers to a subgraph that
        // is not being removed.
        ABSL_DCHECK((subgraph_index == -1) || (new_ind >= 0));

        decomp_ind = new_ind;
        detail::SetTflOptions2(*op, std::move(opts));
      });

  subgraphs_.TransferTo(dest, std::move(indices));
}

namespace detail {

void SetTflOpCodeInd(LiteRtOpT& litert_op, int32_t tfl_op_code_ind) {
  litert_op.tfl_op_code_ind_ = tfl_op_code_ind;
}

int32_t GetTflOpCodeInd(const LiteRtOpT& litert_op) {
  return litert_op.tfl_op_code_ind_;
}

const TflOptions& GetTflOptions(const LiteRtOpT& litert_op) {
  return litert_op.tfl_option_;
}

const TflOptions2& GetTflOptions2(const LiteRtOpT& litert_op) {
  return litert_op.tfl_option_2_;
}

TflOptions&& TakeTflOptions(LiteRtOpT& litert_op) {
  return std::move(litert_op.tfl_option_);
}

TflOptions2&& TakeTflOptions2(LiteRtOpT& litert_op) {
  return std::move(litert_op.tfl_option_2_);
}

const std::vector<TflOpCodePtr>& GetTflOpCodes(
    const LiteRtModelT& litert_model) {
  return litert_model.tfl_operator_codes_;
}

std::vector<TflOpCodePtr>&& TakeTflOpCodes(LiteRtModelT& litert_model) {
  return std::move(litert_model.tfl_operator_codes_);
}

// new stuff start
void SetTflFlatbuffer(LiteRtModelT& litert_model,
                      LiteRtModelT::TflFlatbuffer&& tfl_flatbuffer) {
  litert_model.tfl_flatbuffer_ = std::move(tfl_flatbuffer);
}

const LiteRtModelT::TflFlatbuffer& GetTflFlatbuffer(
    const LiteRtModelT& litert_model) {
  return litert_model.tfl_flatbuffer_;
}
// new stuff end

}  // namespace detail
