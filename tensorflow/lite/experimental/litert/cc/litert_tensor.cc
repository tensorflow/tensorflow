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

#include "tensorflow/lite/experimental/litert/cc/litert_tensor.h"

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"

namespace litert {

absl::Span<const int32_t> LiteRtTensorManager::Dims() const {
  return absl::MakeConstSpan(ranked_tensor_type_.layout.dimensions, Rank());
}

absl::Span<const uint32_t> LiteRtTensorManager::Strides() const {
  if (ranked_tensor_type_.layout.strides) {
    return absl::MakeConstSpan(ranked_tensor_type_.layout.strides, Rank());
  } else {
    return {};
  }
}

uint32_t LiteRtTensorManager::Rank() const {
  return ranked_tensor_type_.layout.rank;
}

LiteRtElementType LiteRtTensorManager::ElementType() const {
  return ranked_tensor_type_.element_type;
}

LiteRtTensor LiteRtTensorManager::Tensor() { return tensor_; }

LiteRtStatus LiteRtTensorManager::MakeFromTensor(LiteRtTensor tensor,
                                                 Unique& result) {
  result = std::make_unique<LiteRtTensorManager>();

  LiteRtTensorTypeId type_id;
  LITERT_RETURN_STATUS_IF_NOT_OK(GetTensorTypeId(tensor, &type_id));
  LITERT_ENSURE_SUPPORTED(
      type_id == kLiteRtRankedTensorType,
      "Only RankedTensorType currently supported in C++ api.");

  LITERT_RETURN_STATUS_IF_NOT_OK(
      GetRankedTensorType(tensor, &result->ranked_tensor_type_));
  result->tensor_ = tensor;

  return kLiteRtStatusOk;
}

bool LiteRtTensorManager::IsSubgraphOutput() const {
  return ::graph_tools::MatchTensorNoUses(tensor_);
}

bool LiteRtTensorManager::IsSubgraphInput() const {
  return ::graph_tools::MatchTensorNoDefiningOp(tensor_) &&
         ::graph_tools::MatchNoWeights(tensor_);
}

}  // namespace litert
