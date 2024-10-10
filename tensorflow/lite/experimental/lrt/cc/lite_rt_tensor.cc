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

#include "tensorflow/lite/experimental/lrt/cc/lite_rt_tensor.h"

#include <cstdint>
#include <memory>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"

namespace lrt {

absl::Span<const int32_t> LrtTensorManager::Dims() const {
  return absl::MakeConstSpan(ranked_tensor_type_.layout.dimensions, Rank());
}

absl::Span<const uint32_t> LrtTensorManager::Strides() const {
  if (ranked_tensor_type_.layout.strides) {
    return absl::MakeConstSpan(ranked_tensor_type_.layout.strides, Rank());
  } else {
    return {};
  }
}

uint32_t LrtTensorManager::Rank() const {
  return ranked_tensor_type_.layout.rank;
}

LrtElementType LrtTensorManager::ElementType() const {
  return ranked_tensor_type_.element_type;
}

LrtTensor LrtTensorManager::Tensor() { return tensor_; }

LrtStatus LrtTensorManager::MakeFromTensor(LrtTensor tensor, Unique& result) {
  result = std::make_unique<LrtTensorManager>();

  LrtTensorTypeId type_id;
  LRT_RETURN_STATUS_IF_NOT_OK(GetTensorTypeId(tensor, &type_id));
  LRT_ENSURE_SUPPORTED(type_id == kLrtRankedTensorType,
                       "Only RankedTensorType currently supported in C++ api.");

  LRT_RETURN_STATUS_IF_NOT_OK(
      GetRankedTensorType(tensor, &result->ranked_tensor_type_));
  result->tensor_ = tensor;

  return kLrtStatusOk;
}

bool LrtTensorManager::IsSubgraphOutput() const {
  return ::graph_tools::MatchTensorNoUses(tensor_);
}

bool LrtTensorManager::IsSubgraphInput() const {
  return ::graph_tools::MatchTensorNoDefiningOp(tensor_) &&
         ::graph_tools::MatchNoWeights(tensor_);
}

}  // namespace lrt
