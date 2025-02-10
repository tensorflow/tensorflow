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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_PREDICATES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_PREDICATES_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

// Predicates used for matching patterns in the graph. NOTE: All optionals in
// matcher arguments are considered to be a vacous match.

namespace litert {

struct TensorTypeInfo {
  std::optional<ElementType> element_type = std::nullopt;
  std::optional<absl::InlinedVector<int32_t, 4>> dims = std::nullopt;

  explicit TensorTypeInfo(ElementType element_type)
      : element_type(element_type) {}
  explicit TensorTypeInfo(absl::InlinedVector<int32_t, 4> dims) : dims(dims) {}
  TensorTypeInfo(ElementType element_type, absl::InlinedVector<int32_t, 4> dims)
      : element_type(element_type), dims(dims) {}
};

struct UseInfo {
  std::optional<LiteRtOpCode> op_code = std::nullopt;
  std::optional<LiteRtParamIndex> user_param_ind = std::nullopt;
};

// Does this tensor have given type and shape info.
bool MatchRankedTensorType(const RankedTensorType& tensor_type,
                           const TensorTypeInfo& expected);

// Does this op have signature matching given types.
bool MatchOpType(
    const Op& op,
    const std::vector<std::optional<TensorTypeInfo>>& expected_inputs,
    const std::vector<std::optional<TensorTypeInfo>>& expected_outputs);

// Does this tensor contain weights whose values match expected_data.
template <typename T>
inline bool MatchWeights(const Tensor& tensor,
                         absl::Span<const T> expected_data) {
  auto weights = tensor.WeightsData<T>();
  return weights.HasValue() && *weights == expected_data;
}

// Does this tensor have a user with the given information.
bool MatchUse(const Tensor& tensor, const UseInfo& expected_use);

// Does this tensor have matching users. If "strict" is true, then expected_uses
// size must equal the number of actual uses, otherwise just checks each
// expected_use match an actual use.
bool MatchUses(const Tensor& tensor, const std::vector<UseInfo>& expected_uses,
               bool strict = true);

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_MODEL_PREDICATES_H_
