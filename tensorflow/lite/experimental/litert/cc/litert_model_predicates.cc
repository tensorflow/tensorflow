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

#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

namespace litert {
namespace {

template <typename T>
bool Any(absl::Span<const T> vals, std::function<bool(const T&)> unary_pred) {
  for (const auto& val : vals) {
    if (unary_pred(val)) {
      return true;
    }
  }
  return false;
}

bool UseSoftEqual(const Tensor::TensorUse& actual_use,
                  const UseInfo& expected_use) {
  if (expected_use.user_param_ind.has_value() &&
      actual_use.user_arg_ind != expected_use.user_param_ind.value()) {
    return false;
  }
  if (expected_use.op_code.has_value() &&
      actual_use.user.Code() != expected_use.op_code.value()) {
    return false;
  }
  return true;
}

}  // namespace

// Does given tensor have given type and shape info. Optional values considered
// to be a vacous match.
bool MatchRankedTensorType(const RankedTensorType& tensor_type,
                           const TensorTypeInfo& expected) {
  if (expected.element_type.has_value() &&
      (tensor_type.ElementType() != expected.element_type.value())) {
    return false;
  }

  if (expected.dims.has_value()) {
    auto actual_dims = tensor_type.Layout().Dimensions();
    auto expected_dims = absl::MakeConstSpan(expected.dims.value());
    return AllZip(actual_dims, expected_dims,
                  [](auto l, auto r) -> bool { return l == r; });
  }
  return true;
}

// Does given op have signature matching given types. Optional values considered
// to be a vacous match.
bool MatchOpType(
    const Op& op,
    const std::vector<std::optional<TensorTypeInfo>>& expected_inputs,
    const std::vector<std::optional<TensorTypeInfo>>& expected_outputs) {
  auto match = [](const Tensor& actual,
                  const std::optional<TensorTypeInfo>& expected) -> bool {
    if (!expected.has_value()) {
      return true;
    }
    auto actual_ranked_tensor_type = actual.RankedTensorType();
    // Don't return a match if the tensor is unranked.
    if (!actual_ranked_tensor_type) {
      return false;
    }
    return MatchRankedTensorType(*actual_ranked_tensor_type, expected.value());
  };

  const bool inputs_match = AllZip(absl::MakeConstSpan(op.Inputs()),
                                   absl::MakeConstSpan(expected_inputs), match);
  const bool outputs_match =
      AllZip(absl::MakeConstSpan(op.Outputs()),
             absl::MakeConstSpan(expected_outputs), match);
  return inputs_match && outputs_match;
}

bool MatchUse(const Tensor& tensor, const UseInfo& expected_use) {
  auto soft_equal = [&expected_use = std::as_const(expected_use)](
                        const Tensor::TensorUse& actual_use) {
    return UseSoftEqual(actual_use, expected_use);
  };
  return Any<Tensor::TensorUse>(tensor.Uses(), soft_equal);
}

bool MatchUses(const Tensor& tensor, const std::vector<UseInfo>& expected_uses,
               bool strict) {
  const auto uses = tensor.Uses();
  if (strict && uses.size() != expected_uses.size()) {
    return false;
  }
  auto not_use = [&tensor =
                      std::as_const(tensor)](const UseInfo& expected_use) {
    return !MatchUse(tensor, expected_use);
  };
  return !Any<UseInfo>(expected_uses, not_use);
}

}  // namespace litert
