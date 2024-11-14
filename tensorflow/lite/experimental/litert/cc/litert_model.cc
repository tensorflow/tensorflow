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

#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"

namespace litert {

bool Tensor::IsSubgraphOutput() const { return Uses().empty(); }

bool Tensor::IsSubgraphInput() const {
  return !HasWeights() && !DefiningOp().has_value();
}

bool Tensor::IsConstant() const {
  return HasWeights() && !DefiningOp().has_value();
}

SmallVec<Tensor::TensorUse> Tensor::Uses() const {
  LiteRtParamIndex num_uses;
  LiteRtOpArray users;
  LiteRtParamIndex* user_arg_inds;
  litert::internal::AssertOk(LiteRtGetTensorUses, Get(), &num_uses, &users,
                             &user_arg_inds);
  SmallVec<Tensor::TensorUse> res;
  for (int i = 0; i < num_uses; ++i) {
    res.push_back(Tensor::TensorUse{Op(users[i]), user_arg_inds[i]});  // NOLINT
  }
  return res;
}

}  // namespace litert
