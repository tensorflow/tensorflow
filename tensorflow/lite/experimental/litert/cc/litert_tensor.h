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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_H_

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"

namespace litert {

// [WIP] Simple C++ wrapper over the C tensor api. Provided for convenience.
// Currently only supports LiteRtRankedTensors.
//
// NOTE ON USAGE: This "unpacks" upfront some of the data behind the
// LiteRtTensor for efficiency and a cleaner interface (no status checks needed
// on getters). Becasuse of this, it is required that `tensor : LiteRtTensor` is
// stable and unmutated throughout the lifetime. This is guaranteed within (but
// not between) calls to an LiteRtCompilerPlugin. Plugins should close all
// LiteRtTensorManagers before exiting a call and initialize fresh ones in later
// calls.
//
// This is an evolution of "graph_tools" and logic will be consolidated in
// the future.
//
// TODO Expand this abstraction
// to handle the union of possible tensor types cleanly as well as
// defining op/users.
class LiteRtTensorManager {
 public:
  using Unique = std::unique_ptr<LiteRtTensorManager>;

  static LiteRtStatus MakeFromTensor(LiteRtTensor tensor, Unique& result);

  uint32_t Rank() const;

  absl::Span<const int32_t> Dims() const;

  bool HasStrides() const {
    return ranked_tensor_type_.layout.strides != nullptr;
  }
  absl::Span<const uint32_t> Strides() const;

  LiteRtElementType ElementType() const;

  bool IsSubgraphOutput() const;

  bool IsSubgraphInput() const;

  LiteRtTensor Tensor();

 private:
  LiteRtTensor tensor_;

  LiteRtRankedTensorType ranked_tensor_type_;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_TENSOR_H_
