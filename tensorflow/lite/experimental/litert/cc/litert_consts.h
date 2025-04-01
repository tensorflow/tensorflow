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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_CONSTS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_CONSTS_H_

#include <cstddef>

namespace litert {

// The following constants are used to properly size absl::InlinedVector<>
// uses used in the LiteRT code. Their values don't need to be exact; they
// are just optimization hints.
static constexpr size_t kExpectedMaxTensorRank = 6;
static constexpr size_t kExpectedMaxNumOfTensorUses = 8;
static constexpr size_t kExpectedMaxNumOfOpInputs = 4;
static constexpr size_t kExpectedMaxNumOfOpOutputs = 8;
static constexpr size_t kExpectedMaxNumOfSubgraphInputs = 4;
static constexpr size_t kExpectedMaxNumOfSubgraphOutputs = 4;

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_CONSTS_H_
