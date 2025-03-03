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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_OP_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_OP_OPTIONS_H_

#include <type_traits>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert {

struct OpOptions {
  virtual LiteRtStatus InitFromOp(LiteRtOp op) = 0;
  virtual ~OpOptions() = default;
};

// Struct to hold LiteRt composite ops.
struct CompositeOptions : public OpOptions {
  // The root op.
  LiteRtOp op;
  // Decomposition subgraph.
  int subgraph;
  // The name of the composite op (stored in model).
  absl::string_view name;

  LiteRtStatus InitFromOp(LiteRtOp op) override;
};

// Returns the composite info for the given op if it is a composite op.
template <typename OptionsT>
Expected<OptionsT> GetOptionsAs(const LiteRtOp& op) {
  if constexpr (std::is_same_v<OptionsT, CompositeOptions>) {
    CompositeOptions options;
    LITERT_RETURN_IF_ERROR(options.InitFromOp(op));
    return options;
  } else {
    // TODO: Add more as needed.
    return Unexpected(kLiteRtStatusErrorInvalidArgument);
  }
}

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_OP_OPTIONS_H_
