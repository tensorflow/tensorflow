/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzStringSplitV2 : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto separator =
        tensorflow::ops::Placeholder(scope.WithOpName("separator"), DT_STRING);
    (void)tensorflow::ops::StringSplitV2(scope.WithOpName("output"),
                                               input, separator);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    Tensor separator_tensor(tensorflow::DT_STRING, TensorShape({}));

    if (size > 0) {
      // The spec for split is that the separator should be 0 or 1 characters.
      // Naturally, fuzz it with something that might be larger. But don't split
      // on a separator that is too large. Let's say we're picking a separator
      // of size 0, 1, 2 up to MaxSepSize (a static limit that has been picked
      // arbitrarily).
      size_t sep_len = static_cast<size_t>(data[0]) % kMaxSepSize;

      // We still have to handle the case when fuzzing input is shorter than the
      // minimum length required to get the separator
      if (sep_len > size) {
        sep_len = size - 1;
      }
      separator_tensor.scalar<string>()() =
          string(reinterpret_cast<const char*>(data), sep_len);
      input_tensor.scalar<string>()() = string(
          reinterpret_cast<const char*>(data + sep_len), size - sep_len);
    }

    RunInputs({{"input", input_tensor}, {"separator", separator_tensor}});
  }

 private:
  static const size_t kMaxSepSize = 4;
};

STANDARD_TF_FUZZ_FUNCTION(FuzzStringSplitV2);

}  // end namespace fuzzing
}  // end namespace tensorflow
