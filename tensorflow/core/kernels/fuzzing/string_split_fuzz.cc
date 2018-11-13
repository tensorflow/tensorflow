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

class FuzzStringSplit : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input1"), DT_STRING);
    auto delimeter =
        tensorflow::ops::Placeholder(scope.WithOpName("input2"), DT_STRING);
    (void)tensorflow::ops::StringSplit(scope.WithOpName("output"), input,
                                       delimeter);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    Tensor delimeter_tensor(tensorflow::DT_STRING, TensorShape({}));

    if (size > 0) {
      // The spec for split is that the delimeter should be 0 or 1 characters.
      // Naturally, fuzz it with something larger.  (This omits the possibility
      // of handing it a > int32_max size string, which should be tested for in
      // an
      // explicit test).
      size_t delim_len = static_cast<size_t>(data[0]);
      if (delim_len > size) {
        delim_len = size - 1;
      }
      delimeter_tensor.scalar<string>()() =
          string(reinterpret_cast<const char*>(data), delim_len);
      input_tensor.scalar<string>()() = string(
          reinterpret_cast<const char*>(data + delim_len), size - delim_len);
    }

    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    RunTwoInputs(input_tensor, delimeter_tensor).IgnoreError();
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzStringSplit);

}  // end namespace fuzzing
}  // end namespace tensorflow
