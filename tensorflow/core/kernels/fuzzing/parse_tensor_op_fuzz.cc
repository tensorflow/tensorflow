/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tensorflow {
namespace fuzzing {

// Fuzz inputs to the serialized Tensor decoder.

class FuzzParseTensor : public FuzzSession {
  void BuildGraph(const Scope& scope) final {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    // The serialized proto.
    auto input = Placeholder(scope.WithOpName("input1"), DT_STRING);

    std::ignore = ParseTensor(scope.WithOpName("output"), input, DT_FLOAT);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<string>()() =
        string(reinterpret_cast<const char*>(data), size);
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    RunOneInput(input_tensor).IgnoreError();
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzParseTensor);

}  // end namespace fuzzing
}  // end namespace tensorflow
