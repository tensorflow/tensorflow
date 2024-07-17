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

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzCheckNumerics : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);
    auto prefix = "Error: ";
    (void)tensorflow::ops::CheckNumerics(scope.WithOpName("output"), input,
                                         prefix);
  }

  void FuzzImpl(const uint8_t* data, size_t size) override {
    size_t ratio = sizeof(float) / sizeof(uint8_t);
    size_t num_floats = size / ratio;
    const float* float_data = reinterpret_cast<const float*>(data);

    Tensor input_tensor(tensorflow::DT_FLOAT,
                        TensorShape({static_cast<int64_t>(num_floats)}));
    auto flat_tensor = input_tensor.flat<float>();
    for (size_t i = 0; i < num_floats; i++) {
      flat_tensor(i) = float_data[i];
    }
    RunInputs({{"input", input_tensor}});
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzCheckNumerics);

}  // end namespace fuzzing
}  // end namespace tensorflow
