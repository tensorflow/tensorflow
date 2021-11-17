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
#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>

#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/security/fuzzing/op_fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzMatMul : public FuzzSession {
  void BuildGraph(const Scope& scope) final {
    // output = a * b
    auto a = tensorflow::ops::Placeholder(scope.WithOpName("a"), DT_INT32);
    auto b = tensorflow::ops::Placeholder(scope.WithOpName("b"), DT_INT32);
    (void)tensorflow::ops::MatMul(scope.WithOpName("output"), a, b);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
    FuzzedDataProvider fdp(data, size);

    // tensor a of shape {x, y}, tensor b of shape {z, t} (allow shape mismatch)
    const auto x = fdp.ConsumeIntegral<uint32_t>();
    const auto y = fdp.ConsumeIntegral<uint32_t>();
    const auto z = fdp.ConsumeIntegral<uint32_t>();
    const auto t = fdp.ConsumeIntegral<uint32_t>();

    // prevent out of memory in fuzzer
    const int32_t max_dimension = 1 << 14;
    if (x >= max_dimension || y >= max_dimension || z >= max_dimension ||
        t >= max_dimension) {
      return;
    }

    // fill in the tensors
    Tensor a(tensorflow::DT_INT32, TensorShape({x, y}));
    auto a_flat = a.flat<int32_t>();
    for (int i = 0; i < a.NumElements(); i++)
      a_flat(i) = fdp.ConsumeIntegral<uint8_t>();
    Tensor b(tensorflow::DT_INT32, TensorShape({z, t}));
    auto b_flat = b.flat<int32_t>();
    for (int i = 0; i < b.NumElements(); i++)
      b_flat(i) = fdp.ConsumeIntegral<uint8_t>();

    // Do the matrix multiply now
    RunInputs({{"a", a}, {"b", b}});
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzMatMul);

}  // end namespace fuzzing
}  // end namespace tensorflow
