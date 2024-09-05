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
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace fuzzing {

// Don't generate tensors that are too large as we don't test that branch here
constexpr size_t kMaxSize = 1024;

class FuzzOneHot : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_UINT8);
    auto depth =
        tensorflow::ops::Placeholder(scope.WithOpName("depth"), DT_INT32);
    auto on = tensorflow::ops::Placeholder(scope.WithOpName("on"), DT_UINT8);
    auto off = tensorflow::ops::Placeholder(scope.WithOpName("off"), DT_UINT8);
    (void)tensorflow::ops::OneHot(scope.WithOpName("output"), input, depth, on,
                                  off);
  }

  void FuzzImpl(const uint8_t* data, size_t size) override {
    int64_t input_size;
    int32_t depth;
    uint8 on, off;
    const uint8_t* input_data;

    if (size > 3) {
      // Since we only care about the one hot decoding and not about the size of
      // the tensor, limit `size` to at most `kMaxSize`.
      if (size > kMaxSize) {
        size = kMaxSize;
      }
      depth = static_cast<int32>(data[0]);
      on = data[1];
      off = data[2];
      input_size = static_cast<int64_t>(size - 3);
      input_data = data + 3;
    } else {
      depth = 1;
      on = 1;
      off = 0;
      input_size = static_cast<int64_t>(size);
      input_data = data;
    }

    Tensor input_tensor(tensorflow::DT_UINT8, TensorShape({input_size}));
    Tensor depth_tensor(tensorflow::DT_INT32, TensorShape({}));
    Tensor on_tensor(tensorflow::DT_UINT8, TensorShape({}));
    Tensor off_tensor(tensorflow::DT_UINT8, TensorShape({}));

    auto flat_tensor = input_tensor.flat<uint8>();
    for (size_t i = 0; i < input_size; i++) {
      flat_tensor(i) = input_data[i];
    }
    depth_tensor.scalar<int32>()() = depth;
    on_tensor.scalar<uint8>()() = on;
    off_tensor.scalar<uint8>()() = off;

    RunInputs({{"input", input_tensor},
               {"depth", depth_tensor},
               {"on", on_tensor},
               {"off", off_tensor}});
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzOneHot);

}  // end namespace fuzzing
}  // end namespace tensorflow
