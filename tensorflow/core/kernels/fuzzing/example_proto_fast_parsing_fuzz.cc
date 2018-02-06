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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Fuzz inputs to the example proto decoder.
// TODO(dga):  Make this more comprehensive.
// Right now, it's just a quick PoC to show how to attach the
// plumbing, but it needs some real protos to chew on as a
// corpus, and the sparse/dense parts should be made more rich
// to achieve higher code coverage.

class FuzzExampleProtoFastParsing : public FuzzSession {
  void BuildGraph(const Scope& scope) final {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    // The serialized proto.
    auto input = Placeholder(scope.WithOpName("input1"), DT_STRING);

    auto in_expanded = ExpandDims(scope, input, Const<int>(scope, 0));

    auto names = Const(scope, {"noname"});
    std::vector<Output> dense_keys = {Const(scope, {"a"})};
    std::vector<Output> sparse_keys;  // Empty.
    std::vector<Output> dense_defaults = {Const(scope, {1.0f})};

    DataTypeSlice sparse_types = {};
    std::vector<PartialTensorShape> dense_shapes;
    dense_shapes.push_back(PartialTensorShape());

    std::ignore = ParseExample(scope.WithOpName("output"), in_expanded, names,
                               sparse_keys, dense_keys, dense_defaults,
                               sparse_types, dense_shapes);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
    // TODO(dga):  Test the batch case also.
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<string>()() =
        string(reinterpret_cast<const char*>(data), size);
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    RunOneInput(input_tensor).IgnoreError();
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzExampleProtoFastParsing);

}  // end namespace fuzzing
}  // end namespace tensorflow
