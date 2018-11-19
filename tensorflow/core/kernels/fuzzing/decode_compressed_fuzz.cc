/* Copyright 2018 Google Inc. All Rights Reserved.

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

class FuzzDecodeCompressed : public FuzzStringInputOp {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input1"), DT_STRING);
    auto d1 = tensorflow::ops::DecodeCompressed(
        scope.WithOpName("d1"), input,
        tensorflow::ops::DecodeCompressed::CompressionType(""));
    auto d2 = tensorflow::ops::DecodeCompressed(
        scope.WithOpName("d2"), input,
        tensorflow::ops::DecodeCompressed::CompressionType("ZLIB"));
    auto d3 = tensorflow::ops::DecodeCompressed(
        scope.WithOpName("d3"), input,
        tensorflow::ops::DecodeCompressed::CompressionType("GZIP"));
    Scope grouper =
        scope.WithControlDependencies(std::vector<tensorflow::Operation>{
            d1.output.op(), d2.output.op(), d3.output.op()});
    (void)tensorflow::ops::NoOp(grouper.WithOpName("output"));
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzDecodeCompressed);

}  // namespace fuzzing
}  // namespace tensorflow
