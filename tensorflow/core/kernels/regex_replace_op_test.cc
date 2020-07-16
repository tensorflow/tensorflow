/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

// Test data from the TensorFlow README.md.
const char* lines[] = {
    "**TensorFlow** is an open source software library for numerical "
    "computation using data flow graphs.",
    "The graph nodes represent mathematical operations, while the graph edges "
    "represent the multidimensional data arrays (tensors) that flow between "
    "them.",
    "This flexible architecture enables you to deploy computation to one or "
    "more CPUs or GPUs in a desktop, server, or mobile device without "
    "rewriting code.",
    "TensorFlow also includes "
    "[TensorBoard](https://www.tensorflow.org/guide/"
    "summaries_and_tensorboard), a data visualization toolkit.",
    "TensorFlow was originally developed by researchers and engineers working "
    "on the Google Brain team within Google's Machine Intelligence Research "
    "organization for the purposes of conducting machine learning and deep "
    "neural networks research.",
    "The system is general enough to be applicable in a wide variety of other "
    "domains, as well.",
    "TensorFlow provides stable Python API and C APIs as well as without API "
    "backwards compatibility guarantee like C++, Go, Java, JavaScript and "
    "Swift."};

const char kRegExPattern[] = "\\p{P}";
const char kRewrite[] = " ";

Tensor GetTestTensor(int batch) {
  const int sz = TF_ARRAYSIZE(lines);
  Tensor t(DT_STRING, {batch});
  auto s = t.flat<tstring>();
  for (int i = 0; i < batch; ++i) {
    s(i) = lines[i % sz];
  }
  return t;
}

Graph* SetupRegexReplaceGraph(const Tensor& input, const string& input_pattern,
                              const string& input_rewrite) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor pattern(DT_STRING, TensorShape({}));
  pattern.flat<tstring>().setConstant(input_pattern);
  Tensor rewrite(DT_STRING, TensorShape({}));
  rewrite.flat<tstring>().setConstant(input_rewrite);

  TF_CHECK_OK(NodeBuilder("regex_replace_op", "RegexReplace")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, pattern))
                  .Input(test::graph::Constant(g, rewrite))
                  .Attr("replace_global", true)
                  .Finalize(g, nullptr /* node */));
  return g;
}

void BM_RegexReplace(int iters, int batch_size) {
  testing::StopTiming();
  testing::ItemsProcessed(static_cast<int64>(iters));
  testing::UseRealTime();
  Tensor input = GetTestTensor(batch_size);
  Graph* g = SetupRegexReplaceGraph(input, kRegExPattern, kRewrite);
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
}

BENCHMARK(BM_RegexReplace)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

Graph* SetupStaticGraph(const Tensor& input, const string& input_pattern,
                        const string& rewrite) {
  Graph* g = new Graph(OpRegistry::Global());

  TF_CHECK_OK(NodeBuilder("static_regex_replace_op", "StaticRegexReplace")
                  .Attr("pattern", input_pattern)
                  .Attr("rewrite", rewrite)
                  .Input(test::graph::Constant(g, input))
                  .Attr("replace_global", true)
                  .Finalize(g, nullptr /* node */));
  return g;
}
void BM_StaticRegexReplace(int iters, int batch_size) {
  testing::StopTiming();
  testing::ItemsProcessed(static_cast<int64>(iters));
  testing::UseRealTime();
  Tensor input = GetTestTensor(batch_size);
  Graph* g = SetupStaticGraph(input, kRegExPattern, kRewrite);
  testing::StartTiming();
  test::Benchmark("cpu", g).Run(iters);
}

BENCHMARK(BM_StaticRegexReplace)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

}  // end namespace tensorflow
