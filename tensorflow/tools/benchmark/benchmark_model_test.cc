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

#include "tensorflow/tools/benchmark/benchmark_model.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

TEST(BenchmarkModelTest, InitializeAndRun) {
  const string dir = testing::TmpDir();
  const string filename_pb = io::JoinPath(dir, "graphdef.pb");

  // Create a simple graph and write it to filename_pb.
  const int input_width = 400;
  const int input_height = 10;
  const TensorShape input_shape({input_width, input_height});
  const TensorShape constant_shape({input_height, input_width});

  Tensor constant_tensor(DT_FLOAT, constant_shape);
  test::FillFn<float>(&constant_tensor, [](int) -> float { return 3.0; });

  GraphDefBuilder b(GraphDefBuilder::kFailImmediately);
  Node* placeholder =
      ops::Placeholder(DT_FLOAT, b.opts().WithAttr("shape", input_shape));
  const string input_name = placeholder->name();
  Node* constant = ops::Const(constant_tensor, b.opts());
  const string output_name =
      ops::MatMul(placeholder, constant, b.opts())->name();

  GraphDef graph_def;
  TF_ASSERT_OK(b.ToGraphDef(&graph_def));
  string graph_def_serialized;
  graph_def.SerializeToString(&graph_def_serialized);
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), filename_pb, graph_def_serialized));

  std::unique_ptr<Session> session;
  std::unique_ptr<StatSummarizer> stats;
  TF_ASSERT_OK(
      benchmark_model::InitializeSession(1, filename_pb, &session, &stats));

  TF_ASSERT_OK(benchmark_model::TimeMultipleRuns(0.0, 10, DT_FLOAT, input_shape,
                                                 input_name, output_name,
                                                 session.get(), stats.get()));
}

}  // namespace
}  // namespace tensorflow
