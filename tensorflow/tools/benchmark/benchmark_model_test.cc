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

void CreateTestGraph(const ::tensorflow::Scope& root,
                     benchmark_model::InputLayerInfo* input,
                     string* output_name, GraphDef* graph_def) {
  // Create a simple graph and write it to filename_pb.
  const int input_width = 400;
  const int input_height = 10;
  input->shape = TensorShape({input_width, input_height});
  input->data_type = DT_FLOAT;
  const TensorShape constant_shape({input_height, input_width});

  Tensor constant_tensor(DT_FLOAT, constant_shape);
  test::FillFn<float>(&constant_tensor, [](int) -> float { return 3.0; });

  auto placeholder =
      ops::Placeholder(root, DT_FLOAT, ops::Placeholder::Shape(input->shape));
  input->name = placeholder.node()->name();
  auto m = ops::MatMul(root, placeholder, constant_tensor);
  *output_name = m.node()->name();
  TF_ASSERT_OK(root.ToGraphDef(graph_def));
}

TEST(BenchmarkModelTest, InitializeAndRun) {
  const string dir = testing::TmpDir();
  const string filename_pb = io::JoinPath(dir, "graphdef.pb");
  auto root = Scope::NewRootScope().ExitOnError();

  benchmark_model::InputLayerInfo input;
  string output_name;
  GraphDef graph_def;
  CreateTestGraph(root, &input, &output_name, &graph_def);
  string graph_def_serialized;
  graph_def.SerializeToString(&graph_def_serialized);
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), filename_pb, graph_def_serialized));

  std::unique_ptr<Session> session;
  std::unique_ptr<GraphDef> loaded_graph_def;
  TF_ASSERT_OK(benchmark_model::InitializeSession(1, filename_pb, &session,
                                                  &loaded_graph_def));
  std::unique_ptr<StatSummarizer> stats;
  stats.reset(new tensorflow::StatSummarizer(*(loaded_graph_def.get())));
  int64 time;
  int64 num_runs = 0;
  TF_ASSERT_OK(benchmark_model::TimeMultipleRuns(
      0.0, 10, 0.0, {input}, {output_name}, {}, session.get(), stats.get(),
      &time, &num_runs));
  ASSERT_EQ(num_runs, 10);
}

TEST(BenchmarkModeTest, TextProto) {
  const string dir = testing::TmpDir();
  const string filename_txt = io::JoinPath(dir, "graphdef.pb.txt");
  auto root = Scope::NewRootScope().ExitOnError();

  benchmark_model::InputLayerInfo input;
  string output_name;
  GraphDef graph_def;
  CreateTestGraph(root, &input, &output_name, &graph_def);
  TF_ASSERT_OK(WriteTextProto(Env::Default(), filename_txt, graph_def));

  std::unique_ptr<Session> session;
  std::unique_ptr<GraphDef> loaded_graph_def;
  TF_ASSERT_OK(benchmark_model::InitializeSession(1, filename_txt, &session,
                                                  &loaded_graph_def));
  std::unique_ptr<StatSummarizer> stats;
  stats.reset(new tensorflow::StatSummarizer(*(loaded_graph_def.get())));
  int64 time;
  int64 num_runs = 0;
  TF_ASSERT_OK(benchmark_model::TimeMultipleRuns(
      0.0, 10, 0.0, {input}, {output_name}, {}, session.get(), stats.get(),
      &time, &num_runs));
  ASSERT_EQ(num_runs, 10);
}

}  // namespace
}  // namespace tensorflow
