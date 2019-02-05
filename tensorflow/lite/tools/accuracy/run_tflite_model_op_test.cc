/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {
tensorflow::string* g_test_model_file = nullptr;
}

namespace tensorflow {
namespace {

TEST(RunTfliteModelOpTest, ModelIsRun) {
  ASSERT_TRUE(g_test_model_file != nullptr);
  string test_model_file = *g_test_model_file;
  ASSERT_FALSE(test_model_file.empty());

  Scope scope = Scope::NewRootScope();
  TF_CHECK_OK(scope.status());
  // Passed graph has 4 inputs : a,b,c,d and 2 outputs x,y
  //  x = a+b+c, y=b+c+d

  std::vector<Input> graph_inputs = {
      ops::Const(scope, 1.0f, {1, 8, 8, 3}),  // a
      ops::Const(scope, 2.1f, {1, 8, 8, 3}),  // b
      ops::Const(scope, 3.2f, {1, 8, 8, 3}),  // c
      ops::Const(scope, 4.3f, {1, 8, 8, 3}),  // d
  };

  std::vector<NodeBuilder::NodeOut> input_data;
  std::transform(graph_inputs.begin(), graph_inputs.end(),
                 std::back_inserter(input_data), [&scope](Input model_input) {
                   return ops::AsNodeOut(scope, model_input);
                 });

  std::vector<DataType> model_input_type = {DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                            DT_FLOAT};
  ::tensorflow::Node* ret;
  auto builder = ::tensorflow::NodeBuilder("run_model_op", "RunTFLiteModel")
                     .Input(input_data)
                     .Attr("model_file_path", test_model_file)
                     .Attr("input_type", model_input_type)
                     .Attr("output_type", {DT_FLOAT, DT_FLOAT});

  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  TF_CHECK_OK(scope.status());

  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));

  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<Tensor> outputs;
  TF_CHECK_OK(
      session->Run({}, {"run_model_op:0", "run_model_op:1"}, {}, &outputs));
  EXPECT_EQ(2, outputs.size());

  for (const auto& tensor : outputs) {
    EXPECT_TRUE(tensor.shape().IsSameSize({1, 8, 8, 3}));
  }
  auto output_x = outputs[0].flat<float>();
  auto output_y = outputs[1].flat<float>();
  EXPECT_EQ(1 * 8 * 8 * 3, output_x.size());
  EXPECT_EQ(1 * 8 * 8 * 3, output_y.size());
  for (int i = 0; i < output_x.size(); i++) {
    EXPECT_NEAR(6.3f, output_x(i), 1e-6f);  // a+b+c
    EXPECT_NEAR(9.6f, output_y(i), 1e-6f);  // b+c+d
  }
}

TEST(RunTfliteModelOpTest, NumInputsMismatch) {
  ASSERT_TRUE(g_test_model_file != nullptr);
  string test_model_file = *g_test_model_file;
  ASSERT_FALSE(test_model_file.empty());

  Scope scope = Scope::NewRootScope();
  TF_CHECK_OK(scope.status());
  // Passed graph has 4 inputs : a,b,c,d and 2 outputs x,y
  //  x = a+b+c, y=b+c+d
  //  Remove a from input.

  std::vector<Input> graph_inputs = {
      ops::Const(scope, 2.1f, {1, 8, 8, 3}),  // b
      ops::Const(scope, 3.2f, {1, 8, 8, 3}),  // c
      ops::Const(scope, 4.3f, {1, 8, 8, 3}),  // d
  };

  std::vector<NodeBuilder::NodeOut> input_data;
  std::transform(graph_inputs.begin(), graph_inputs.end(),
                 std::back_inserter(input_data), [&scope](Input model_input) {
                   return ops::AsNodeOut(scope, model_input);
                 });

  std::vector<DataType> model_input_type = {DT_FLOAT, DT_FLOAT, DT_FLOAT};

  ::tensorflow::Node* ret;
  auto builder = ::tensorflow::NodeBuilder("run_model_op", "RunTFLiteModel")
                     .Input(input_data)
                     .Attr("model_file_path", test_model_file)
                     .Attr("input_type", model_input_type)
                     .Attr("output_type", {DT_FLOAT, DT_FLOAT});

  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  TF_CHECK_OK(scope.status());

  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<Tensor> outputs;
  auto status =
      (session->Run({}, {"run_model_op:0", "run_model_op:1"}, {}, &outputs));
  EXPECT_FALSE(status.ok());
}

TEST(RunTfliteModelOpTest, InputSizesMismatch) {
  ASSERT_TRUE(g_test_model_file != nullptr);
  string test_model_file = *g_test_model_file;
  ASSERT_FALSE(test_model_file.empty());

  Scope scope = Scope::NewRootScope();
  TF_CHECK_OK(scope.status());
  // Passed graph has 4 inputs : a,b,c,d and 2 outputs x,y
  //  x = a+b+c, y=b+c+d
  // Set a to be invalid size.
  std::vector<Input> graph_inputs = {
      ops::Const(scope, 1.0f, {1, 8, 8, 4}),  // a invalid size,
      ops::Const(scope, 2.1f, {1, 8, 8, 3}),  // b
      ops::Const(scope, 3.2f, {1, 8, 8, 3}),  // c
      ops::Const(scope, 4.3f, {1, 8, 8, 3}),  // d
  };

  std::vector<NodeBuilder::NodeOut> input_data;
  std::transform(graph_inputs.begin(), graph_inputs.end(),
                 std::back_inserter(input_data), [&scope](Input model_input) {
                   return ops::AsNodeOut(scope, model_input);
                 });

  std::vector<DataType> model_input_type = {DT_FLOAT, DT_FLOAT, DT_FLOAT,
                                            DT_FLOAT};
  ::tensorflow::Node* ret;
  auto builder = ::tensorflow::NodeBuilder("run_model_op", "RunTFLiteModel")
                     .Input(input_data)
                     .Attr("model_file_path", test_model_file)
                     .Attr("input_type", model_input_type)
                     .Attr("output_type", {DT_FLOAT, DT_FLOAT});

  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  TF_CHECK_OK(scope.status());

  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  std::vector<Tensor> outputs;
  auto status =
      (session->Run({}, {"run_model_op:0", "run_model_op:1"}, {}, &outputs));
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  g_test_model_file = new tensorflow::string();
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_model_file", g_test_model_file,
                       "Path to test tflite model file."),
  };
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  CHECK(parse_result) << "Required test_model_file";
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
