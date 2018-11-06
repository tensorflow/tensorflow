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

#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/tools/accuracy/ilsvrc/inception_preprocessing.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {
tensorflow::string* g_test_image_file = nullptr;
}  // namespace

namespace tensorflow {
namespace metrics {

namespace {

using tensorflow::Status;
using tensorflow::Tensor;

Status GetContents(const string& filename, string* output) {
  std::ifstream input(filename, std::ios::binary);
  const int kBufferSize = 2048;
  char buffer[kBufferSize];
  while (true) {
    input.read(buffer, kBufferSize);
    output->append(buffer, input.gcount());
    if (!input.good()) {
      if (input.eof()) return Status::OK();
      return Status(tensorflow::error::ABORTED, "Failed to read file.");
    }
  }
}

TEST(InceptionPreprocessingTest, TestImagePreprocessQuantized) {
  ASSERT_TRUE(g_test_image_file != nullptr);
  string image_contents;
  string image_path = *g_test_image_file;
  auto status = GetContents(image_path, &image_contents);
  ASSERT_TRUE(status.ok()) << status.error_message();
  const int width = 224;
  const int height = 224;
  const bool is_quantized = true;
  InceptionPreprocessingStage preprocess_stage(width, height, is_quantized);
  Scope scope = Scope::NewRootScope();
  preprocess_stage.AddToGraph(scope, image_contents);
  TF_CHECK_OK(scope.status());

  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));
  std::vector<Tensor> outputs;
  auto run_status =
      session->Run({},                                   /*inputs*/
                   {preprocess_stage.output_name()}, {}, /*target node names */
                   &outputs);
  TF_CHECK_OK(run_status);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DT_UINT8, outputs[0].dtype());
  EXPECT_TRUE(outputs[0].shape().IsSameSize({1, 224, 224, 3}));
}

TEST(InceptionPreprocessingTest, TestImagePreprocessFloat) {
  ASSERT_TRUE(g_test_image_file != nullptr);
  string image_contents;
  string image_path = *g_test_image_file;
  auto status = GetContents(image_path, &image_contents);
  ASSERT_TRUE(status.ok()) << status.error_message();
  const int width = 224;
  const int height = 224;
  const bool is_quantized = false;
  InceptionPreprocessingStage preprocess_stage(width, height, is_quantized);
  Scope scope = Scope::NewRootScope();
  preprocess_stage.AddToGraph(scope, image_contents);
  TF_CHECK_OK(scope.status());

  GraphDef graph_def;
  TF_CHECK_OK(scope.ToGraphDef(&graph_def));
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));
  std::vector<Tensor> outputs;
  auto run_status =
      session->Run({},                                   /*inputs*/
                   {preprocess_stage.output_name()}, {}, /*target node names */
                   &outputs);
  TF_CHECK_OK(run_status);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DT_FLOAT, outputs[0].dtype());
  EXPECT_TRUE(outputs[0].shape().IsSameSize({1, 224, 224, 3}));
}

}  // namespace
}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char** argv) {
  g_test_image_file = new tensorflow::string();
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_image", g_test_image_file,
                       "Path to image file for test."),
  };
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  CHECK(parse_result) << "Required test_model_file";
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
