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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/tools/accuracy/utils.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {
tensorflow::string* g_test_model_file = nullptr;
}

namespace tensorflow {
namespace metrics {
namespace utils {
namespace {

TEST(UtilsTest, GetTFLiteModelInfoReturnsCorrectly) {
  ASSERT_TRUE(g_test_model_file != nullptr);
  string test_model_file = *g_test_model_file;
  ASSERT_FALSE(test_model_file.empty());
  // Passed graph has 4 inputs : a,b,c,d and 2 outputs x,y
  //  x = a+b+c, y=b+c+d
  // Input and outputs have shape : {1,8,8,3}
  ModelInfo model_info;
  auto status = GetTFliteModelInfo(test_model_file, &model_info);
  TF_CHECK_OK(status);
  ASSERT_EQ(4, model_info.input_shapes.size());
  ASSERT_EQ(4, model_info.input_types.size());

  for (int i = 0; i < 4; i++) {
    const TensorShape& shape = model_info.input_shapes[i];
    DataType dataType = model_info.input_types[i];
    EXPECT_TRUE(shape.IsSameSize({1, 8, 8, 3}));
    EXPECT_EQ(DT_FLOAT, dataType);
  }
}

TEST(UtilsTest, GetTFliteModelInfoIncorrectFile) {
  ModelInfo model_info;
  auto status = GetTFliteModelInfo("non_existent_file", &model_info);
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace utils
}  // namespace metrics
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
