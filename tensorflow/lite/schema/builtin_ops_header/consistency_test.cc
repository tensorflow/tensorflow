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
#include "tensorflow/lite/schema/builtin_ops_header/generator.h"

namespace {

const char* kHeaderFileName =
    "tensorflow/lite/builtin_ops.h";

// The test ensures that `builtin_ops.h` is consistent with the FlatBuffer
// schema definition. When the schema is modified, it's required to run the
// generator to re-generate the header.
// Please see README.md for more details.
TEST(BuiltinOpsHeaderTest, TestConsistency) {
  std::ifstream input_stream(kHeaderFileName, std::ios::binary);
  ASSERT_TRUE(input_stream);
  std::string file_content((std::istreambuf_iterator<char>(input_stream)),
                           std::istreambuf_iterator<char>());

  std::ostringstream output_stream;
  tflite::builtin_ops_header::GenerateHeader(output_stream);
  std::string generated_content = output_stream.str();

  EXPECT_EQ(file_content, generated_content);
}

}  // anonymous namespace
