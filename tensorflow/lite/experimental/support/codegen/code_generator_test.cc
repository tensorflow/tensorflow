/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/support/codegen/code_generator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace support {
namespace codegen {
namespace {

using ::testing::ElementsAreArray;

class CodeGeneratorTest : public ::testing::Test {
 public:
  class TestingCodeGenerator : public CodeGenerator {
   public:
    explicit TestingCodeGenerator() : CodeGenerator() {}

    // Make tested method public.
    static std::string ConvertToValidName(const std::string& name) {
      return CodeGenerator::ConvertToValidName(name);
    }
    static void ResolveConflictedInputAndOutputNames(
        std::vector<std::string>* input, std::vector<std::string>* output) {
      CodeGenerator::ResolveConflictedInputAndOutputNames(input, output);
    }
  };
};

TEST_F(CodeGeneratorTest, UpperCasesShouldLower) {
  EXPECT_THAT(TestingCodeGenerator::ConvertToValidName("AlphaBetCOOL"),
              "alphabetcool");
}

TEST_F(CodeGeneratorTest, NonAlphaNumShouldReplace) {
  EXPECT_THAT(TestingCodeGenerator::ConvertToValidName("A+=B C\t"), "a__b_c_");
}

TEST_F(CodeGeneratorTest, NoLeadingUnderscore) {
  EXPECT_THAT(TestingCodeGenerator::ConvertToValidName("+KAI Z"), "kai_z");
}

TEST_F(CodeGeneratorTest, NoLeadingNumbers) {
  EXPECT_THAT(TestingCodeGenerator::ConvertToValidName("3000 Cool Tensors"),
              "tensor_3000_cool_tensors");
}

TEST_F(CodeGeneratorTest, TestSimpleIONames) {
  std::vector<std::string> inputs = {"image"};
  std::vector<std::string> outputs = {"output"};
  TestingCodeGenerator::ResolveConflictedInputAndOutputNames(&inputs, &outputs);
  EXPECT_THAT(inputs, ElementsAreArray({"image"}));
  EXPECT_THAT(outputs, ElementsAreArray({"output"}));
}

TEST_F(CodeGeneratorTest, TestIOConflict) {
  std::vector<std::string> inputs = {"image"};
  std::vector<std::string> outputs = {"image"};
  TestingCodeGenerator::ResolveConflictedInputAndOutputNames(&inputs, &outputs);
  EXPECT_THAT(inputs, ElementsAreArray({"input_image"}));
  EXPECT_THAT(outputs, ElementsAreArray({"output_image"}));
}

TEST_F(CodeGeneratorTest, TestInternalConflict) {
  std::vector<std::string> inputs = {"image", "image"};
  std::vector<std::string> outputs = {"output"};
  TestingCodeGenerator::ResolveConflictedInputAndOutputNames(&inputs, &outputs);
  EXPECT_THAT(inputs, ElementsAreArray({"image1", "image2"}));
  EXPECT_THAT(outputs, ElementsAreArray({"output"}));
}

TEST_F(CodeGeneratorTest, TestAllConflictNTo1) {
  std::vector<std::string> inputs = {"image", "image"};
  std::vector<std::string> outputs = {"image"};
  TestingCodeGenerator::ResolveConflictedInputAndOutputNames(&inputs, &outputs);
  EXPECT_THAT(inputs, ElementsAreArray({"input_image1", "input_image2"}));
  EXPECT_THAT(outputs, ElementsAreArray({"output_image"}));
}

TEST_F(CodeGeneratorTest, TestAllConflict) {
  std::vector<std::string> inputs = {"image", "audio", "image", "audio",
                                     "audio"};
  std::vector<std::string> outputs = {"image", "image", "audio", "feature",
                                      "feature"};
  TestingCodeGenerator::ResolveConflictedInputAndOutputNames(&inputs, &outputs);
  EXPECT_THAT(inputs,
              ElementsAreArray({"input_image1", "input_audio1", "input_image2",
                                "input_audio2", "input_audio3"}));
  EXPECT_THAT(outputs,
              ElementsAreArray({"output_image1", "output_image2",
                                "output_audio", "feature1", "feature2"}));
}

TEST_F(CodeGeneratorTest, TestAllConflictReversed) {
  std::vector<std::string> inputs = {"image", "image", "audio", "feature",
                                     "feature"};
  std::vector<std::string> outputs = {"image", "audio", "image", "audio",
                                      "audio"};
  TestingCodeGenerator::ResolveConflictedInputAndOutputNames(&inputs, &outputs);
  EXPECT_THAT(inputs,
              ElementsAreArray({"input_image1", "input_image2", "input_audio",
                                "feature1", "feature2"}));
  EXPECT_THAT(outputs, ElementsAreArray({"output_image1", "output_audio1",
                                         "output_image2", "output_audio2",
                                         "output_audio3"}));
}

}  // namespace
}  // namespace codegen
}  // namespace support
}  // namespace tflite
