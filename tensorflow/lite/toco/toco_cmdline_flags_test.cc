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

#include "tensorflow/lite/toco/toco_cmdline_flags.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace toco {
namespace {

TEST(TocoCmdlineFlagsTest, DefaultValue) {
  int argc = 1;
  // Invariant in ANSI C, len(argv) == argc +1 also argv[argc] == nullptr
  // TF flag parsing lib is relaying on this invariant.
  const char* args[] = {"toco", nullptr};

  string message;
  ParsedTocoFlags result_flags;

  EXPECT_TRUE(ParseTocoFlagsFromCommandLineFlags(
      &argc, const_cast<char**>(args), &message, &result_flags));
  EXPECT_EQ(result_flags.allow_dynamic_tensors.value(), true);
}

TEST(TocoCmdlineFlagsTest, ParseFlags) {
  int argc = 2;
  const char* args[] = {"toco", "--allow_dynamic_tensors=false", nullptr};

  string message;
  ParsedTocoFlags result_flags;

  EXPECT_TRUE(ParseTocoFlagsFromCommandLineFlags(
      &argc, const_cast<char**>(args), &message, &result_flags));
  EXPECT_EQ(result_flags.allow_dynamic_tensors.value(), false);
}

}  // namespace
}  // namespace toco

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
