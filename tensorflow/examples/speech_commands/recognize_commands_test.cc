/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/examples/speech_commands/recognize_commands.h"

#include <cstdint>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

TEST(RecognizeCommandsTest, Basic) {
  RecognizeCommands recognize_commands({"_silence_", "a", "b"});

  Tensor results(DT_FLOAT, {3});
  test::FillValues<float>(&results, {1.0f, 0.0f, 0.0f});

  string found_command;
  float score;
  bool is_new_command;
  TF_EXPECT_OK(recognize_commands.ProcessLatestResults(
      results, 0, &found_command, &score, &is_new_command));
}

TEST(RecognizeCommandsTest, FindCommands) {
  RecognizeCommands recognize_commands({"_silence_", "a", "b"}, 1000, 0.2f);

  Tensor results(DT_FLOAT, {3});

  test::FillValues<float>(&results, {0.0f, 1.0f, 0.0f});
  bool has_found_new_command = false;
  string new_command;
  for (int i = 0; i < 10; ++i) {
    string found_command;
    float score;
    bool is_new_command;
    int64_t current_time_ms = 0 + (i * 100);
    TF_EXPECT_OK(recognize_commands.ProcessLatestResults(
        results, current_time_ms, &found_command, &score, &is_new_command));
    if (is_new_command) {
      EXPECT_FALSE(has_found_new_command);
      has_found_new_command = true;
      new_command = found_command;
    }
  }
  EXPECT_TRUE(has_found_new_command);
  EXPECT_EQ("a", new_command);

  test::FillValues<float>(&results, {0.0f, 0.0f, 1.0f});
  has_found_new_command = false;
  new_command = "";
  for (int i = 0; i < 10; ++i) {
    string found_command;
    float score;
    bool is_new_command;
    int64_t current_time_ms = 1000 + (i * 100);
    TF_EXPECT_OK(recognize_commands.ProcessLatestResults(
        results, current_time_ms, &found_command, &score, &is_new_command));
    if (is_new_command) {
      EXPECT_FALSE(has_found_new_command);
      has_found_new_command = true;
      new_command = found_command;
    }
  }
  EXPECT_TRUE(has_found_new_command);
  EXPECT_EQ("b", new_command);
}

TEST(RecognizeCommandsTest, BadInputLength) {
  RecognizeCommands recognize_commands({"_silence_", "a", "b"}, 1000, 0.2f);

  Tensor bad_results(DT_FLOAT, {2});
  test::FillValues<float>(&bad_results, {1.0f, 0.0f});

  string found_command;
  float score;
  bool is_new_command;
  EXPECT_FALSE(recognize_commands
                   .ProcessLatestResults(bad_results, 0, &found_command, &score,
                                         &is_new_command)
                   .ok());
}

TEST(RecognizeCommandsTest, BadInputTimes) {
  RecognizeCommands recognize_commands({"_silence_", "a", "b"}, 1000, 0.2f);

  Tensor results(DT_FLOAT, {3});
  test::FillValues<float>(&results, {1.0f, 0.0f, 0.0f});

  string found_command;
  float score;
  bool is_new_command;
  TF_EXPECT_OK(recognize_commands.ProcessLatestResults(
      results, 100, &found_command, &score, &is_new_command));
  EXPECT_FALSE(recognize_commands
                   .ProcessLatestResults(results, 0, &found_command, &score,
                                         &is_new_command)
                   .ok());
}

}  // namespace tensorflow
