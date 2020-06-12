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

#include "tensorflow/lite/micro/examples/micro_speech/recognize_commands.h"

#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(PreviousResultsQueueBasic) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  PreviousResultsQueue queue(error_reporter);
  TF_LITE_MICRO_EXPECT_EQ(0, queue.size());

  int8_t scores_a[4] = {0, 0, 0, 1};
  queue.push_back({0, scores_a});
  TF_LITE_MICRO_EXPECT_EQ(1, queue.size());
  TF_LITE_MICRO_EXPECT_EQ(0, queue.front().time_);
  TF_LITE_MICRO_EXPECT_EQ(0, queue.back().time_);

  int8_t scores_b[4] = {0, 0, 1, 0};
  queue.push_back({1, scores_b});
  TF_LITE_MICRO_EXPECT_EQ(2, queue.size());
  TF_LITE_MICRO_EXPECT_EQ(0, queue.front().time_);
  TF_LITE_MICRO_EXPECT_EQ(1, queue.back().time_);

  PreviousResultsQueue::Result pop_result = queue.pop_front();
  TF_LITE_MICRO_EXPECT_EQ(0, pop_result.time_);
  TF_LITE_MICRO_EXPECT_EQ(1, queue.size());
  TF_LITE_MICRO_EXPECT_EQ(1, queue.front().time_);
  TF_LITE_MICRO_EXPECT_EQ(1, queue.back().time_);

  int8_t scores_c[4] = {0, 1, 0, 0};
  queue.push_back({2, scores_c});
  TF_LITE_MICRO_EXPECT_EQ(2, queue.size());
  TF_LITE_MICRO_EXPECT_EQ(1, queue.front().time_);
  TF_LITE_MICRO_EXPECT_EQ(2, queue.back().time_);
}

TF_LITE_MICRO_TEST(PreviousResultsQueuePushPop) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  PreviousResultsQueue queue(error_reporter);
  TF_LITE_MICRO_EXPECT_EQ(0, queue.size());

  for (int i = 0; i < 123; ++i) {
    int8_t scores[4] = {0, 0, 0, 1};
    queue.push_back({i, scores});
    TF_LITE_MICRO_EXPECT_EQ(1, queue.size());
    TF_LITE_MICRO_EXPECT_EQ(i, queue.front().time_);
    TF_LITE_MICRO_EXPECT_EQ(i, queue.back().time_);

    PreviousResultsQueue::Result pop_result = queue.pop_front();
    TF_LITE_MICRO_EXPECT_EQ(i, pop_result.time_);
    TF_LITE_MICRO_EXPECT_EQ(0, queue.size());
  }
}

TF_LITE_MICRO_TEST(RecognizeCommandsTestBasic) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  RecognizeCommands recognize_commands(error_reporter);

  std::initializer_list<int8_t> result_data = {127, -128, -128, -128};
  auto result_dims = {2, 1, 4};
  TfLiteTensor results = tflite::testing::CreateQuantizedTensor(
      result_data, tflite::testing::IntArrayFromInitializer(result_dims),
      -128.0f, 127.0f);

  const char* found_command;
  uint8_t score;
  bool is_new_command;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, recognize_commands.ProcessLatestResults(
                     &results, 0, &found_command, &score, &is_new_command));
}

TF_LITE_MICRO_TEST(RecognizeCommandsTestFindCommands) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  RecognizeCommands recognize_commands(error_reporter, 1000, 51);

  std::initializer_list<int8_t> yes_data = {-128, -128, 127, -128};
  auto yes_dims = {2, 1, 4};
  TfLiteTensor yes_results = tflite::testing::CreateQuantizedTensor(
      yes_data, tflite::testing::IntArrayFromInitializer(yes_dims), -128.0f,
      127.0f);

  bool has_found_new_command = false;
  const char* new_command;
  for (int i = 0; i < 10; ++i) {
    const char* found_command;
    uint8_t score;
    bool is_new_command;
    int32_t current_time_ms = 0 + (i * 100);
    TF_LITE_MICRO_EXPECT_EQ(
        kTfLiteOk, recognize_commands.ProcessLatestResults(
                       &yes_results, current_time_ms, &found_command, &score,
                       &is_new_command));
    if (is_new_command) {
      TF_LITE_MICRO_EXPECT(!has_found_new_command);
      has_found_new_command = true;
      new_command = found_command;
    }
  }
  TF_LITE_MICRO_EXPECT(has_found_new_command);
  if (has_found_new_command) {
    TF_LITE_MICRO_EXPECT_EQ(0, tflite::testing::TestStrcmp("yes", new_command));
  }

  std::initializer_list<int8_t> no_data = {-128, -128, -128, 127};
  auto no_dims = {2, 1, 4};
  TfLiteTensor no_results = tflite::testing::CreateQuantizedTensor(
      no_data, tflite::testing::IntArrayFromInitializer(no_dims), -128.0f,
      127.0f);
  has_found_new_command = false;
  new_command = "";
  uint8_t score;
  for (int i = 0; i < 10; ++i) {
    const char* found_command;
    bool is_new_command;
    int32_t current_time_ms = 1000 + (i * 100);
    TF_LITE_MICRO_EXPECT_EQ(
        kTfLiteOk, recognize_commands.ProcessLatestResults(
                       &no_results, current_time_ms, &found_command, &score,
                       &is_new_command));
    if (is_new_command) {
      TF_LITE_MICRO_EXPECT(!has_found_new_command);
      has_found_new_command = true;
      new_command = found_command;
    }
  }
  TF_LITE_MICRO_EXPECT(has_found_new_command);
  if (has_found_new_command) {
    TF_LITE_MICRO_EXPECT_EQ(231, score);
    TF_LITE_MICRO_EXPECT_EQ(0, tflite::testing::TestStrcmp("no", new_command));
  }
}

TF_LITE_MICRO_TEST(RecognizeCommandsTestBadInputLength) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  RecognizeCommands recognize_commands(error_reporter, 1000, 51);

  std::initializer_list<int8_t> bad_data = {-128, -128, 127};
  auto bad_dims = {2, 1, 3};
  TfLiteTensor bad_results = tflite::testing::CreateQuantizedTensor(
      bad_data, tflite::testing::IntArrayFromInitializer(bad_dims), -128.0f,
      127.0f);

  const char* found_command;
  uint8_t score;
  bool is_new_command;
  TF_LITE_MICRO_EXPECT_NE(
      kTfLiteOk, recognize_commands.ProcessLatestResults(
                     &bad_results, 0, &found_command, &score, &is_new_command));
}

TF_LITE_MICRO_TEST(RecognizeCommandsTestBadInputTimes) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  RecognizeCommands recognize_commands(error_reporter, 1000, 51);

  std::initializer_list<int8_t> result_data = {-128, -128, 127, -128};
  auto result_dims = {2, 1, 4};
  TfLiteTensor results = tflite::testing::CreateQuantizedTensor(
      result_data, tflite::testing::IntArrayFromInitializer(result_dims),
      -128.0f, 127.0f);

  const char* found_command;
  uint8_t score;
  bool is_new_command;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, recognize_commands.ProcessLatestResults(
                     &results, 100, &found_command, &score, &is_new_command));
  TF_LITE_MICRO_EXPECT_NE(
      kTfLiteOk, recognize_commands.ProcessLatestResults(
                     &results, 0, &found_command, &score, &is_new_command));
}

TF_LITE_MICRO_TEST(RecognizeCommandsTestTooFewInputs) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  RecognizeCommands recognize_commands(error_reporter, 1000, 51);

  std::initializer_list<int8_t> result_data = {-128, -128, 127, -128};
  auto result_dims = {2, 1, 4};
  TfLiteTensor results = tflite::testing::CreateQuantizedTensor(
      result_data, tflite::testing::IntArrayFromInitializer(result_dims),
      -128.0f, 127.0f);

  const char* found_command;
  uint8_t score;
  bool is_new_command;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, recognize_commands.ProcessLatestResults(
                     &results, 100, &found_command, &score, &is_new_command));
  TF_LITE_MICRO_EXPECT_EQ(0, score);
  TF_LITE_MICRO_EXPECT_EQ(false, is_new_command);
}

TF_LITE_MICRO_TESTS_END
