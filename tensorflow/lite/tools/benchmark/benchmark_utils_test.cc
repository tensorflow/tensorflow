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
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace benchmark {
namespace {

TEST(BenchmarkHelpersTest, SleepForNegativeSeconds) {
  const auto start_ts = tflite::profiling::time::NowMicros();
  // The following should return immediately.
  util::SleepForSeconds(-5.0);
  const auto end_ts = tflite::profiling::time::NowMicros();

  // As we don't have a mocked clock, we simply expect <1 sec has elapsed, which
  // is admittedly not quite accurate.
  EXPECT_LT(end_ts - start_ts, 1000000);
}

TEST(BenchmarkHelpersTest, SleepForSomeSeconds) {
  const auto start_ts = tflite::profiling::time::NowMicros();
  // The following should return after 2.0 secs
  util::SleepForSeconds(2.0);
  const auto end_ts = tflite::profiling::time::NowMicros();

  // As we don't have a mocked clock, we simply expect >1.9 sec has elapsed.
  EXPECT_GT(end_ts - start_ts, 1900000);
}

TEST(BenchmarkHelpersTest, SplitAndParseFailed) {
  std::vector<int> results;
  const bool splitted = util::SplitAndParse("hello;world", ';', &results);

  EXPECT_FALSE(splitted);
}

TEST(BenchmarkHelpersTest, SplitAndParseString) {
  std::vector<std::string> results;
  const bool splitted = util::SplitAndParse("hello,world", ',', &results);

  EXPECT_TRUE(splitted);
  EXPECT_EQ(2, results.size());

  EXPECT_EQ("hello", results[0]);
  EXPECT_EQ("world", results[1]);
}

TEST(BenchmarkHelpersTest, SplitAndParseInts) {
  std::vector<int> results;
  const bool splitted = util::SplitAndParse("1,2", ',', &results);

  EXPECT_TRUE(splitted);
  EXPECT_EQ(2, results.size());

  EXPECT_EQ(1, results[0]);
  EXPECT_EQ(2, results[1]);
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite
