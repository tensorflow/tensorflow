/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define _XOPEN_SOURCE  // for setenv, unsetenv
#include <cstdlib>

#include "tensorflow/core/util/reporter.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Tests of all the error paths in log_reader.cc follow:
static void ExpectHasSubstr(absl::string_view s, absl::string_view expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << s << " does not contain " << expected;
}

TEST(TestReporter, NoLogging) {
  TestReporter test_reporter("b1");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.Close());
}

TEST(TestReporter, UsesEnv) {
  const char* old_env = std::getenv(TestReporter::kTestReporterEnv);

  // Set a file we can't possibly create, check for failure
  setenv(TestReporter::kTestReporterEnv, "/cant/find/me:!", 1);
  CHECK_EQ(string(std::getenv(TestReporter::kTestReporterEnv)),
           string("/cant/find/me:!"));
  TestReporter test_reporter("b1");
  absl::Status s = test_reporter.Initialize();
  ExpectHasSubstr(s.ToString(), "/cant/find/me");

  // Remove the env variable, no logging is performed
  unsetenv(TestReporter::kTestReporterEnv);
  CHECK_EQ(std::getenv(TestReporter::kTestReporterEnv), nullptr);
  TestReporter test_reporter_empty("b1");
  s = test_reporter_empty.Initialize();
  TF_EXPECT_OK(s);
  s = test_reporter_empty.Close();
  TF_EXPECT_OK(s);

  if (old_env == nullptr) {
    unsetenv(TestReporter::kTestReporterEnv);
  } else {
    setenv(TestReporter::kTestReporterEnv, old_env, 1);
  }
}

TEST(TestReporter, CreateTwiceFails) {
  {
    TestReporter test_reporter(
        strings::StrCat(testing::TmpDir(), "/test_reporter_dupe"), "t1");
    TF_EXPECT_OK(test_reporter.Initialize());
  }
  {
    TestReporter test_reporter(
        strings::StrCat(testing::TmpDir(), "/test_reporter_dupe"), "t1");
    absl::Status s = test_reporter.Initialize();
    ExpectHasSubstr(s.ToString(), "file exists:");
  }
}

TEST(TestReporter, CreateCloseCreateAgainSkipsSecond) {
  TestReporter test_reporter(
      strings::StrCat(testing::TmpDir(), "/test_reporter_create_close"), "t1");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.Close());
  TF_EXPECT_OK(test_reporter.Benchmark(1, 1.0, 2.0, 3.0));  // No-op, closed
  TF_EXPECT_OK(test_reporter.Close());                      // No-op, closed
  absl::Status s = test_reporter.Initialize();  // Try to reinitialize
  ExpectHasSubstr(s.ToString(), "file exists:");
}

TEST(TestReporter, Benchmark) {
  string fname =
      strings::StrCat(testing::TmpDir(), "/test_reporter_benchmarks_");
  TestReporter test_reporter(fname, "b1/2/3");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.Benchmark(1, 1.0, 2.0, 3.0));
  TF_EXPECT_OK(test_reporter.Close());

  string expected_fname = strings::StrCat(fname, "b1__2__3");
  string read;
  TF_EXPECT_OK(ReadFileToString(Env::Default(), expected_fname, &read));

  BenchmarkEntries benchmark_entries;
  ASSERT_TRUE(benchmark_entries.ParseFromString(read));
  ASSERT_EQ(1, benchmark_entries.entry_size());
  const BenchmarkEntry& benchmark_entry = benchmark_entries.entry(0);

  EXPECT_EQ(benchmark_entry.name(), "b1/2/3");
  EXPECT_EQ(benchmark_entry.iters(), 1);
  EXPECT_EQ(benchmark_entry.cpu_time(), 1.0);
  EXPECT_EQ(benchmark_entry.wall_time(), 2.0);
  EXPECT_EQ(benchmark_entry.throughput(), 3.0);
}

TEST(TestReporter, SetProperties) {
  string fname =
      strings::StrCat(testing::TmpDir(), "/test_reporter_benchmarks_");
  TestReporter test_reporter(fname, "b2/3/4");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.SetProperty("string_prop", "abc"));
  TF_EXPECT_OK(test_reporter.SetProperty("double_prop", 4.0));

  TF_EXPECT_OK(test_reporter.Close());
  string expected_fname = strings::StrCat(fname, "b2__3__4");
  string read;
  TF_EXPECT_OK(ReadFileToString(Env::Default(), expected_fname, &read));

  BenchmarkEntries benchmark_entries;
  ASSERT_TRUE(benchmark_entries.ParseFromString(read));
  ASSERT_EQ(1, benchmark_entries.entry_size());
  const BenchmarkEntry& benchmark_entry = benchmark_entries.entry(0);
  const auto& extras = benchmark_entry.extras();
  ASSERT_EQ(2, extras.size());
  EXPECT_EQ("abc", extras.at("string_prop").string_value());
  EXPECT_EQ(4.0, extras.at("double_prop").double_value());
}

TEST(TestReporter, AddMetrics) {
  string fname =
      strings::StrCat(testing::TmpDir(), "/test_reporter_benchmarks_");
  TestReporter test_reporter(fname, "b3/4/5");
  TF_EXPECT_OK(test_reporter.Initialize());
  TF_EXPECT_OK(test_reporter.AddMetric("metric1", 2.0));
  TF_EXPECT_OK(test_reporter.AddMetric("metric2", 3.0));

  TF_EXPECT_OK(test_reporter.Close());
  string expected_fname = strings::StrCat(fname, "b3__4__5");
  string read;
  TF_EXPECT_OK(ReadFileToString(Env::Default(), expected_fname, &read));

  BenchmarkEntries benchmark_entries;
  ASSERT_TRUE(benchmark_entries.ParseFromString(read));
  ASSERT_EQ(1, benchmark_entries.entry_size());
  const BenchmarkEntry& benchmark_entry = benchmark_entries.entry(0);
  const auto& metrics = benchmark_entry.metrics();
  ASSERT_EQ(2, metrics.size());
  EXPECT_EQ("metric1", metrics.at(0).name());
  EXPECT_EQ(2.0, metrics.at(0).value());
  EXPECT_EQ("metric2", metrics.at(1).name());
  EXPECT_EQ(3.0, metrics.at(1).value());
}

}  // namespace
}  // namespace tensorflow
