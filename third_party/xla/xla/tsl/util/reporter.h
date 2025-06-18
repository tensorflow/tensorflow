/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_UTIL_REPORTER_H_
#define XLA_TSL_UTIL_REPORTER_H_

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_set>

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/protobuf/test_log.pb.h"

namespace tsl {

// The TestReportFile provides a file abstraction for TF tests to use.
class TestReportFile {
 public:
  // Create a TestReportFile with the test name 'test_name'.
  TestReportFile(const string& fname, const string& test_name);

  // Initialize the TestReportFile.  If the reporting env flag is set,
  // try to create the reporting file.  Fails if the file already exists.
  absl::Status Initialize();

  // Append the report file w/ 'content'.
  absl::Status Append(const string& content);

  // Close the report file.
  absl::Status Close();

  bool IsClosed() const { return closed_; }

  ~TestReportFile() { Close().IgnoreError(); }  // Autoclose in destructor.

 private:
  bool closed_;
  string fname_;
  string test_name_;
  std::unique_ptr<WritableFile> log_file_;
  TestReportFile(const TestReportFile&) = delete;
  void operator=(const TestReportFile&) = delete;
};

// The TestReporter writes test / benchmark output to binary Protobuf files when
// the environment variable "TEST_REPORT_FILE_PREFIX" is defined.
//
// If this environment variable is not defined, no logging is performed.
//
// The intended use is via the following lines:
//
//  TestReporter reporter(test_name);
//  TF_CHECK_OK(reporter.Initialize()));
//  TF_CHECK_OK(reporter.Benchmark(iters, cpu_time, wall_time, throughput));
//  TF_CHECK_OK(reporter.SetProperty("some_string_property", "some_value");
//  TF_CHECK_OK(reporter.SetProperty("some_double_property", double_value);
//  TF_CHECK_OK(reporter.Close());
//
// For example, if the environment variable
//   TEST_REPORT_FILE_PREFIX="/tmp/run_"
// is set, and test_name is "BM_Foo/1/2", then a BenchmarkEntries pb
// with a single entry is written to file:
//   /tmp/run_BM_Foo__1__2
//
class TestReporter {
 public:
  static constexpr const char* kTestReporterEnv = "TEST_REPORT_FILE_PREFIX";

  // Create a TestReporter with the test name 'test_name'.
  explicit TestReporter(const string& test_name)
      : TestReporter(GetLogEnv(), test_name) {}

  // Provide a prefix filename, mostly used for testing this class.
  TestReporter(const string& fname, const string& test_name);

  // Initialize the TestReporter.  If the reporting env flag is set,
  // try to create the reporting file.  Fails if the file already exists.
  absl::Status Initialize();

  // Finalize the report.  If the reporting env flag is set,
  // flush the reporting file and close it.
  // Once Close is called, no other methods should be called other
  // than Close and the destructor.
  absl::Status Close();

  // Set the report to be a Benchmark and log the given parameters.
  // Only does something if the reporting env flag is set.
  // Does not guarantee the report is written.  Use Close() to
  // enforce I/O operations.
  absl::Status Benchmark(int64_t iters, double cpu_time, double wall_time,
                         double throughput);

  // Set property on Benchmark to the given value.
  absl::Status SetProperty(const string& name, double value);

  // Set property on Benchmark to the given value.
  absl::Status SetProperty(const string& name, const string& value);

  // Add the given value to the metrics on the Benchmark.
  absl::Status AddMetric(const string& name, double value);

  // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
  ~TestReporter() { Close().IgnoreError(); }  // Autoclose in destructor.

 private:
  static string GetLogEnv() {
    const char* fname_ptr = getenv(kTestReporterEnv);
    return (fname_ptr != nullptr) ? fname_ptr : "";
  }
  TestReportFile report_file_;
  tensorflow::BenchmarkEntry benchmark_entry_;
  TestReporter(const TestReporter&) = delete;
  void operator=(const TestReporter&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_UTIL_REPORTER_H_
