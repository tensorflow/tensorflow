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

#include "xla/tsl/util/reporter.h"

#include "xla/tsl/platform/errors.h"
#include "tsl/platform/str_util.h"

namespace tsl {

TestReportFile::TestReportFile(const string& fname, const string& test_name)
    : closed_(true), fname_(fname), test_name_(test_name) {}

absl::Status TestReportFile::Append(const string& content) {
  if (closed_) return absl::OkStatus();
  return log_file_->Append(content);
}

absl::Status TestReportFile::Close() {
  if (closed_) return absl::OkStatus();
  closed_ = true;
  return log_file_->Close();
}

absl::Status TestReportFile::Initialize() {
  if (fname_.empty()) {
    return absl::OkStatus();
  }
  string mangled_fname = strings::StrCat(
      fname_, absl::StrJoin(str_util::Split(test_name_, '/'), "__"));
  Env* env = Env::Default();
  if (env->FileExists(mangled_fname).ok()) {
    return errors::InvalidArgument(
        "Cannot create TestReportFile, file exists: ", mangled_fname);
  }
  TF_RETURN_IF_ERROR(env->NewWritableFile(mangled_fname, &log_file_));
  TF_RETURN_IF_ERROR(log_file_->Flush());

  closed_ = false;
  return absl::OkStatus();
}

TestReporter::TestReporter(const string& fname, const string& test_name)
    : report_file_(fname, test_name) {
  benchmark_entry_.set_name(test_name);
}

absl::Status TestReporter::Close() {
  if (report_file_.IsClosed()) return absl::OkStatus();

  tensorflow::BenchmarkEntries entries;
  *entries.add_entry() = benchmark_entry_;
  TF_RETURN_IF_ERROR(report_file_.Append(entries.SerializeAsString()));
  benchmark_entry_.Clear();

  return report_file_.Close();
}

absl::Status TestReporter::Benchmark(int64_t iters, double cpu_time,
                                     double wall_time, double throughput) {
  if (report_file_.IsClosed()) return absl::OkStatus();
  benchmark_entry_.set_iters(iters);
  benchmark_entry_.set_cpu_time(cpu_time / iters);
  benchmark_entry_.set_wall_time(wall_time / iters);
  benchmark_entry_.set_throughput(throughput);
  return absl::OkStatus();
}

absl::Status TestReporter::SetProperty(const string& name,
                                       const string& value) {
  if (report_file_.IsClosed()) return absl::OkStatus();
  (*benchmark_entry_.mutable_extras())[name].set_string_value(value);
  return absl::OkStatus();
}

absl::Status TestReporter::SetProperty(const string& name, double value) {
  if (report_file_.IsClosed()) return absl::OkStatus();
  (*benchmark_entry_.mutable_extras())[name].set_double_value(value);
  return absl::OkStatus();
}

absl::Status TestReporter::AddMetric(const string& name, double value) {
  if (report_file_.IsClosed()) return absl::OkStatus();
  auto* metric = benchmark_entry_.add_metrics();
  metric->set_name(name);
  metric->set_value(value);
  return absl::OkStatus();
}

absl::Status TestReporter::Initialize() { return report_file_.Initialize(); }

}  // namespace tsl
