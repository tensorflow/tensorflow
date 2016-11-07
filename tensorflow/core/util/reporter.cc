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

#include "tensorflow/core/util/reporter.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

TestReporter::TestReporter(const string& fname, const string& test_name)
    : closed_(true), fname_(fname), test_name_(test_name) {}

Status TestReporter::Close() {
  if (closed_) return Status::OK();

  BenchmarkEntries entries;
  *entries.add_entry() = benchmark_entry_;
  TF_RETURN_IF_ERROR(log_file_->Append(entries.SerializeAsString()));

  benchmark_entry_.Clear();
  closed_ = true;

  return log_file_->Close();
}

Status TestReporter::Benchmark(int64 iters, double cpu_time, double wall_time,
                               double throughput) {
  if (closed_) return Status::OK();
  benchmark_entry_.set_iters(iters);
  benchmark_entry_.set_cpu_time(cpu_time);
  benchmark_entry_.set_wall_time(wall_time);
  benchmark_entry_.set_throughput(throughput);
  return Status::OK();
}

Status TestReporter::Initialize() {
  if (fname_.empty()) {
    return Status::OK();
  }
  string mangled_fname = strings::StrCat(
      fname_, str_util::Join(str_util::Split(test_name_, '/'), "__"));
  Env* env = Env::Default();
  if (env->FileExists(mangled_fname).ok()) {
    return errors::InvalidArgument("Cannot create TestReporter, file exists: ",
                                   mangled_fname);
  }
  TF_RETURN_IF_ERROR(env->NewWritableFile(mangled_fname, &log_file_));
  TF_RETURN_IF_ERROR(log_file_->Flush());

  benchmark_entry_.set_name(test_name_);
  closed_ = false;
  return Status::OK();
}

}  // namespace tensorflow
