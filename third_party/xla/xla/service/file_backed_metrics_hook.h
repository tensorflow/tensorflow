/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FILE_BACKED_METRICS_HOOK_H_
#define XLA_SERVICE_FILE_BACKED_METRICS_HOOK_H_
#include <string>

#include "google/protobuf/duration.pb.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/service/metrics.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/proto/proto_utils.h"

namespace xla {

class FileBackedMetricsHook {
 public:
  explicit FileBackedMetricsHook(absl::string_view filepath)
      : filepath_(filepath) {}

  void RecordCompilationMetrics(CompilationLogEntry::CompilationStage stage,
                                absl::Duration latency,
                                absl::string_view hlo_module_name) {
    CompilationLogEntry entry;
    entry.set_stage(stage);
    *entry.mutable_duration() = tsl::proto_utils::ToDurationProto(latency);
    entry.set_hlo_module_name(hlo_module_name);
    absl::MutexLock lock(mu_);
    *logs_.add_entries() = entry;
  }

  // Dumps the collected metrics to the file path, overwriting the file if it
  // exists.
  absl::Status DumpMetrics() {
    absl::MutexLock lock(mu_);
    return tsl::WriteTextProto(tsl::Env::Default(), filepath_, logs_);
  }

 private:
  absl::Mutex mu_;
  CompilationLogs logs_ ABSL_GUARDED_BY(mu_);
  std::string filepath_;
};

}  // namespace xla

#endif  // XLA_SERVICE_FILE_BACKED_METRICS_HOOK_H_
