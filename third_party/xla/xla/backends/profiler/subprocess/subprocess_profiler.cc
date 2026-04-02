/* Copyright 2026 The OpenXLA Authors.

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
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/backends/profiler/subprocess/subprocess_profiling_session.h"
#include "xla/backends/profiler/subprocess/subprocess_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/profiler/utils/profiler_options_util.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

class SubprocessProfiler : public ::tsl::profiler::ProfilerInterface {
 public:
  explicit SubprocessProfiler(std::vector<SubprocessInfo> subprocesses,
                              const tensorflow::ProfileOptions& options)
      : subprocesses_(std::move(subprocesses)), options_(options) {
    sessions_.reserve(subprocesses_.size());
    for (const auto& subprocess : subprocesses_) {
      auto session = SubprocessProfilingSession::Create(subprocess, options);
      if (!session.ok()) {
        LOG(ERROR) << "Failed to create subprocess profiling session: "
                   << session.status();
        continue;
      }
      sessions_.push_back(std::move(session).value());
    }
  }

  absl::Status Start() override {
    if (sessions_.empty()) {
      return absl::OkStatus();
    }
    std::vector<absl::Status> statuses(sessions_.size());
    {
      tsl::thread::ThreadPool thread_pool(
          tsl::Env::Default(), "subprocess_profiler_start", sessions_.size());
      for (int i = 0; i < sessions_.size(); ++i) {
        thread_pool.Schedule(
            [this, i, &statuses] { statuses[i] = sessions_[i]->Start(); });
      }
      // ThreadPool is not joined until destruction.
    }
    UpdateStatus(statuses);
    return status_;
  }

  absl::Status Stop() override {
    if (sessions_.empty()) {
      return absl::OkStatus();
    }
    std::vector<absl::Status> statuses(sessions_.size());
    {
      tsl::thread::ThreadPool thread_pool(
          tsl::Env::Default(), "subprocess_profiler_stop", sessions_.size());
      for (int i = 0; i < sessions_.size(); ++i) {
        absl::Status& session_status = statuses[i];
        thread_pool.Schedule([this, i, &session_status] {
          session_status = sessions_[i]->Stop();
        });
      }
      // ThreadPool is not joined until destruction.
    }
    UpdateStatus(statuses);
    return status_;
  }

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override {
    if (!status_.ok()) {
      return status_;
    }
    for (auto& session : sessions_) {
      // CollectData is synchronous and thread-safe aggregation might be tricky
      // if parallelized without locking the XSpace. For now, we do it
      // synchronously as requested.
      status_.Update(session->CollectData(space));
    }
    return status_;
  }

 private:
  void UpdateStatus(absl::Span<const absl::Status> statuses) {
    for (const auto& status : statuses) {
      status_.Update(status);
    }
  }
  const std::vector<SubprocessInfo> subprocesses_;
  const tensorflow::ProfileOptions options_;
  std::vector<std::unique_ptr<SubprocessProfilingSession>> sessions_;
  absl::Status status_;
};

std::unique_ptr<::tsl::profiler::ProfilerInterface> CreateSubprocessProfiler(
    const tensorflow::ProfileOptions& options) {
  auto profile_subprocesses =
      tsl::profiler::GetConfigValue(options, "profile_subprocesses");
  if (profile_subprocesses.has_value() &&
      std::get<bool>(*profile_subprocesses)) {
    return std::make_unique<SubprocessProfiler>(GetRegisteredSubprocesses(),
                                                options);
  }
  return nullptr;
}

auto register_subprocess_profiler_factory = [] {
  tsl::profiler::RegisterProfilerFactory(&CreateSubprocessProfiler);
  return 0;
}();

}  // namespace
}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
