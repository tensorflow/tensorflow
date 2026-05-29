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

#ifndef XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_REGISTRY_H_
#define XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_REGISTRY_H_

#include <sys/types.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {

// Information about a registered subprocess.
struct SubprocessInfo {
  int32_t pid;
  std::string address;
  std::shared_ptr<tensorflow::grpc::ProfilerService::Stub> profiler_stub;

  template <typename H>
  friend H AbslHashValue(H h, const SubprocessInfo& subprocess) {
    return H::combine(std::move(h), subprocess.pid);
  }

  bool operator==(const SubprocessInfo& other) const {
    return pid == other.pid;
  }

  bool operator!=(const SubprocessInfo& other) const {
    return !(*this == other);
  }

  std::string DebugString() const;
};

// TODO(b/507517171): replace with open-source equivalent once available.
// A simple RAII class that executes a function upon destruction.
class SubprocessCleanup {
 public:
  SubprocessCleanup() = default;
  explicit SubprocessCleanup(std::function<void()> cleanup)
      : cleanup_(std::move(cleanup)) {}
  ~SubprocessCleanup() { Invoke(); }
  SubprocessCleanup(const SubprocessCleanup&) = delete;
  SubprocessCleanup& operator=(const SubprocessCleanup&) = delete;
  SubprocessCleanup(SubprocessCleanup&& other) noexcept
      : cleanup_(std::exchange(other.cleanup_, nullptr)) {}

  SubprocessCleanup& operator=(SubprocessCleanup&& other) noexcept {
    SubprocessCleanup(std::move(other)).swap(*this);
    return *this;
  }
  // Cancel the execution of the underlying callable.
  void Cancel() { cleanup_ = {}; }

  void Invoke() {
    if (!empty()) {
      std::exchange(cleanup_, nullptr)();
    }
  }

  bool empty() const { return cleanup_ == nullptr; }

 private:
  void swap(SubprocessCleanup& other) noexcept {
    cleanup_.swap(other.cleanup_);
  }
  absl::AnyInvocable<void() &&> cleanup_;
};

// Registers a subprocess that has a running HTTP server listening on the given
// port or Unix domain socket, so that it can be profiled using the
// subprocess profiler.
// It is expected to call `RegisterSubprocess` after starting the subprocess.
// This method will create a stub to the ProfilerService in the subprocess, and
// may block for a while until the stub is ready or connection times out.
// RETURNS: an error if the subprocess is already registered or if the
// subprocess stub cannot be created.
absl::StatusOr<SubprocessCleanup> RegisterSubprocess(
    int32_t pid, std::optional<int> port,
    std::optional<absl::string_view> unix_domain_socket);

// Returns all currently registered subprocesses.
std::vector<SubprocessInfo> GetRegisteredSubprocesses();

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_REGISTRY_H_
