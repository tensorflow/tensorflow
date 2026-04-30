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

#include "xla/backends/profiler/subprocess/subprocess_registry.h"

#include <cstdint>
#include <ctime>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpc/support/time.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

constexpr absl::Duration kConnectTimeout = absl::Seconds(30);

// Registry of subprocesses.
struct Registry {
  absl::Mutex mu;
  absl::flat_hash_map<int32_t, SubprocessInfo> subprocesses ABSL_GUARDED_BY(mu);
};

// Global registry of subprocesses.
Registry& registry() {
  static absl::NoDestructor<Registry> registry;
  return *registry;
}

absl::StatusOr<SubprocessCleanup> RegisterSubprocess(
    SubprocessInfo&& subprocess_info) {
  // This check ensures that there can't be a loop in the subprocess profiler
  // graph.
  if (subprocess_info.pid == tsl::Env::Default()->GetProcessId()) {
    return absl::InvalidArgumentError(
        "Cannot register subprocess with the same pid as current process.");
  }
  ::grpc::ChannelArguments channel_args;
  channel_args.SetMaxReceiveMessageSize(std::numeric_limits<int32_t>::max());
  auto channel = ::grpc::CreateCustomChannel(
      subprocess_info.address, ::grpc::InsecureChannelCredentials(),
      channel_args);
  if (!channel) {
    return absl::InternalError(
        absl::StrCat("Unable to create channel to ", subprocess_info.address));
  }
  // TODO(b/507516897): Remove manual conversion once grpc releases the
  // absl::Time support.
  gpr_timespec connect_timeout_spec;
  timespec absl_timespec = absl::ToTimespec(absl::Now() + kConnectTimeout);
  connect_timeout_spec.tv_sec = absl_timespec.tv_sec;
  connect_timeout_spec.tv_nsec = absl_timespec.tv_nsec;
  connect_timeout_spec.clock_type = GPR_CLOCK_REALTIME;
  if (!channel->WaitForConnected(connect_timeout_spec)) {
    return absl::DeadlineExceededError(
        absl::StrCat("Timeout while connecting to ", subprocess_info.address));
  }
  subprocess_info.profiler_stub =
      tensorflow::grpc::ProfilerService::NewStub(channel);
  int32_t pid_to_unregister = subprocess_info.pid;
  {
    absl::MutexLock l(registry().mu);
    if (!registry()
             .subprocesses
             .try_emplace(subprocess_info.pid, std::move(subprocess_info))
             .second) {
      return absl::AlreadyExistsError(
          absl::StrCat(subprocess_info.DebugString(), " already registered"));
    }
  }
  return SubprocessCleanup([pid_to_unregister]() {
    absl::MutexLock l(registry().mu);
    if (registry().subprocesses.erase(pid_to_unregister) == 0) {
      LOG_IF(WARNING, registry().subprocesses.find(pid_to_unregister) !=
                          registry().subprocesses.end())
          << "Failed to unregister "
          << registry().subprocesses.at(pid_to_unregister).DebugString();
    }
  });
}

}  // namespace

std::string SubprocessInfo::DebugString() const {
  return absl::StrCat("SubprocessInfo(pid: ", pid, ", address: ", address, ")");
}

absl::StatusOr<SubprocessCleanup> RegisterSubprocess(
    int32_t pid, std::optional<int> port,
    std::optional<absl::string_view> unix_domain_socket) {
  if (!port.has_value() && !unix_domain_socket.has_value()) {
    return absl::InvalidArgumentError(
        "Either port or unix_domain_socket must be set");
  }
  const std::string address = unix_domain_socket.has_value()
                                  ? absl::StrCat("unix:", *unix_domain_socket)
                                  : absl::StrCat("localhost:", *port);
  return RegisterSubprocess({pid, std::move(address)});
}

std::vector<SubprocessInfo> GetRegisteredSubprocesses() {
  absl::MutexLock l(registry().mu);
  std::vector<SubprocessInfo> subprocesses;
  subprocesses.reserve(registry().subprocesses.size());
  for (const auto& [_, subprocess_info] : registry().subprocesses) {
    subprocesses.push_back(subprocess_info);
  }
  return subprocesses;
}

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
