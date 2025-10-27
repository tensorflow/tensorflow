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
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/profiler_service.grpc.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {
namespace {

// Registry of subprocesses.
struct Registry {
  absl::Mutex mu;
  absl::flat_hash_set<SubprocessInfo> subprocesses ABSL_GUARDED_BY(mu);
};

// Global registry of subprocesses.
Registry& registry() {
  static absl::NoDestructor<Registry> registry;
  return *registry;
}

absl::Status RegisterSubprocess(SubprocessInfo&& subprocess_info) {
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
  subprocess_info.profiler_stub =
      tensorflow::grpc::ProfilerService::NewStub(channel);
  {
    absl::MutexLock l(registry().mu);
    if (!registry().subprocesses.insert(subprocess_info).second) {
      return absl::AlreadyExistsError(
          absl::StrCat(subprocess_info.DebugString(), " already registered"));
    }
  }
  return absl::OkStatus();
}

}  // namespace

std::string SubprocessInfo::DebugString() const {
  return absl::StrCat("SubprocessInfo(pid: ", pid, ", address: ", address, ")");
}

absl::Status RegisterSubprocess(uint32_t pid, int port) {
  return RegisterSubprocess({pid, absl::StrCat("localhost:", port)});
}

absl::Status RegisterSubprocess(uint32_t pid,
                                absl::string_view unix_domain_socket) {
  return RegisterSubprocess({pid, absl::StrCat("unix:", unix_domain_socket)});
}

absl::Status UnregisterSubprocess(uint32_t pid) {
  absl::MutexLock l(registry().mu);
  if (registry().subprocesses.erase({pid, ""}) == 0) {
    LOG(WARNING) << "Subprocess " << pid << " not found";
    return absl::NotFoundError(absl::StrCat(pid, " not found"));
  }
  return absl::OkStatus();
}

std::vector<SubprocessInfo> GetRegisteredSubprocesses() {
  absl::MutexLock l(registry().mu);
  std::vector<SubprocessInfo> subprocesses;
  subprocesses.reserve(registry().subprocesses.size());
  for (const SubprocessInfo& subprocess_info : registry().subprocesses) {
    subprocesses.push_back(subprocess_info);
  }
  return subprocesses;
}

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla
