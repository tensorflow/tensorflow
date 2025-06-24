/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_
#define XLA_TSL_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/rpc/client/profiler_client.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
namespace profiler {

using AddressResolver = std::function<std::string(absl::string_view)>;

// Manages one or more remote profiling sessions.
class RemoteProfilerSessionManager {
 public:
  struct Response {
    std::string service_address;
    std::unique_ptr<tensorflow::ProfileResponse> profile_response;
    absl::Status status;
  };
  // Instantiates a collection of RemoteProfilerSessions starts profiling on
  // each of them immediately. Assumes that options have already been validated.
  static std::unique_ptr<RemoteProfilerSessionManager> Create(
      const tensorflow::RemoteProfilerSessionManagerOptions& options,
      const tensorflow::ProfileRequest& request, absl::Status& out_status,
      AddressResolver resolver = nullptr);

  // Awaits for responses from remote profiler sessions and returns them as a
  // list. Subsequent calls beyond the first will yield a list of errors.
  std::vector<Response> WaitForCompletion();

  // Not copyable or movable.
  RemoteProfilerSessionManager(const RemoteProfilerSessionManager&) = delete;
  RemoteProfilerSessionManager& operator=(const RemoteProfilerSessionManager&) =
      delete;

  ~RemoteProfilerSessionManager();

 private:
  explicit RemoteProfilerSessionManager(
      tensorflow::RemoteProfilerSessionManagerOptions options,
      tensorflow::ProfileRequest request, AddressResolver resolver);

  // Initialization of all client contexts.
  absl::Status Init();

  absl::Mutex mutex_;
  // Remote profiler session options.
  tensorflow::RemoteProfilerSessionManagerOptions options_
      TF_GUARDED_BY(mutex_);
  tensorflow::ProfileRequest request_ TF_GUARDED_BY(mutex_);
  // List of clients, each connects to a profiling service.
  std::vector<std::unique_ptr<RemoteProfilerSession>> clients_
      TF_GUARDED_BY(mutex_);
  // Resolves an address into a format that gRPC understands.
  AddressResolver resolver_ TF_GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_
