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

#ifndef TENSORFLOW_CORE_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_
#define TENSORFLOW_CORE_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/rpc/client/profiler_client.h"

namespace tensorflow {
namespace profiler {

using AddressResolver = std::function<std::string(absl::string_view)>;

// Manages one or more remote profiling sessions.
class RemoteProfilerSessionManager {
 public:
  struct Response {
    std::string service_address;
    std::unique_ptr<ProfileResponse> profile_response;
    Status status;
  };
  // Instantiates a collection of RemoteProfilerSessions starts profiling on
  // each of them immediately. Assumes that options have already been validated.
  static std::unique_ptr<RemoteProfilerSessionManager> Create(
      const RemoteProfilerSessionManagerOptions& options,
      const ProfileRequest& request, tensorflow::Status& out_status,
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
      RemoteProfilerSessionManagerOptions options, ProfileRequest request,
      AddressResolver resolver);

  // Initialization of all client contexts.
  Status Init();

  mutex mutex_;
  // Remote profiler session options.
  RemoteProfilerSessionManagerOptions options_ TF_GUARDED_BY(mutex_);
  ProfileRequest request_ TF_GUARDED_BY(mutex_);
  // List of clients, each connects to a profiling service.
  std::vector<std::unique_ptr<RemoteProfilerSession>> clients_
      TF_GUARDED_BY(mutex_);
  // Resolves an address into a format that gRPC understands.
  AddressResolver resolver_ TF_GUARDED_BY(mutex_);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_RPC_CLIENT_REMOTE_PROFILER_SESSION_MANAGER_H_
