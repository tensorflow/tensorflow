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

#ifndef XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_PROFILING_SESSION_H_
#define XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_PROFILING_SESSION_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/async_unary_call.h"
#include "grpcpp/support/status.h"
#include "xla/backends/profiler/subprocess/subprocess_registry.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/profiler_service.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace subprocess {

class SubprocessProfilingSession : public tsl::profiler::ProfilerInterface {
 public:
  // Creates a profiler for the specified subprocess. The subprocess must have a
  // ProfilerService gRPC server listening on the address specified in
  // `subprocess_info`.
  static absl::StatusOr<std::unique_ptr<SubprocessProfilingSession>> Create(
      const SubprocessInfo& subprocess_info,
      const tensorflow::ProfileOptions& options);
  // Not copyable or movable
  SubprocessProfilingSession(const SubprocessProfilingSession&) = delete;
  SubprocessProfilingSession& operator=(const SubprocessProfilingSession&) =
      delete;

  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  SubprocessProfilingSession(const SubprocessInfo& subprocess_info,
                             const tensorflow::ProfileRequest& request);

  SubprocessInfo subprocess_info_;
  tensorflow::ProfileRequest request_;
  tensorflow::ProfileResponse response_;
  grpc::ClientContext context_;
  grpc::CompletionQueue completion_queue_;
  grpc::Status grpc_status_;
  std::unique_ptr<grpc::ClientAsyncResponseReader<tensorflow::ProfileResponse>>
      rpc_;
};

}  // namespace subprocess
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_SUBPROCESS_SUBPROCESS_PROFILING_SESSION_H_
