/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_PYTHON_PGLE_SESSION_H_
#define XLA_PYTHON_PGLE_SESSION_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {
// Allows to run profile sessions several times and collect FDO results after.
class PGLESession {
 public:
  std::unique_ptr<tsl::ProfilerSession> Trace();

  void StopTrace(std::unique_ptr<tsl::ProfilerSession> profiler_session);

  absl::StatusOr<std::string> GetFdoProfile(
      xla::PjRtClient* pjrt_client,
      const CompileOptions& compile_options) const;

 private:
  absl::StatusOr<tensorflow::profiler::ProfiledInstructionsProto>
  AggregateFdoProfileCrossHost(
      tensorflow::profiler::ProfiledInstructionsProto fdo_profile,
      xla::PjRtClient* pjrt_client,
      const CompileOptions& compile_options) const;

  std::vector<tensorflow::profiler::ProfiledInstructionsProto> fdo_profiles_;
};

// Allow to run profile collection in scope and collect FDO results after.
// Supports multi-host setup.
class PGLESessionRunner {
 public:
  // PGLE session scoped wrapper.
  class PGLESessionHandler {
   public:
    explicit PGLESessionHandler(PGLESession& pgle_session);
    ~PGLESessionHandler();

   private:
    PGLESession& pgle_session_;
    std::unique_ptr<tsl::ProfilerSession> profiler_session_ = nullptr;
  };

  PGLESessionRunner(
      int64_t pgle_data_collecting_retries, std::string distributed_data_key,
      std::shared_ptr<xla::DistributedRuntimeClient> distributed_client)
      : pgle_data_collecting_retries_(pgle_data_collecting_retries),
        distributed_data_key_(distributed_data_key),
        distributed_client_(distributed_client) {}

  // Runs the profile session in scope if needed.
  std::optional<PGLESessionHandler> Run(int process_index);

  // FDO profile will be returned only in case when profiles were collected
  // specified amount of times. In case of multi-host FDO will be collected at
  // the first host and be shared with other hosts through distributed runtime
  // service.
  // Method result is cached.
  absl::StatusOr<std::optional<std::string>> GetFdoProfile(
      xla::PjRtClient* pjrt_client, const CompileOptions& options);

 private:
  int64_t pgle_data_collecting_retries_;
  std::string distributed_data_key_;
  std::shared_ptr<xla::DistributedRuntimeClient> distributed_client_;
  int64_t call_times_ = 0;
  PGLESession pgle_session_;
  std::optional<std::string> collected_fdo_;
};

// Creates unique session runner for a given module fingerprint and number of
// retries for FDO profile collection.
class PGLESessionRunnerFactory {
 public:
  // Create or get existing session runner across multi-host environment.
  // In case of single host run distributed_client should be nullptr.
  std::shared_ptr<PGLESessionRunner> Create(
      std::optional<int64_t> pgle_data_collecting_retries,
      std::optional<std::string> module_fingerprint,
      std::shared_ptr<xla::DistributedRuntimeClient> distributed_client);

 private:
  absl::flat_hash_map<std::string, std::shared_ptr<PGLESessionRunner>>
      module_to_session_runner_;
  std::shared_ptr<PGLESessionRunner> empty_runner_ =
      std::make_shared<PGLESessionRunner>(0, "", nullptr);
};
}  // namespace xla

#endif  // XLA_PYTHON_PGLE_SESSION_H_
