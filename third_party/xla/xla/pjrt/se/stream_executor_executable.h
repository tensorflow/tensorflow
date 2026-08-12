/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_SE_STREAM_EXECUTOR_EXECUTABLE_H_
#define XLA_PJRT_SE_STREAM_EXECUTOR_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "riegeli/base/any.h"
#include "riegeli/bytes/reader.h"
#include "xla/client/local_client.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiled_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/abi/executable_abi_version.h"

namespace xla {

class StreamExecutorExecutable : public PjRtExecutable {
 public:
  // Deprecated: use single unique_ptr<CompiledModule> overload instead.
  StreamExecutorExecutable(
      PjRtPlatformId platform_id, const CompileOptions& compile_options,
      std::vector<std::unique_ptr<CompiledModule>> executables,
      int num_replicas, int num_partitions, absl::string_view name,
      absl::string_view fingerprint, absl::string_view default_memory_kind);

  StreamExecutorExecutable(PjRtPlatformId platform_id,
                           const CompileOptions& compile_options,
                           std::unique_ptr<CompiledModule> executable,
                           int num_replicas, int num_partitions,
                           absl::string_view name,
                           absl::string_view fingerprint,
                           absl::string_view default_memory_kind);

  StreamExecutorExecutable(
      PjRtPlatformId platform_id, const CompileOptions& compile_options,
      std::optional<HloModuleProto> unoptimized_hlo_module_proto,
      std::shared_ptr<LocalExecutable> local_executable,
      LocalClient* local_client, int num_replicas, int num_partitions,
      absl::string_view name, absl::string_view fingerprint,
      absl::string_view default_memory_kind);

  absl::StatusOr<std::string> SerializeExecutable() const override;

  absl::string_view name() const override { return name_; }
  int num_replicas() const override { return num_replicas_; }
  int num_partitions() const override { return num_partitions_; }
  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    if (hlo_module_ == nullptr) {
      return std::vector<std::shared_ptr<HloModule>>{};
    }
    return std::vector<std::shared_ptr<HloModule>>{hlo_module_};
  }

  const std::shared_ptr<HloModule>& hlo_module() const { return hlo_module_; }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetParameterMemoryKinds() const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  const CompileOptions& compile_options() const { return compile_options_; }

  absl::StatusOr<std::shared_ptr<LocalExecutable>> GetOrLoadExecutable(
      LocalClient* client);

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  }

  const std::optional<HloModuleProto>& unoptimized_hlo_module_proto() const {
    return unoptimized_hlo_module_proto_;
  }

  absl::StatusOr<std::unique_ptr<PjRtExecutableAbiVersion>> GetAbiVersion()
      const override;

  static absl::StatusOr<std::unique_ptr<StreamExecutorExecutable>> Deserialize(
      riegeli::Any<riegeli::Reader*> reader,
      const PjRtTopologyDescription& topology,
      std::optional<CompileOptions> options = std::nullopt);

 private:
  int64_t SizeOfGeneratedCodeInBytesLocked() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  PjRtPlatformId platform_id_;

  CompileOptions compile_options_;
  // The unoptimized HLO module proto is necessary for HLO debug dumping. It is
  // not available for deserialized executables.
  std::optional<HloModuleProto> unoptimized_hlo_module_proto_;
  mutable absl::Mutex mu_;
  std::variant<std::unique_ptr<CompiledModule>,
               std::shared_ptr<LocalExecutable>>
      executables_ ABSL_GUARDED_BY(mu_);
  LocalClient* local_client_ = nullptr;
  std::shared_ptr<HloModule> hlo_module_;
  int num_replicas_;
  int num_partitions_;
  std::string name_;
  std::string fingerprint_;
  absl::string_view default_memory_kind_;

  absl::StatusOr<stream_executor::ExecutableAbiVersion>
  ExtractExecutableAbiVersion() const;
};

// Reads a serialized ExecutableAndOptionsProto from a string. GPU executables
// are serialized as a custom format that supports executables larger than 2GB
// (unlike regular protos).
absl::StatusOr<ExecutableAndOptionsProto> SerializedGpuExecutableFromReader(
    riegeli::Any<riegeli::Reader*> reader);
absl::StatusOr<ExecutableAndOptionsProto> SerializedGpuExecutableFromString(
    absl::string_view serialized);
absl::StatusOr<ExecutableAndOptionsProto> SerializedGpuExecutableFromString(
    const absl::Cord& serialized);

}  // namespace xla

#endif  // XLA_PJRT_SE_STREAM_EXECUTOR_EXECUTABLE_H_
