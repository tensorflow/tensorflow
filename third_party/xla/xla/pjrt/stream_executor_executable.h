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

#ifndef XLA_PJRT_STREAM_EXECUTOR_EXECUTABLE_H_
#define XLA_PJRT_STREAM_EXECUTOR_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/client/local_client.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"

namespace xla {
class StreamExecutorExecutable : public PjRtExecutable {
 public:
  StreamExecutorExecutable(
      const CompileOptions& compile_options,
      std::vector<std::unique_ptr<xla::AotCompilationResult>> executables,
      int num_replicas, int num_partitions, absl::string_view name,
      absl::string_view fingerprint, absl::string_view default_memory_kind)
      : compile_options_(compile_options),
        executables_(std::move(executables)),
        num_replicas_(num_replicas),
        num_partitions_(num_partitions),
        name_(name),
        fingerprint_(fingerprint),
        default_memory_kind_(default_memory_kind) {}

  StreamExecutorExecutable(
      const CompileOptions& compile_options,
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      LocalClient* local_client, int num_replicas, int num_partitions,
      absl::string_view name, absl::string_view fingerprint,
      absl::string_view default_memory_kind)
      : compile_options_(compile_options),
        executables_(std::move(local_executables)),
        local_client_(local_client),
        num_replicas_(num_replicas),
        num_partitions_(num_partitions),
        name_(name),
        fingerprint_(fingerprint),
        default_memory_kind_(default_memory_kind) {
    std::vector<std::shared_ptr<HloModule>> hlo_modules;
    for (const auto& local_executable :
         std::get<std::vector<std::unique_ptr<LocalExecutable>>>(
             executables_)) {
      hlo_modules.push_back(local_executable->executable()->shared_module());
    }
    hlo_modules_ = std::move(hlo_modules);
  }

  absl::StatusOr<std::string> SerializeExecutable() const override;

  absl::string_view name() const override { return name_; }
  int num_replicas() const override { return num_replicas_; }
  int num_partitions() const override { return num_partitions_; }
  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    if (!hlo_modules_.has_value()) {
      return absl::UnimplementedError("GetHloModules is not supported.");
    }
    return *hlo_modules_;
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    if (std::holds_alternative<
            std::vector<std::unique_ptr<xla::AotCompilationResult>>>(
            executables_)) {
      return absl::UnimplementedError(
          "Retrieving CompiledMemoryStats is not supported.");
    }
    const auto& local_executables =
        std::get<std::vector<std::unique_ptr<LocalExecutable>>>(executables_);
    if (local_executables.size() != 1) {
      return absl::UnimplementedError(
          "Retrieving CompiledMemoryStats is not supported for multiple "
          "executables.");
    }
    CompiledMemoryStats memory_stats = CompiledMemoryStats();
    memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
    const HloProto* proto = local_executables[0]->executable()->hlo_proto();
    if (proto != nullptr) {
      memory_stats.serialized_hlo_proto = proto->SerializeAsString();
    }
    memory_stats.PopulateBufferStatsFromAllocations(
        local_executables[0]->executable()->GetAllocations());
    return memory_stats;
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const override {
    return absl::UnimplementedError("GetCostAnalysis is not supported.");
  }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    if (std::holds_alternative<
            std::vector<std::unique_ptr<xla::AotCompilationResult>>>(
            executables_)) {
      return 0;
    }
    int64_t size = 0;
    for (auto& executable :
         std::get<std::vector<std::unique_ptr<LocalExecutable>>>(
             executables_)) {
      size += executable->executable()->SizeOfGeneratedCodeInBytes();
    }
    return size;
  }

  const CompileOptions& compile_options() const { return compile_options_; }

  absl::StatusOr<std::vector<std::unique_ptr<LocalExecutable>>>
  ConsumeExecutable(LocalClient* client, const CompileOptions& compile_options);

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  }

 private:
  CompileOptions compile_options_;
  std::variant<std::vector<std::unique_ptr<xla::AotCompilationResult>>,
               std::vector<std::unique_ptr<LocalExecutable>>>
      executables_;
  LocalClient* local_client_ = nullptr;
  std::optional<std::vector<std::shared_ptr<HloModule>>> hlo_modules_;
  int num_replicas_;
  int num_partitions_;
  std::string name_;
  std::string fingerprint_;
  absl::string_view default_memory_kind_;
};
}  // namespace xla

#endif  // XLA_PJRT_STREAM_EXECUTOR_EXECUTABLE_H_
