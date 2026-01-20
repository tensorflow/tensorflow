/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_RUNNER_PJRT_H_
#define XLA_SERVICE_HLO_RUNNER_PJRT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_layout.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A base class for running an HloModule using the PjRt API. This class
// abstracts execution for a given HloModule using PjRt interfaces.
// HloModule can be explicitly built, or loaded from a serialization file (e.g.,
// hlo proto file), or parsed from a hlo textual IR string.
class HloRunnerPjRt : public HloRunnerInterface {
 public:
  explicit HloRunnerPjRt(std::unique_ptr<PjRtClient> pjrt_client);

  // Transfers data between the host and device, using the given parameter
  // layouts.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  TransferLiteralsToDevice(absl::Span<const ShapeLayout> layouts,
                           absl::Span<const Literal* const> literals);
  // Transfers data between the host and device, using the layout of each
  // literal itself.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  TransferLiteralsToDevice(absl::Span<const Literal* const> literals);
  absl::StatusOr<Literal> TransferLiteralsFromDevice(
      absl::Span<const std::unique_ptr<PjRtBuffer>> output_buffers,
      bool untuple_result);

  // Executes the given module with given literals as input and returns the
  // result as a Literal.
  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal* const> arguments,
                                  bool run_hlo_passes) override;

  // Like Execute(), but accepts and returns pjrt buffers instead of literals.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  ExecuteWithDeviceBuffers(
      OpaqueExecutable* executable,
      const std::vector<std::unique_ptr<PjRtBuffer>>& arguments,
      const ExecuteOptions* execute_options = nullptr);

  // Creates an executable object given an HLO module. If run_hlo_passes is
  // true, the HLO passes will be run as part of compilation.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  // Creates a runner-internal executable object given a runner and
  // platform-specific serialized executable representation. The serialized
  // representation must have been produced by a compiler of the same platform
  // and version as this one.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> DeserializeExecutable(
      absl::string_view serialized) const override;

  using HloRunnerInterface::ExecuteWithExecutable;
  absl::StatusOr<std::vector<absl::StatusOr<Literal>>> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
      int64_t num_repeats) override;

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) override;

  // Same as above, but with specified device assignment.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithExecutable(
      OpaqueExecutable* absl_nonnull executable,
      const ReplicatedExecuteOptions& options) override;

  // Same as above, but with specified device assignment.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithExecutable(
      OpaqueExecutable* absl_nonnull executable,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
      absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
      absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  absl::string_view Name() const override;

  int device_count() const override { return pjrt_client_->device_count(); }

  bool HasProperty(HloRunnerPropertyTag::Type tag) const override;

  absl::StatusOr<const HloModule* absl_nonnull> HloModuleFromWrapped(
      const OpaqueExecutable* wrapped) const override;

  // Returns true if the two given OpaqueExecutables originate from the same
  // runner and are equivalent according to some notion specific to that runner.
  // Executables that were created by different runners can never be equivalent.
  bool ExecutablesAreEquivalent(
      const OpaqueExecutable* absl_nonnull lhs,
      const OpaqueExecutable* absl_nonnull rhs) const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

 private:
  absl::StatusOr<CompileOptions> GenerateDefaultCompileOptions(
      HloModule* module, bool run_hlo_passes);

  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedImpl(
      absl::AnyInvocable<
          absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(
              absl::Span<const std::vector<PjRtBuffer*>>,
              absl::AnyInvocable<OpaqueExecutable*(int64_t)>,
              absl::Span<PjRtDevice* const>)>
          execution_helper,
      absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
      absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
      absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> TransferLiteralToDevice(
      const Literal& literal, PjRtMemorySpace* absl_nonnull memory_space,
      const Layout& on_device_layout);
  absl::StatusOr<Literal> TransferLiteralFromDevice(PjRtBuffer& buffer);

  std::unique_ptr<PjRtClient> pjrt_client_;
};

// This class works just like a HloRunnerPjRt, but it only runs compilation
// (persisting the executable to disk) and does not run the executable.
class CompilePhaseHloRunnerPjRt : public HloRunnerPjRt {
 public:
  CompilePhaseHloRunnerPjRt(std::unique_ptr<PjRtClient> pjrt_client,
                            absl::string_view artifact_dir)
      : HloRunnerPjRt(std::move(pjrt_client)), artifact_dir_(artifact_dir) {}

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  absl::StatusOr<std::vector<absl::StatusOr<Literal>>> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
      int64_t num_repeats) override {
    return absl::UnimplementedError(
        "CompilePhaseHloRunnerPjRt does not support execution. This is "
        "expected.");
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
      absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
      absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override {
    return absl::UnimplementedError(
        "CompilePhaseHloRunnerPjRt does not support execution. This is "
        "expected.");
  }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

 private:
  std::string artifact_dir_;
};

// This class works just like a HloRunnerPjRt, but it only runs execution
// (reading the executable from disk) and does not compile the executable.
//
// If `compile_if_not_found` is true, this class will attempt to compile the
// executable if the serialized version from the compile phase could not be
// found. This effectively makes this class equivalent to HloRunnerPjRt.
//
// If `fail_duplicate_loads` is true, calls to CreateExecutable will fail if the
// executable was previously loaded using the same runner. Most tests do not
// need to load an executable more than once and setting this can help catch
// instances where e.g. fingerprints are colliding.
class ExecutePhaseHloRunnerPjRt : public HloRunnerPjRt {
 public:
  ExecutePhaseHloRunnerPjRt(std::unique_ptr<PjRtClient> pjrt_client,
                            absl::string_view artifact_dir,
                            bool compile_if_not_found = true,
                            bool fail_duplicate_loads = true)
      : HloRunnerPjRt(std::move(pjrt_client)),
        artifact_dir_(artifact_dir),
        compile_if_not_found_(compile_if_not_found),
        fail_duplicate_loads_(fail_duplicate_loads) {}

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

 private:
  std::string artifact_dir_;
  bool compile_if_not_found_;
  bool fail_duplicate_loads_;

  absl::flat_hash_set<std::string> loaded_executable_paths_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_RUNNER_PJRT_H_
