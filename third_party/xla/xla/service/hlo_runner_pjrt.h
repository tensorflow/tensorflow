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
#include <functional>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_layout.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

// A base class for running an HloModule using the PjRt API. This class
// abstracts execution for a given HloModule using PjRt interfaces.
// HloModule can be explicitly built, or loaded from a serialization file (e.g.,
// hlo proto file), or parsed from a hlo textual IR string.
class HloRunnerPjRt : public HloRunnerInterface {
 public:
  explicit HloRunnerPjRt(
      std::unique_ptr<PjRtClient> pjrt_client,
      DeviceShapeRepresentationFn device_shape_representation_fn,
      DeviceShapeSizeFn device_shape_size_fn);

  ~HloRunnerPjRt() override;

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
                                  bool run_hlo_passes,
                                  ExecutionProfile* profile) override;

  // As Execute(), but accepts and returns device buffers instead of host
  // buffers.
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  ExecuteWithDeviceBuffers(
      PjRtLoadedExecutable* executable, const ExecuteOptions& execute_options,
      const std::vector<std::unique_ptr<PjRtBuffer>>& arguments);

  struct ExecuteWithDeviceBuffersResult {
    std::vector<std::unique_ptr<PjRtBuffer>> buffers;
    bool untuple_result = false;
  };
  absl::StatusOr<ExecuteWithDeviceBuffersResult> ExecuteWithDeviceBuffers(
      OpaqueExecutable* executable,
      const std::vector<std::unique_ptr<PjRtBuffer>>& arguments,
      const ExecuteOptions* execute_options = nullptr);

  // Creates an executable object for an HloModule.
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CreateExecutable(
      HloModule* module, CompileOptions compile_options);

  // Creates an executable object given an HLO module. If run_hlo_passes is
  // true, the HLO passes will be run as part of compilation.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override;

  // Creates a runner-internal executable object given a runner and
  // platform-specific serialized executable representation. The serialized
  // representation must have been produced by a compiler of the same platform
  // and version as this one.
  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> DeserializeExecutable(
      absl::Nonnull<const tsl::protobuf::Message*> serialized) const override;

  absl::StatusOr<Literal> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
      ExecutionProfile* profile) override;

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) override;

  // Same as above, but with specified device assignment.
  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::function<OpaqueExecutable*(int64_t)> executable_provider,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override;

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      OpaqueExecutable* executable,
      const HloRunnerInterface::ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment, ExecutionProfile* profile = nullptr);

  absl::string_view Name() const override;

  void UpdateEntryComputationLayout(HloModule* module) {
    // TODO - b/391868033: Remove UpdateEntryComputationLayout from this class.
    xla::UpdateEntryComputationLayout(module, device_shape_representation_fn_);
  }

  DeviceShapeRepresentationFn device_shape_representation_fn() const override {
    return device_shape_representation_fn_;
  }

  DeviceShapeSizeFn device_shape_size_fn() const override {
    return device_shape_size_fn_;
  }

  int device_count() const override { return pjrt_client_->device_count(); }

  bool HasProperty(HloRunnerPropertyTag::Type tag) const override;

  absl::StatusOr<absl::Nonnull<const HloModule*>> HloModuleFromWrapped(
      const OpaqueExecutable* wrapped) const override;

 private:
  absl::StatusOr<CompileOptions> GenerateDefaultCompileOptions(
      HloModule* module, bool run_hlo_passes);

  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedImpl(
      std::function<absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>(
          absl::Span<const std::vector<PjRtBuffer*>>)>
          execution_helper,
      std::function<int64_t(int64_t)> argument_count_provider,
      std::function<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment);

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> TransferLiteralToDevice(
      const Literal& literal, absl::Nonnull<PjRtMemorySpace*> memory_space,
      const Layout& on_device_layout);
  absl::StatusOr<Literal> TransferLiteralFromDevice(PjRtBuffer& buffer);

  std::unique_ptr<PjRtClient> pjrt_client_;
  DeviceShapeRepresentationFn device_shape_representation_fn_;
  DeviceShapeSizeFn device_shape_size_fn_;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_RUNNER_PJRT_H_
