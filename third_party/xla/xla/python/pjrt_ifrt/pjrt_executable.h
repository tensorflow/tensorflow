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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_EXECUTABLE_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// PjRt-compatible `Executable` interface.
class PjRtCompatibleExecutable
    : public llvm::RTTIExtends<PjRtCompatibleExecutable, Executable> {
 public:
  // APIs that allow direct access to `xla::PjRtExecutable` for PjRt-only
  // operations.
  virtual xla::PjRtExecutable* pjrt_executable() = 0;

  static char ID;  // NOLINT
};

// PjRt-compatible `LoadedExecutable` interface.
class PjRtCompatibleLoadedExecutable
    : public llvm::RTTIExtends<PjRtCompatibleLoadedExecutable,
                               LoadedExecutable> {
 public:
  // APIs that allow direct access to `xla::PjRtLoadedExecutable` for PjRt-only
  // operations.
  virtual xla::PjRtLoadedExecutable* pjrt_loaded_executable() = 0;
  virtual std::shared_ptr<xla::PjRtLoadedExecutable>
  shared_ptr_pjrt_loaded_executable() = 0;

  static char ID;  // NOLINT
};

// `Executable` implementation that wraps a `xla::PjRtExecutable`.
class PjRtExecutable final
    : public llvm::RTTIExtends<PjRtExecutable, PjRtCompatibleExecutable> {
 public:
  // Creates PjRtExecutable from xla::PjRtExecutable.
  static absl::StatusOr<std::unique_ptr<Executable>> Create(
      std::shared_ptr<xla::PjRtExecutable> pjrt_executable,
      std::unique_ptr<XlaCompileOptions> compile_options);

  // PjRtCompatibleExecutable implementation.

  xla::PjRtExecutable* pjrt_executable() override {
    DCHECK(this);
    return pjrt_executable_.get();
  }

  // Executable implementation.

  ~PjRtExecutable() override = default;

  absl::string_view name() const override {
    DCHECK(this);
    return pjrt_executable_->name();
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    DCHECK(this);
    return pjrt_executable_->GetParameterShardings();
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    DCHECK(this);
    return pjrt_executable_->GetOutputShardings();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Layout>>> GetParameterLayouts()
      const override {
    DCHECK(this);
    return pjrt_executable_->GetParameterLayouts();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Layout>>> GetOutputLayouts()
      const override {
    DCHECK(this);
    return pjrt_executable_->GetOutputLayouts();
  }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const override;

  absl::StatusOr<std::string> Serialize() const override;

  int num_devices() const override {
    DCHECK(this);
    return pjrt_executable_->num_replicas() *
           pjrt_executable_->num_partitions();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    DCHECK(this);
    return pjrt_executable_->SizeOfGeneratedCodeInBytes();
  }
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    DCHECK(this);
    return pjrt_executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    DCHECK(this);
    return pjrt_executable_->GetHloModules();
  }

  absl::StatusOr<
      absl::flat_hash_map<std::string, Executable::CostAnalysisValue>>
  GetCostAnalysis() const override {
    return pjrt_executable_->GetCostAnalysis();
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return pjrt_executable_->GetOutputMemoryKinds();
  }

  const XlaCompileOptions* GetCompileOptions() const override {
    return compile_options_.get();
  }

  static char ID;  // NOLINT

 protected:
  explicit PjRtExecutable(std::shared_ptr<xla::PjRtExecutable> pjrt_executable,
                          std::unique_ptr<XlaCompileOptions> compile_options)
      : pjrt_executable_(std::move(pjrt_executable)),
        compile_options_(std::move(compile_options)) {}

  std::shared_ptr<xla::PjRtExecutable> pjrt_executable_;
  std::unique_ptr<XlaCompileOptions> compile_options_;
};

// `LoadedExecutable` implementation that wraps a `xla::PjRtLoadedExecutable`.
class PjRtLoadedExecutable final
    : public llvm::RTTIExtends<PjRtLoadedExecutable,
                               PjRtCompatibleLoadedExecutable> {
 public:
  using LoadedExecutable::ExecuteOptions;
  using LoadedExecutable::ExecuteResult;

  // Creates PjRtExecutable from xla::PjRtLoadedExecutable. We expect that
  // xla::PjRtLoadedExecutable has fixed output dtypes/shapes/shardings.
  // PjRtLoadedExecutable::GetHloModules() must be implemented.
  static absl::StatusOr<std::unique_ptr<LoadedExecutable>> Create(
      PjRtCompatibleClient* client,
      std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks);

  // Creates PjRtExecutable from an MHLO or StableHLO MLIR module. We expect
  // that xla::PjRtLoadedExecutable has fixed output dtypes/shapes/shardings. If
  // options.executable_build_options has use_auto_spmd_partitioning or
  // allow_spmd_sharding_propagation_to_output enabled,
  // PjRtLoadedExecutable::GetHloModules() must be implemented.
  static absl::StatusOr<std::unique_ptr<LoadedExecutable>> Create(
      PjRtCompatibleClient* client, mlir::ModuleOp module,
      xla::CompileOptions compile_options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks);

  // PjRtCompatibleLoadedExecutable implementation.

  xla::PjRtLoadedExecutable* pjrt_loaded_executable() override {
    DCHECK(this);
    return pjrt_loaded_executable_.get();
  }
  std::shared_ptr<xla::PjRtLoadedExecutable> shared_ptr_pjrt_loaded_executable()
      override {
    DCHECK(this);
    return pjrt_loaded_executable_;
  }

  // LoadedExecutable implementation.

  ~PjRtLoadedExecutable() override;

  absl::string_view name() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->name();
  }

  Future<> GetReadyFuture() const override {
    // PjRtCompiler blocks until compilation finishes and returns only the
    // executables that are ready.
    return Future<>(absl::OkStatus());
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetParameterShardings();
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetOutputShardings();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Layout>>> GetParameterLayouts()
      const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetParameterLayouts();
  }

  absl::StatusOr<std::vector<std::unique_ptr<Layout>>> GetOutputLayouts()
      const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetOutputLayouts();
  }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const override;

  absl::StatusOr<std::string> Serialize() const override;

  int num_devices() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->num_replicas() *
           pjrt_loaded_executable_->num_partitions();
  }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->SizeOfGeneratedCodeInBytes();
  }
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetHloModules();
  }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->GetOutputMemoryKinds();
  }

  PjRtCompatibleClient* client() const override {
    DCHECK(this);
    return client_;
  }
  absl::StatusOr<ExecuteResult> Execute(
      absl::Span<tsl::RCReference<Array>> args, const ExecuteOptions& options,
      std::optional<DeviceList> devices) override;

  Future<> Delete() override;
  bool IsDeleted() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->IsDeleted();
  }

  absl::Span<const LoadedExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const override {
    DCHECK(this);
    return pjrt_loaded_executable_->addressable_device_logical_ids();
  }
  absl::Span<Device* const> addressable_devices() const override {
    DCHECK(this);
    return addressable_devices_;
  }

  absl::StatusOr<
      absl::flat_hash_map<std::string, Executable::CostAnalysisValue>>
  GetCostAnalysis() const override {
    return pjrt_loaded_executable_->GetCostAnalysis();
  }

  static char ID;  // NOLINT

 private:
  static absl::StatusOr<std::unique_ptr<LoadedExecutable>> CreateInternal(
      PjRtCompatibleClient* client,
      std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
      absl::Span<const xla::PrimitiveType> result_element_types,
      absl::Span<const xla::DimensionVector> result_dimensions,
      const std::optional<xla::HloSharding>& result_hlo_sharding,
      const std::optional<std::vector<absl::string_view>>& result_memory_kinds,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks);

  PjRtLoadedExecutable(
      PjRtCompatibleClient* client,
      std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable,
      DeviceList devices, std::vector<Device*> addressable_devices,
      std::vector<tsl::RCReference<LoadedHostCallback>>
          all_loaded_host_callbacks,
      std::vector<PjRtHostSendAndRecvLoadedHostCallback*>
          host_send_recv_callbacks,
      std::vector<DType> output_dtypes, std::vector<Shape> output_shapes,
      std::vector<std::shared_ptr<const Sharding>> output_shardings);

  PjRtCompatibleClient* client_;
  std::shared_ptr<xla::PjRtLoadedExecutable> pjrt_loaded_executable_;
  // Devices that `pjrt_loaded_executable_` runs on. Empty if the executable is
  // portable.
  DeviceList devices_;
  std::vector<Device*> addressable_devices_;
  std::shared_ptr<std::vector<tsl::RCReference<LoadedHostCallback>>>
      all_loaded_host_callbacks_;
  std::vector<PjRtHostSendAndRecvLoadedHostCallback*> host_send_recv_callbacks_;

  // Output array specs. If the executable is portable, shardings in
  // `output_shardings_` will use an arbitrary addressable device, and will be
  // overridden by a `SingleDeviceSharding` generated on the fly at execution
  // time.
  std::vector<DType> output_dtypes_;
  std::vector<Shape> output_shapes_;
  std::vector<std::shared_ptr<const Sharding>> output_shardings_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_EXECUTABLE_H_
