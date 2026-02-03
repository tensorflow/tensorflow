/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_IR_LOADED_EXECUTABLE_H_
#define XLA_PYTHON_IFRT_IR_IFRT_IR_LOADED_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/python/ifrt/mpmd_executable.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

class IfrtIrLoadedExecutable
    : public llvm::RTTIExtends<IfrtIrLoadedExecutable,
                               xla::ifrt::MpmdLoadedExecutable> {
 public:
  // Creates a LoadedExecutable from a compilation result.
  static absl::StatusOr<xla::ifrt::LoadedExecutableRef> Create(
      xla::ifrt::Client* client,
      std::shared_ptr<CompiledIfrtIrProgram> program);

  // LoadedExecutable implementation.

  ~IfrtIrLoadedExecutable() override = default;

  xla::ifrt::Client* client() const override {
    DCHECK(this);
    return client_;
  }

  absl::string_view name() const override {
    DCHECK(this);
    return program_->program_name;
  }

  absl::StatusOr<std::optional<std::string>> Fingerprint() const override;

  absl::StatusOr<std::shared_ptr<const xla::ifrt::ExecutableVersion>>
  executable_version() const override;

  absl::StatusOr<std::string> Serialize() const override;

  absl::StatusOr<std::string> GetHumanReadableProgramText() const override;

  xla::ifrt::UserContextRef user_context() const override {
    return user_context_;
  }

  int num_devices() const override;
  int64_t SizeOfGeneratedCodeInBytes() const override;
  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats()
      const override;

  std::optional<std::vector<xla::OpSharding>> GetParameterShardings()
      const override;

  std::optional<std::vector<xla::OpSharding>> GetOutputShardings()
      const override;

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override;

  absl::StatusOr<absl::Span<const int>> GetDonatableInputIndices()
      const override;

  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override;

  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<xla::ifrt::AttributeMap> GetCostAnalysis() const override;

  absl::StatusOr<ExecuteResult> Execute(
      absl::Span<xla::ifrt::ArrayRef> args, const ExecuteOptions& options,
      std::optional<xla::ifrt::DeviceListRef> devices) override;

  std::optional<xla::ifrt::DeviceListRef> devices() const override;

  absl::Span<xla::ifrt::Device* const> addressable_devices() const override;

  absl::StatusOr<
      absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>>
  GetMpmdAddressableDevices() const override;

  absl::StatusOr<absl::flat_hash_map<std::string, xla::CompiledMemoryStats>>
  GetMpmdCompiledMemoryStats() const override;

  absl::StatusOr<absl::flat_hash_map<
      std::string, std::vector<std::shared_ptr<xla::HloModule>>>>
  GetMpmdHloModules() const override;

  absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>
  GetMpmdCostAnalysis() const override;

  // IFRT IR specific methods.

  // Returns a URL to a generated xprof of predicted memory profile.
  absl::StatusOr<std::string> GetIfrtIrProgramXprofUrl() const;

  // Returns predicted memory stats for the IFRT IR program.
  // This method differs from `GetMpmdCompiledMemoryStats` which returns
  // information per SPMD program. Instead, this method takes into account
  // live arrays and the order of execution to predict memory usage for the
  // entire IFRT IR program.
  absl::StatusOr<IfrtIrProgramMemoryStats> GetIfrtIrProgramMemoryStats() const;

  static char ID;  // NOLINT

 private:
  IfrtIrLoadedExecutable(xla::ifrt::Client* client,
                         std::shared_ptr<CompiledIfrtIrProgram> program,
                         xla::ifrt::DeviceListRef devices,
                         std::unique_ptr<ProgramMemoryTracer> memory_tracer);

  // Returns the layout of a parameter from the consumer executable. If the
  // parameter is not used by any executable or is used by a transfer
  // (ifrt::CopyArraysOp), then the default device layout is returned.
  absl::StatusOr<std::shared_ptr<const xla::PjRtLayout>>
  GetParameterLayoutFromConsumer(mlir::SymbolTableCollection& symbol_table,
                                 mlir::OpOperand& param_operand) const;

  xla::ifrt::Client* client_;
  std::shared_ptr<CompiledIfrtIrProgram> program_;
  xla::ifrt::DeviceListRef devices_;
  std::unique_ptr<ProgramMemoryTracer> memory_tracer_;
  const xla::ifrt::UserContextRef user_context_;

  mutable absl::once_flag version_once_;
  mutable absl::StatusOr<std::shared_ptr<const xla::ifrt::ExecutableVersion>>
      version_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_IFRT_IR_LOADED_EXECUTABLE_H_
