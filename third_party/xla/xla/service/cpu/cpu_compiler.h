/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_CPU_COMPILER_H_
#define XLA_SERVICE_CPU_CPU_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/llvm_compiler.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {
namespace cpu {

// CPU-targeting implementation of the XLA Compiler interface.
//
// The compiler translates XLA HLO code into LLVM IR and uses LLVM's JIT
// infrastructure to create an executable "blob" that can then be returned
// wrapped in CpuExecutable and actually invoked.
class CpuCompiler : public LLVMCompiler {
 public:
  CpuCompiler();
  ~CpuCompiler() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  absl::StatusOr<std::unique_ptr<AotCompilationResult>> Export(
      Executable* executable) const override;

  // Returns a (deserialized) AotCompilationResult from a serialized
  // AotCompilationResult.
  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  LoadAotCompilationResult(const std::string& serialized_aot_result) override;

  absl::StatusOr<HloSchedule> CreateHloSchedule(
      const HloModule& hlo_module) const;

  absl::StatusOr<std::unique_ptr<BufferAssignment>> CreateBufferAssignment(
      const HloModule& module) const;

 private:
  // Initialize the LLVM target.
  static void InitializeLLVMTarget();

  // Runs the HLO passes which are necessary for both optimizations and
  // correctness.
  absl::Status RunHloPasses(HloModule* module, bool is_aot_compile,
                            llvm::TargetMachine* target_machine,
                            const CompileOptions& compile_options);

  // Runs HLO passes up to and including layout assignment.
  absl::Status RunHloPassesThroughLayoutAssn(
      HloModule* module, bool /*is_aot_compile*/,
      TargetMachineFeatures* target_machine_features);

  // Runs HLO passes after layout assignment.
  absl::Status RunHloPassesAfterLayoutAssn(
      HloModule* module, bool is_aot_compile,
      TargetMachineFeatures* target_machine_features,
      const CompileOptions& compile_options);

  absl::StatusOr<std::unique_ptr<CpuExecutable>> CompileCpuExecutable(
      std::unique_ptr<HloModule> module);

  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  CompileAheadOfTimeLegacy(std::unique_ptr<HloModule> module,
                           std::shared_ptr<llvm::TargetMachine> target_machine,
                           const CpuAotCompilationOptions& aot_options,
                           const llvm::Triple& triple,
                           const llvm::PICLevel::Level& pic_level,
                           const llvm::PIELevel::Level& pie_level);

  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  CompileAheadOfTimeThunks(std::unique_ptr<HloModule> module,
                           std::shared_ptr<llvm::TargetMachine> target_machine,
                           const CpuAotCompilationOptions& aot_options,
                           const llvm::Triple& triple,
                           const llvm::PICLevel::Level& pic_level,
                           const llvm::PIELevel::Level& pie_level);

  CpuCompiler(const CpuCompiler&) = delete;
  CpuCompiler& operator=(const CpuCompiler&) = delete;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_COMPILER_H_
