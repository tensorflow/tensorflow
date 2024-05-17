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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llvm/Target/TargetMachine.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/target_machine_features.h"
#include "xla/service/cpu/xla_framework.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/service/llvm_compiler.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace xla {
namespace cpu {

class CpuExecutable;

// This class wraps the configurability options that LLVM exposes including: the
// target triple, the target cpu and the target features.  It also includes the
// desired linkage name for the computation entry point.
class CpuAotCompilationOptions : public AotCompilationOptions {
 public:
  // Relocation models available for compilation.
  enum class RelocationModel {
    // Corresponds to the -fno-pic compiler option.
    Static,
    // Corresponds to the -fpic compiler option.
    SmallPic,
    // Corresponds to the -fPIC compiler option.
    BigPic,
    // Corresponds to the -fpie compiler option.
    SmallPie,
    // Corresponds to the -fPIE compiler option.
    BigPie
  };

  CpuAotCompilationOptions(std::string triple, std::string cpu_name,
                           std::string features, std::string entry_point_name,
                           RelocationModel relocation_model);

  ~CpuAotCompilationOptions() override;

  se::Platform::Id PlatformId() const override;

  // The triple used for compilation, similar to clang's -target flag.
  const std::string& triple() const { return triple_; }
  // The CPU name used for compilation, similar to clang's -mcpu flag.
  const std::string& cpu_name() const { return cpu_name_; }
  // The target features used for compilation ("+avx2", "+neon", etc).
  const std::string& features() const { return features_; }
  // The name to be used for the compiled code's entry point.
  const std::string& entry_point_name() const { return entry_point_name_; }
  // The relocation model used for compilation.
  RelocationModel relocation_model() const { return relocation_model_; }

  bool use_mlir_hlo_lowering() const { return use_mlir_hlo_lowering_; }
  void set_use_mlir_hlo_lowering(bool value) { use_mlir_hlo_lowering_ = value; }

 private:
  const std::string triple_;
  const std::string cpu_name_;
  const std::string features_;
  const std::string entry_point_name_;
  const RelocationModel relocation_model_;
  bool use_mlir_hlo_lowering_ = false;
};

class CpuAotCompilationResult : public AotCompilationResult {
 public:
  CpuAotCompilationResult(
      ObjectFileData object_file_data,
      std::vector<cpu_function_runtime::BufferInfo> buffer_infos,
      int64_t result_buffer_index, std::unique_ptr<HloModule> module,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data);
  ~CpuAotCompilationResult() override = default;

  HloProfilePrinterData* hlo_profile_printer_data() const {
    return hlo_profile_printer_data_.get();
  }

  const ObjectFileData& object_file_data() const { return object_file_data_; }
  const std::vector<cpu_function_runtime::BufferInfo>& buffer_infos() const {
    return buffer_infos_;
  }
  int64_t result_buffer_index() const { return result_buffer_index_; }

  const HloModule* optimized_module() const override;
  std::unique_ptr<HloModule> consume_optimized_module() override;

 private:
  // Contains the compiled computation: an object file.
  const ObjectFileData object_file_data_;

  // A list of BufferInfo objects describing the buffers used by the XLA
  // computation.
  const std::vector<cpu_function_runtime::BufferInfo> buffer_infos_;

  // Contains which buffer index into |buffer_sizes| was designated to the
  // result of the computation.  This buffer should be passed into the output
  // parameter when calling the compiled computation.
  const int64_t result_buffer_index_;

  // Contains the optimized HLO module.
  std::unique_ptr<HloModule> module_;

  // Contains an instance of HloProfilePrinterData if HLO profiling is enabled,
  // otherwise is nullptr.
  std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data_;
};

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

  // The optional `registry` supports MLIR dialects and plugins to be loaded
  // during optimization. If non-null, it will be used to construct relevant
  // MLIR contexts.
  absl::StatusOr<std::unique_ptr<CpuExecutable>> CompileXlaRuntimeCpuExecutable(
      std::unique_ptr<HloModule> module,
      mlir::DialectRegistry* registry = nullptr);

 private:
  // Initialize the LLVM target.
  static void InitializeLLVMTarget();

  // Runs the HLO passes which are necessary for both optimizations and
  // correctness.
  absl::Status RunHloPasses(HloModule* module, bool is_aot_compile,
                            llvm::TargetMachine* target_machine,
                            const CompileOptions& compile_options,
                            bool is_mlir_compile = false);

  // Runs HLO passes up to and including layout assignment.
  absl::Status RunHloPassesThroughLayoutAssn(
      HloModule* module, bool /*is_aot_compile*/,
      LLVMTargetMachineFeatures* target_machine_features,
      bool is_mlir_compile = false);

  // Runs HLO passes after layout assignment.
  absl::Status RunHloPassesAfterLayoutAssn(
      HloModule* module, bool is_aot_compile,
      LLVMTargetMachineFeatures* target_machine_features,
      const CompileOptions& compile_options, bool is_mlir_compile);

  absl::StatusOr<std::unique_ptr<CpuExecutable>> CompileLegacyCpuExecutable(
      std::unique_ptr<HloModule> module);

  CpuCompiler(const CpuCompiler&) = delete;
  CpuCompiler& operator=(const CpuCompiler&) = delete;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_COMPILER_H_
