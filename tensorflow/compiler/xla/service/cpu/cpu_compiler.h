/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_COMPILER_H_

#include <memory>

#include "absl/types/span.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace cpu {

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

 private:
  const std::string triple_;
  const std::string cpu_name_;
  const std::string features_;
  const std::string entry_point_name_;
  const RelocationModel relocation_model_;
};

class CpuAotCompilationResult : public AotCompilationResult {
 public:
  CpuAotCompilationResult(
      ObjectFileData object_file_data,
      std::vector<cpu_function_runtime::BufferInfo> buffer_infos,
      int64_t result_buffer_index,
      std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data);
  ~CpuAotCompilationResult();

  HloProfilePrinterData* hlo_profile_printer_data() const {
    return hlo_profile_printer_data_.get();
  }

  const ObjectFileData& object_file_data() const { return object_file_data_; }
  const std::vector<cpu_function_runtime::BufferInfo>& buffer_infos() const {
    return buffer_infos_;
  }
  int64_t result_buffer_index() const { return result_buffer_index_; }

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
  ~CpuCompiler() override {}

  StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModuleGroup> module_group,
      std::vector<std::vector<se::StreamExecutor*>> stream_execs,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      const HloModule* module) override;

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(std::unique_ptr<HloModuleGroup> module_group,
                     const AotCompilationOptions& options) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

 private:
  // Initialize the LLVM target.
  static void InitializeLLVMTarget();

  // Runs the HLO passes which are necessary for both optimizations and
  // correctness.
  Status RunHloPasses(HloModule* module, bool is_aot_compile,
                      llvm::TargetMachine* target_machine);

  // Runs HLO passes up to and including layout assignment.
  Status RunHloPassesThroughLayoutAssn(
      HloModule* module, bool /*is_aot_compile*/,
      LLVMTargetMachineFeatures* target_machine_features);

  // Runs HLO passes after layout assignment.
  Status RunHloPassesAfterLayoutAssn(
      HloModule* module, bool is_aot_compile,
      LLVMTargetMachineFeatures* target_machine_features);

  mutable std::unique_ptr<HloProto> hlo_proto_;

  TF_DISALLOW_COPY_AND_ASSIGN(CpuCompiler);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_COMPILER_H_
