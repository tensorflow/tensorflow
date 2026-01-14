/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_CPU_AOT_COMPILATION_RESULT_H_
#define XLA_SERVICE_CPU_CPU_AOT_COMPILATION_RESULT_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/buffer_allocation_info.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

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
                           RelocationModel relocation_model,
                           bool compile_copy_as_llvm_kernel = false);

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
  // Whether to compile copy as LLVM kernel. This is used to avoid dependencies
  // on pjrt/transpose for tfcompiled models.
  bool compile_copy_as_llvm_kernel() const {
    return compile_copy_as_llvm_kernel_;
  }

 private:
  const std::string triple_;
  const std::string cpu_name_;
  const std::string features_;
  const std::string entry_point_name_;
  const RelocationModel relocation_model_;
  const bool compile_copy_as_llvm_kernel_;
};

// This class represents the result of a CPU AOT compilation.
class CpuAotCompilationResult : public CompiledModule {
 public:
  static absl::StatusOr<std::unique_ptr<CpuAotCompilationResult>> Create(
      const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
      absl::string_view function_name, std::vector<ObjFileProto> obj_files,
      std::vector<SymbolProto> symbols, const ThunkSequence& thunks,
      std::unique_ptr<FunctionLibrary> function_library,
      TargetMachineOptionsProto target_machine_options =
          TargetMachineOptionsProto());

  ~CpuAotCompilationResult() override = default;

  absl::StatusOr<std::string> SerializeAsString() const override {
    return proto_.SerializeAsString();
  }

  absl::StatusOr<std::unique_ptr<Executable>>
      LoadExecutable(const se::StreamExecutor* stream_exec) && override;

  const HloModule* optimized_module() const override { return module_.get(); }

  std::shared_ptr<HloModule> shared_optimized_module() override {
    return module_;
  }

  const CompilationResultProto& proto() const { return proto_; }

  std::vector<absl::string_view> obj_files() const {
    std::vector<absl::string_view> obj_files;
    for (const auto& obj_file : proto_.object_files()) {
      obj_files.push_back(obj_file.contents());
    }
    return obj_files;
  }

  std::vector<ObjFileProto> obj_files_protos() const {
    std::vector<ObjFileProto> obj_files;
    for (const auto& obj_file : proto_.object_files()) {
      obj_files.push_back(obj_file);
    }
    return obj_files;
  }

  std::optional<size_t> temp_allocation_index() const {
    return temp_allocation_index_;
  }

  absl::Span<const BufferAllocationInfo> buffer_allocation_infos() const {
    return buffer_allocation_infos_;
  }

  static absl::StatusOr<std::unique_ptr<CpuAotCompilationResult>> FromProto(
      CompilationResultProto proto,
      std::unique_ptr<FunctionLibrary> function_library) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProtoWithConfig(proto.hlo_module()));

    return std::unique_ptr<CpuAotCompilationResult>(new CpuAotCompilationResult(
        proto, std::move(module), std::move(function_library)));
  }

  static absl::StatusOr<std::unique_ptr<CpuAotCompilationResult>> FromString(
      const std::string& serialized,
      std::unique_ptr<FunctionLibrary> function_library) {
    CompilationResultProto proto;
    if (!proto.ParseFromString(serialized)) {
      return Internal("Failed to parse serialized CpuAotCompilationResult.");
    }

    return FromProto(std::move(proto), std::move(function_library));
  }

 private:
  CpuAotCompilationResult(
      const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
      absl::string_view function_name, std::vector<ObjFileProto> obj_files,
      std::vector<SymbolProto> symbols, const ThunkSequenceProto& thunks,
      std::optional<size_t> temp_allocation_index,
      std::vector<BufferAllocationInfo> buffer_allocation_infos,
      std::unique_ptr<FunctionLibrary> function_library,
      TargetMachineOptionsProto target_machine_options);

  explicit CpuAotCompilationResult(
      CompilationResultProto proto, std::unique_ptr<HloModule> module,
      std::unique_ptr<FunctionLibrary> function_library)
      : proto_(std::move(proto)),
        module_(std::move(module)),
        function_library_(std::move(function_library)) {}

  CompilationResultProto proto_;
  std::shared_ptr<HloModule> module_;
  std::optional<size_t> temp_allocation_index_;
  std::vector<BufferAllocationInfo> buffer_allocation_infos_;

  std::unique_ptr<FunctionLibrary> function_library_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_AOT_COMPILATION_RESULT_H_
