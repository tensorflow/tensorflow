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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/stream_executor/platform.h"
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

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_CPU_AOT_COMPILATION_RESULT_H_
