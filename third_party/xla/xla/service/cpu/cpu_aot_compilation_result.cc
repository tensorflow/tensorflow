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

#include "xla/service/cpu/cpu_aot_compilation_result.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_profile_printer_data.pb.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/util.h"

namespace xla::cpu {
using BufferInfo = cpu_function_runtime::BufferInfo;

CpuAotCompilationOptions::CpuAotCompilationOptions(
    std::string triple, std::string cpu_name, std::string features,
    std::string entry_point_name, RelocationModel relocation_model)
    : triple_(std::move(triple)),
      cpu_name_(std::move(cpu_name)),
      features_(std::move(features)),
      entry_point_name_(std::move(entry_point_name)),
      relocation_model_(relocation_model) {}

CpuAotCompilationOptions::~CpuAotCompilationOptions() = default;

se::Platform::Id CpuAotCompilationOptions::PlatformId() const {
  return se::host::kHostPlatformId;
}

CpuAotCompilationResult::CpuAotCompilationResult(
    ObjectFileData object_file_data, std::vector<BufferInfo> buffer_infos,
    int64_t result_buffer_index, std::unique_ptr<HloModule> module,
    std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data)
    : object_file_data_(std::move(object_file_data)),
      buffer_infos_(std::move(buffer_infos)),
      result_buffer_index_(result_buffer_index),
      module_(std::move(module)),
      hlo_profile_printer_data_(std::move(hlo_profile_printer_data)) {}

const HloModule* CpuAotCompilationResult::optimized_module() const {
  return module_.get();
}

std::unique_ptr<HloModule> CpuAotCompilationResult::consume_optimized_module() {
  return std::move(module_);
}

}  // namespace xla::cpu
