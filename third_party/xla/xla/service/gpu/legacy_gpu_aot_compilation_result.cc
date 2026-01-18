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

#include "xla/service/gpu/legacy_gpu_aot_compilation_result.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
LegacyGpuAotCompilationResult::FromModule(
    const HloModule* hlo_module, const BufferAssignment* buffer_assignment,
    absl::string_view asm_text, absl::Span<const uint8_t> binary,
    const BinaryMap& dnn_compiled_graphs, int pointer_size,
    Compiler* compiler) {
  tsl::profiler::TraceMe traceme("ResultFromModule");
  GpuExecutableProto proto;
  *proto.mutable_hlo_module_with_config() = hlo_module->ToProtoWithConfig();
  *proto.mutable_buffer_assignment() = buffer_assignment->ToProto();
  proto.set_asm_text(asm_text);
  proto.set_binary(binary.data(), binary.size());
  proto.mutable_dnn_compiled_graphs()->insert(dnn_compiled_graphs.cbegin(),
                                              dnn_compiled_graphs.cend());
  return std::unique_ptr<LegacyGpuAotCompilationResult>(
      new LegacyGpuAotCompilationResult(hlo_module->Clone(), std::move(proto),
                                        pointer_size, compiler));
}

absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
LegacyGpuAotCompilationResult::FromString(const std::string& serialized,
                                          int pointer_size,
                                          Compiler* compiler) {
  tsl::profiler::TraceMe traceme("ResultFromString");
  GpuExecutableProto proto;
  if (!proto.ParseFromString(serialized)) {
    return Internal(
        "Failed to parse serialized LegacyGpuAotCompilationResult.");
  }

  return FromProto(proto, pointer_size, compiler);
}

absl::StatusOr<std::unique_ptr<LegacyGpuAotCompilationResult>>
LegacyGpuAotCompilationResult::FromProto(const GpuExecutableProto& proto,
                                         int pointer_size, Compiler* compiler) {
  tsl::profiler::TraceMe traceme("ResultFromProto");
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      HloModule::CreateFromProtoWithConfig(proto.hlo_module_with_config()));
  return std::unique_ptr<LegacyGpuAotCompilationResult>(
      new LegacyGpuAotCompilationResult(std::move(module), std::move(proto),
                                        pointer_size, compiler));
}

absl::StatusOr<std::string> LegacyGpuAotCompilationResult::SerializeAsString()
    const {
  return proto_.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<Executable>>
LegacyGpuAotCompilationResult::LoadExecutable(
    const se::StreamExecutor* stream_exec) && {
  if (stream_exec == nullptr) {
    return InvalidArgument("Stream executor is null.");
  }

  return compiler_->LoadExecutableFromAotResult(*this, *stream_exec);
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
LegacyGpuAotCompilationResult::buffer_assignment() const {
  auto buffer_size_bytes_function =
      [pointer_size = pointer_size_](const BufferValue& buffer) {
        return gpu::ShapeSizeBytesFunction(pointer_size)(buffer.shape());
      };

  // Recreate BufferAssignment from proto.
  // Technically, we should pass the proper GpuAliasInfo, but the FromProto()
  // method does not actually make use of the MayAlias function. And for now, we
  // don't have backend-specific MustAlias rules.
  // TODO(b/424109294): This needs to be fixed when we implement
  // backend-specific MustAlias rules.
  AliasInfo alias_info;
  return BufferAssignment::FromProto(proto_.buffer_assignment(), module_.get(),
                                     buffer_size_bytes_function, &alias_info);
}

absl::StatusOr<std::string> EarlyExitCompilationResult::SerializeAsString()
    const {
  return Unavailable(
      "SerializeAsString() is not supported by EarlyExitCompilationResult.");
}

absl::StatusOr<std::unique_ptr<Executable>>
EarlyExitCompilationResult::LoadExecutable(
    const se::StreamExecutor* stream_exec) && {
  return Unavailable(
      "LoadExecutable() is not supported by EarlyExitCompilationResult.");
}

absl::StatusOr<std::unique_ptr<BufferAssignment>>
EarlyExitCompilationResult::buffer_assignment() const {
  return Unavailable(
      "buffer_assignment() is not supported by EarlyExitCompilationResult.");
}

}  // namespace gpu
}  // namespace xla
