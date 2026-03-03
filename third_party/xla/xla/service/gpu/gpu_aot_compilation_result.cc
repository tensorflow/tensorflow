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

#include "xla/service/gpu/gpu_aot_compilation_result.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/functional/overload.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_symbol_registry.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
GpuAotCompilationResult::FromProto(GpuExecutableProto executable_proto) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProtoWithConfig(
                          executable_proto.hlo_module_with_config()));
  return absl::WrapUnique(new GpuAotCompilationResult(
      std::move(executable_proto), std::move(module)));
}

absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>>
GpuAotCompilationResult::FromSerialized(
    std::unique_ptr<riegeli::Reader> reader) {
  auto arena = std::make_unique<google::protobuf::Arena>();
  GpuExecutableProto* executable_proto =
      google::protobuf::Arena::Create<GpuExecutableProto>(arena.get());

  TF_RETURN_IF_ERROR(ReadSplitProto(std::move(reader), *executable_proto));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProtoWithConfig(
                          executable_proto->hlo_module_with_config()));
  return absl::WrapUnique(
      new GpuAotCompilationResult(internal::ArenaAllocatedGpuExecutableProto(
                                      std::move(arena), executable_proto),
                                  std::move(module)));
}

absl::StatusOr<std::string> GpuAotCompilationResult::SerializeAsString() const {
  std::string serialized;
  TF_RETURN_IF_ERROR(WriteSplitGpuExecutable(
      GetExecutableProto(),
      std::make_unique<riegeli::StringWriter<>>(&serialized)));
  return serialized;
}

absl::StatusOr<std::unique_ptr<Executable>>
GpuAotCompilationResult::LoadExecutable(
    se::Platform::Id platform_id,
    const se::DeviceDescription& device_description) && {
  const auto symbol_resolver = [&](absl::string_view symbol_name) {
    stream_executor::KernelSymbolRegistry& registry =
        stream_executor::KernelSymbolRegistry::GetGlobalInstance();
    return registry.FindSymbol(symbol_name, platform_id);
  };
  return GpuExecutable::FromProto(GetExecutableProto(), device_description,
                                  platform_id->ToName(),
                                  GetDebugOptionsFromFlags(), symbol_resolver);
}

const GpuExecutableProto& GpuAotCompilationResult::GetExecutableProto() const {
  return std::visit(
      absl::Overload(
          [](const internal::ArenaAllocatedGpuExecutableProto& arena_proto)
              -> const GpuExecutableProto& { return *arena_proto.proto; },
          [](const GpuExecutableProto& stack_proto)
              -> const GpuExecutableProto& { return stack_proto; }),
      gpu_executable_proto_);
}

}  // namespace xla::gpu
