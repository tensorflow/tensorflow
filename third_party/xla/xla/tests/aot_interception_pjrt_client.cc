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

#include "xla/tests/aot_interception_pjrt_client.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util/split_proto/human_readable_aot_executable.pb.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

absl::StatusOr<std::unique_ptr<HumanReadableAotExecutable>>
AOTInterceptionPjrtClient::DeserializeToHumanReadable(
    absl::string_view serialized) {
  auto executable_and_options = std::make_unique<ExecutableAndOptionsProto>();
  ABSL_RETURN_IF_ERROR(
      ReadSplitProto(std::make_unique<riegeli::StringReader<>>(serialized),
                     *executable_and_options));

  auto gpu_executable = std::make_unique<gpu::GpuExecutableProto>();
  ABSL_RETURN_IF_ERROR(
      ReadSplitProto(std::make_unique<riegeli::StringReader<>>(
                         executable_and_options->serialized_executable()),
                     *gpu_executable));

  executable_and_options->clear_serialized_executable();

  auto human_readable = std::make_unique<HumanReadableAotExecutable>();
  human_readable->set_allocated_executable_and_options(
      executable_and_options.release());
  human_readable->set_allocated_gpu_executable(gpu_executable.release());
  return human_readable;
}

absl::Status AOTInterceptionPjrtClient::CompareGoldenExecutable(
    const std::unique_ptr<HumanReadableAotExecutable>& fresh,
    const std::unique_ptr<HumanReadableAotExecutable>& golden) {
  tsl::protobuf::util::MessageDifferencer differencer;
  differencer.set_message_field_comparison(
      tsl::protobuf::util::MessageDifferencer::EQUIVALENT);

  for (const auto* field : {
           gpu::GpuExecutableProto::descriptor()->FindFieldByName("binary"),
           gpu::GpuExecutableProto::descriptor()->FindFieldByName("asm_text"),
           gpu::GpuExecutableProto::descriptor()->FindFieldByName(
               "buffer_allocations"),
           HloModuleConfigProto::descriptor()->FindFieldByName("debug_options"),
           ExecutableAndOptionsProto::descriptor()->FindFieldByName(
               "pjrt_client_name"),
           ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
               "device_ordinal"),
           ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
               "debug_options"),
       }) {
    if (field != nullptr) {
      differencer.IgnoreField(field);
    }
  }

  std::string diff_string;
  differencer.ReportDifferencesToString(&diff_string);
  if (!differencer.Compare(*golden, *fresh)) {
    return absl::InternalError(absl::StrCat(
        "Golden Proto structural comparison failed:\n", diff_string));
  }
  return absl::OkStatus();
}

absl::Status AOTInterceptionPjrtClient::VerifyAgainstGolden(
    PjRtExecutable* fresh_executable) {
  ABSL_ASSIGN_OR_RETURN(std::string fresh_serialized,
                   fresh_executable->SerializeExecutable());

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HumanReadableAotExecutable> fresh_human,
                   DeserializeToHumanReadable(fresh_serialized));

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HumanReadableAotExecutable> golden,
                   LoadHumanReadableArtifact());
  return CompareGoldenExecutable(fresh_human, golden);
}

absl::StatusOr<std::unique_ptr<HumanReadableAotExecutable>>
AOTInterceptionPjrtClient::LoadHumanReadableArtifact() {
  LOG(INFO) << "AOTInterceptionPjrtClient: Loading serialized executable "
               "from: "
            << artifact_path_;
  std::string text_proto;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), artifact_path_, &text_proto));

  auto human_readable = std::make_unique<HumanReadableAotExecutable>();
  if (!tsl::protobuf::TextFormat::ParseFromString(text_proto,
                                                  human_readable.get())) {
    return absl::InternalError(absl::StrCat(
        "Failed to parse HumanReadableAotExecutable from ", artifact_path_));
  }
  LOG(INFO) << "AOTInterceptionPjrtClient: Parsed HumanReadableAotExecutable.";
  return human_readable;
}

// Reads serialized executable and packs it back to ExecutableAndOptions in
// riegeli format.
absl::StatusOr<std::string>
AOTInterceptionPjrtClient::LoadSerializedArtifact() {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HumanReadableAotExecutable> human_readable,
                   LoadHumanReadableArtifact());

  ExecutableAndOptionsProto executable_and_options =
      std::move(*human_readable->mutable_executable_and_options());

  std::string serialized_gpu_exec;
  ABSL_RETURN_IF_ERROR(WriteSplitGpuExecutable(
      human_readable->gpu_executable(),
      std::make_unique<riegeli::StringWriter<>>(&serialized_gpu_exec)));
  *executable_and_options.mutable_serialized_executable() =
      std::move(serialized_gpu_exec);

  std::string serialized;
  ABSL_RETURN_IF_ERROR(WriteSplitExecutableAndOptions(
      executable_and_options,
      std::make_unique<riegeli::StringWriter<>>(&serialized)));
  LOG(INFO) << "AOTInterceptionPjrtClient: Successfully packed AOT artifact "
               "into ExecutableAndOptions ("
            << serialized.size() << " bytes).";
  return serialized;
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::Compile(const XlaComputation& computation,
                                   CompileOptions options) {
  if (mode_ == AOTTestMode::kBackwardsCompatibility) {
    LOG(INFO) << "AOTInterceptionPjrtClient: Intercepting Compile in "
                 "kBackwardsCompatibility mode for computation ["
              << computation.name()
              << "]. Deserializing executable instead of recompiling.";
    ABSL_ASSIGN_OR_RETURN(std::string serialized, LoadSerializedArtifact());
    LOG(INFO) << "AOTInterceptionPjrtClient: Calling "
                 "inner_client_->DeserializeExecutable.";
    auto exec_or =
        inner_client_->DeserializeExecutable(serialized, std::move(options));
    if (exec_or.ok()) {
      LOG(INFO) << "AOTInterceptionPjrtClient: Successfully deserialized "
                   "executable for ["
                << computation.name() << "]";
    } else {
      LOG(ERROR) << "AOTInterceptionPjrtClient: Failed to deserialize "
                    "executable for ["
                 << computation.name() << "]: " << exec_or.status();
    }
    return exec_or;
  }
  if (mode_ == AOTTestMode::kGoldenVerification) {
    LOG(INFO) << "AOTInterceptionPjrtClient: Compile called in "
                 "kGoldenVerification mode for computation ["
              << computation.name()
              << "]. Compiling fresh and verifying against golden.";
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> exec,
                     inner_client_->Compile(computation, std::move(options)));

    ABSL_RETURN_IF_ERROR(VerifyAgainstGolden(exec.get()));
    LOG(INFO)
        << "AOTInterceptionPjrtClient: Golden Verification successful for ["
        << computation.name() << "]";

    return exec;
  }

  return absl::InvalidArgumentError(
      "Unknown AOTTestMode in AOTInterceptionPjrtClient::Compile");
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::CompileAndLoad(const XlaComputation& computation,
                                          CompileOptions options) {
  if (mode_ == AOTTestMode::kBackwardsCompatibility) {
    LOG(INFO) << "AOTInterceptionPjrtClient: Intercepting CompileAndLoad in "
                 "kBackwardsCompatibility mode for computation ["
              << computation.name()
              << "]. Deserializing and loading executable instead of "
                 "recompiling.";
    ABSL_ASSIGN_OR_RETURN(std::string serialized, LoadSerializedArtifact());
    LOG(INFO) << "AOTInterceptionPjrtClient: Calling "
                 "inner_client_->LoadSerializedExecutable.";
    auto loaded_or = inner_client_->LoadSerializedExecutable(
        serialized, std::move(options), LoadOptions());
    if (loaded_or.ok()) {
      LOG(INFO) << "AOTInterceptionPjrtClient: Successfully deserialized and "
                   "loaded executable for ["
                << computation.name() << "]";
    } else {
      LOG(ERROR) << "AOTInterceptionPjrtClient: Failed to load serialized "
                    "executable for ["
                 << computation.name() << "]: " << loaded_or.status();
    }
    return loaded_or;
  }
  if (mode_ == AOTTestMode::kGoldenVerification) {
    LOG(INFO) << "AOTInterceptionPjrtClient: CompileAndLoad called in "
                 "kGoldenVerification mode for computation ["
              << computation.name()
              << "]. Compiling fresh and verifying against golden.";
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtLoadedExecutable> exec,
        inner_client_->CompileAndLoad(computation, std::move(options)));

    PjRtExecutable* fresh_exec = exec->GetExecutable();
    if (fresh_exec == nullptr) {
      return absl::InternalError("GetExecutable() returned nullptr");
    }

    ABSL_RETURN_IF_ERROR(VerifyAgainstGolden(fresh_exec));
    LOG(INFO)
        << "AOTInterceptionPjrtClient: Golden Verification successful for ["
        << computation.name() << "]";

    return exec;
  }

  return absl::InvalidArgumentError(
      "Unknown AOTTestMode in AOTInterceptionPjrtClient::CompileAndLoad");
}

absl::StatusOr<PjRtDevice*> AOTInterceptionPjrtClient::LookupDevice(
    GlobalDeviceId global_device_id) const {
  return inner_client_->LookupDevice(global_device_id);
}

absl::StatusOr<PjRtDevice*> AOTInterceptionPjrtClient::LookupAddressableDevice(
    LocalDeviceId local_device_id) const {
  return inner_client_->LookupAddressableDevice(local_device_id);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtMemorySpace* memory_space) {
  return inner_client_->BufferFromHostLiteral(literal, memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                                 PjRtMemorySpace* memory_space,
                                                 const Layout* device_layout) {
  return inner_client_->BufferFromHostLiteral(literal, memory_space,
                                              device_layout);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  return inner_client_->BufferFromHostBuffer(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), memory_space, device_layout);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtBuffer* donated_dst, const Layout* device_layout) {
  return inner_client_->BufferFromHostBuffer(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), donated_dst, device_layout);
}

absl::StatusOr<DeviceAssignment>
AOTInterceptionPjrtClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return inner_client_->GetDefaultDeviceAssignment(num_replicas,
                                                   num_partitions);
}

absl::StatusOr<DeviceAssignment>
AOTInterceptionPjrtClient::GetDefaultDeviceAssignment(
    int num_replicas, std::optional<int> num_replicas_per_slice,
    int num_partitions, const MultiSliceConfig* multi_slice_config) const {
  return inner_client_->GetDefaultDeviceAssignment(
      num_replicas, num_replicas_per_slice, num_partitions, multi_slice_config);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::DeserializeExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  return inner_client_->DeserializeExecutable(serialized, std::move(options));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::Load(std::shared_ptr<PjRtExecutable> executable,
                                const LoadOptions& load_options) {
  return inner_client_->Load(std::move(executable), load_options);
}
}  // namespace xla
