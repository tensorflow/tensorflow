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
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/kernel_spec.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util/split_proto/human_readable_aot_executable.pb.h"
#include "xla/util/split_proto/split_executable_and_options_writer.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

absl::StatusOr<HumanReadableAotExecutable>
AOTInterceptionPjrtClient::DeserializeToHumanReadable(
    absl::string_view serialized) {
  HumanReadableAotExecutable unpacked;

  ABSL_RETURN_IF_ERROR(
      ReadSplitProto(std::make_unique<riegeli::StringReader<>>(serialized),
                     *unpacked.mutable_executable_and_options()));

  std::string serialized_gpu_exec =
      std::move(*unpacked.mutable_executable_and_options()
                     ->mutable_serialized_executable());
  unpacked.mutable_executable_and_options()->clear_serialized_executable();

  ABSL_RETURN_IF_ERROR(
      ReadSplitProto(std::make_unique<riegeli::StringReader<std::string>>(
                         std::move(serialized_gpu_exec)),
                     *unpacked.mutable_gpu_executable()));

  return unpacked;
}

absl::Status AOTInterceptionPjrtClient::CompareGPUExecutables(
    const HumanReadableAotExecutable& fresh,
    const HumanReadableAotExecutable& golden) {
  tsl::protobuf::util::MessageDifferencer differencer;
  differencer.set_message_field_comparison(
      tsl::protobuf::util::MessageDifferencer::EQUIVALENT);

  // TODO(b/528258781): Debug options are currently ignored wholesale in the
  // structural comparison. Evaluate which flags meaningfully affect the
  // compiled artifact and should be compared, versus which are host- or
  // run-specific noise (e.g. dump paths and cache directories) that must be
  // excluded, and ignore only the latter.
  for (const auto* field : {
           gpu::GpuExecutableProto::descriptor()->FindFieldByName("binary"),
           gpu::GpuExecutableProto::descriptor()->FindFieldByName("asm_text"),
           ExecutableAndOptionsProto::descriptor()->FindFieldByName(
               "pjrt_client_name"),
           ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
               "device_ordinal"),
           ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
               "debug_options"),
           HloModuleConfigProto::descriptor()->FindFieldByName("debug_options"),
           gpu::GpuExecutableProto::descriptor()->FindFieldByName(
               "buffer_allocations"),
           gpu::GpuExecutableProto::descriptor()->FindFieldByName(
               "gpu_compute_capability"),
           // `ptx`/`cubin` hold the (non-deterministic) kernel binaries
           // embedded in custom-kernel thunks; excluded like
           // `binary`/`asm_text`.
           stream_executor::KernelLoaderSpecProto::descriptor()
               ->FindFieldByName("ptx"),
           stream_executor::KernelLoaderSpecProto::descriptor()
               ->FindFieldByName("cubin"),
       }) {
    CHECK(field != nullptr)
        << "AOTInterceptionPjrtClient: a proto field descriptor to ignore was "
           "not found; the proto schema changed and this ignore list is stale.";
    differencer.IgnoreField(field);
  }

  std::string diff_string;
  differencer.ReportDifferencesToString(&diff_string);
  // `golden` is the baseline: fields only in the golden are reported as
  // deletions and fields only in the fresh executable as additions.
  if (!differencer.Compare(golden, fresh)) {
    return absl::InternalError(absl::StrCat(
        "Golden Proto structural comparison failed:\n", diff_string));
  }
  return absl::OkStatus();
}

absl::Status AOTInterceptionPjrtClient::VerifyAgainstGolden(
    const PjRtExecutable& fresh_executable) {
  if (inner_client_->platform_name() != "gpu" &&
      inner_client_->platform_name() != "cuda" &&
      inner_client_->platform_name() != "rocm") {
    return absl::UnimplementedError(
        absl::StrCat("AOTInterceptionPjrtClient kGoldenVerification only "
                     "supports GPU platforms. "
                     "Got platform: ",
                     inner_client_->platform_name()));
  }

  ABSL_ASSIGN_OR_RETURN(std::string fresh_serialized,
                   fresh_executable.SerializeExecutable());

  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable fresh_unpacked,
                   DeserializeToHumanReadable(fresh_serialized));

  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable golden,
                   LoadHumanReadableArtifact());
  return CompareGPUExecutables(fresh_unpacked, golden);
}

absl::StatusOr<HumanReadableAotExecutable>
AOTInterceptionPjrtClient::LoadHumanReadableArtifact() {
  VLOG(1) << "AOTInterceptionPjrtClient: Loading serialized executable from: "
          << artifact_path_;
  std::string text_proto;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), artifact_path_, &text_proto));

  HumanReadableAotExecutable unpacked;
  if (!tsl::protobuf::TextFormat::ParseFromString(text_proto, &unpacked)) {
    return absl::InternalError(absl::StrCat(
        "Failed to parse HumanReadableAotExecutable from ", artifact_path_));
  }
  VLOG(1) << "AOTInterceptionPjrtClient: Parsed HumanReadableAotExecutable.";
  return unpacked;
}

// Reads serialized executable and packs it back to ExecutableAndOptions in
// riegeli format.
absl::StatusOr<std::string>
AOTInterceptionPjrtClient::PackArtifactForInnerClient() {
  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable unpacked,
                   LoadHumanReadableArtifact());

  ExecutableAndOptionsProto executable_and_options =
      std::move(*unpacked.mutable_executable_and_options());

  std::string serialized_gpu_exec;
  ABSL_RETURN_IF_ERROR(WriteSplitGpuExecutable(
      unpacked.gpu_executable(),
      std::make_unique<riegeli::StringWriter<>>(&serialized_gpu_exec)));
  *executable_and_options.mutable_serialized_executable() =
      std::move(serialized_gpu_exec);

  std::string serialized;
  ABSL_RETURN_IF_ERROR(WriteSplitExecutableAndOptions(
      executable_and_options,
      std::make_unique<riegeli::StringWriter<>>(&serialized)));
  VLOG(1) << "AOTInterceptionPjrtClient: Successfully packed AOT artifact "
             "into ExecutableAndOptions ("
          << serialized.size() << " bytes).";
  return serialized;
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::Compile(const XlaComputation& computation,
                                   CompileOptions options) {
  if (mode_ == AOTTestMode::kBackwardsCompatibility) {
    VLOG(1) << "AOTInterceptionPjrtClient: Intercepting Compile in "
               "kBackwardsCompatibility mode for computation ["
            << computation.name()
            << "]. Deserializing executable instead of recompiling.";
    ABSL_ASSIGN_OR_RETURN(std::string serialized, PackArtifactForInnerClient());
    VLOG(1) << "AOTInterceptionPjrtClient: Calling "
               "inner_client_->DeserializeExecutable.";
    auto exec_or =
        inner_client_->DeserializeExecutable(serialized, std::move(options));
    if (exec_or.ok()) {
      VLOG(1) << "AOTInterceptionPjrtClient: Successfully deserialized "
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
    VLOG(1) << "AOTInterceptionPjrtClient: Compile called in "
               "kGoldenVerification mode for computation ["
            << computation.name()
            << "]. Compiling fresh and verifying against golden.";
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> exec,
                     inner_client_->Compile(computation, std::move(options)));

    TF_RET_CHECK(exec != nullptr) << "Compile() returned nullptr";
    ABSL_RETURN_IF_ERROR(VerifyAgainstGolden(*exec));
    VLOG(1) << "AOTInterceptionPjrtClient: Golden Verification successful "
            << "for [" << computation.name() << "]";

    return exec;
  }

  return absl::InternalError(
      "Unknown AOTTestMode in AOTInterceptionPjrtClient::Compile");
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::CompileAndLoad(const XlaComputation& computation,
                                          CompileOptions options) {
  if (mode_ == AOTTestMode::kBackwardsCompatibility) {
    VLOG(1) << "AOTInterceptionPjrtClient: Intercepting CompileAndLoad in "
               "kBackwardsCompatibility mode for computation ["
            << computation.name()
            << "]. Deserializing and loading executable instead of "
               "recompiling.";
    ABSL_ASSIGN_OR_RETURN(std::string serialized, PackArtifactForInnerClient());
    VLOG(1) << "AOTInterceptionPjrtClient: Calling "
               "inner_client_->LoadSerializedExecutable.";
    auto loaded_or = inner_client_->LoadSerializedExecutable(
        serialized, std::move(options), LoadOptions());
    if (loaded_or.ok()) {
      VLOG(1) << "AOTInterceptionPjrtClient: Successfully deserialized and "
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
    VLOG(1) << "AOTInterceptionPjrtClient: CompileAndLoad called in "
               "kGoldenVerification mode for computation ["
            << computation.name()
            << "]. Compiling fresh and verifying against golden.";
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtLoadedExecutable> exec,
        inner_client_->CompileAndLoad(computation, std::move(options)));

    PjRtExecutable* fresh_exec = exec->GetExecutable();
    TF_RET_CHECK(fresh_exec != nullptr) << "GetExecutable() returned nullptr";

    ABSL_RETURN_IF_ERROR(VerifyAgainstGolden(*fresh_exec));
    VLOG(1) << "AOTInterceptionPjrtClient: Golden Verification successful "
            << "for [" << computation.name() << "]";

    return exec;
  }

  return absl::InternalError(
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
