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
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/cpu/executable.pb.h"
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
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {

// Resolves a nested field descriptor by walking `path` from `root`; every
// element but the last must name a message field. CHECK-fails on a stale path
// rather than silently ignoring nothing.
const tsl::protobuf::FieldDescriptor* FieldByPath(
    const tsl::protobuf::Descriptor* root,
    std::initializer_list<absl::string_view> path) {
  const tsl::protobuf::Descriptor* current = root;
  const tsl::protobuf::FieldDescriptor* field = nullptr;
  for (const absl::string_view name : path) {
    CHECK(current != nullptr)
        << "AOTInterceptionPjrtClient: cannot resolve field '" << name
        << "'; the proto schema changed and this ignore path is stale.";
    field = current->FindFieldByName(name);
    CHECK(field != nullptr)
        << "AOTInterceptionPjrtClient: field '" << name
        << "' not found; the proto schema changed and this ignore path is "
           "stale.";
    current = field->message_type();
  }
  return field;
}

// Runs the structural comparison shared by all backends. `extra_ignored_fields`
// adds to the common, backend-agnostic ignore list.
absl::Status CompareStructurally(
    const HumanReadableAotExecutable& fresh,
    const HumanReadableAotExecutable& golden,
    absl::Span<const tsl::protobuf::FieldDescriptor* const>
        extra_ignored_fields) {
  tsl::protobuf::util::MessageDifferencer differencer;
  differencer.set_message_field_comparison(
      tsl::protobuf::util::MessageDifferencer::EQUIVALENT);

  std::vector<const tsl::protobuf::FieldDescriptor*> ignored_fields = {
      ExecutableAndOptionsProto::descriptor()->FindFieldByName(
          "pjrt_client_name"),
      ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
          "device_ordinal"),
  };
  ignored_fields.insert(ignored_fields.end(), extra_ignored_fields.begin(),
                        extra_ignored_fields.end());
  for (const auto* field : ignored_fields) {
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

}  // namespace

absl::StatusOr<HumanReadableAotExecutable>
AOTInterceptionPjrtClient::DeserializeToHumanReadable(
    absl::string_view serialized, AOTTestPlatform platform) {
  HumanReadableAotExecutable unpacked;

  switch (platform) {
    case AOTTestPlatform::kGpu: {
      ABSL_RETURN_IF_ERROR(
          ReadSplitProto(std::make_unique<riegeli::StringReader<>>(serialized),
                         *unpacked.mutable_executable_and_options()));

      ABSL_RETURN_IF_ERROR(ReadSplitProto(
          riegeli::Maker<riegeli::StringReader>(
              unpacked.executable_and_options().serialized_executable()),
          *unpacked.mutable_gpu_executable()));
      unpacked.mutable_executable_and_options()->clear_serialized_executable();
      return unpacked;
    }
    case AOTTestPlatform::kCpu: {
      // CPU artifacts are plain protos: an ExecutableAndOptionsProto whose
      // `serialized_executable` holds a cpu::CompilationResultProto.
      ExecutableAndOptionsProto& executable_and_options =
          *unpacked.mutable_executable_and_options();
      if (!executable_and_options.ParseFromString(serialized)) {
        return absl::InternalError(
            "AOTInterceptionPjrtClient: failed to parse CPU "
            "ExecutableAndOptionsProto.");
      }
      if (!unpacked.mutable_cpu_executable()->ParseFromString(
              executable_and_options.serialized_executable())) {
        return absl::InternalError(
            "AOTInterceptionPjrtClient: failed to parse CPU "
            "CompilationResultProto.");
      }
      executable_and_options.clear_serialized_executable();
      return unpacked;
    }
  }
  CHECK(false) << "Unsupported platform for AOT testing!";
}

absl::StatusOr<AOTTestPlatform> AOTInterceptionPjrtClient::PlatformFromName(
    absl::string_view platform_name) {
  if (platform_name == "gpu" || platform_name == "cuda" ||
      platform_name == "rocm") {
    return AOTTestPlatform::kGpu;
  }
  if (platform_name == "cpu" || platform_name == "host") {
    return AOTTestPlatform::kCpu;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported platform: ", platform_name));
}

absl::string_view AOTInterceptionPjrtClient::PlatformSubdir(
    AOTTestPlatform platform) {
  switch (platform) {
    case AOTTestPlatform::kGpu:
      return "gpu";
    case AOTTestPlatform::kCpu:
      return "cpu";
  }
  CHECK(false) << "Unsupported platform for AOT testing!";
}

absl::Status AOTInterceptionPjrtClient::CompareGPUExecutables(
    const HumanReadableAotExecutable& fresh,
    const HumanReadableAotExecutable& golden) {
  // The ignored fields hold backend machine code and device-specific details.
  //
  // TODO(b/528258781): Debug options are ignored wholesale. Work out which
  // flags actually affect the artifact and should be compared, versus which are
  // host- or run-specific noise (dump paths, cache dirs), and ignore only
  // those.
  return CompareStructurally(
      fresh, golden,
      {
          gpu::GpuExecutableProto::descriptor()->FindFieldByName("binary"),
          gpu::GpuExecutableProto::descriptor()->FindFieldByName("asm_text"),
          gpu::GpuExecutableProto::descriptor()->FindFieldByName(
              "buffer_allocations"),
          gpu::GpuExecutableProto::descriptor()->FindFieldByName(
              "gpu_compute_capability"),
          // Non-deterministic kernel binaries in custom-kernel thunks. Both
          // oneof variants are ignored, so a ptx<->cubin flip is tolerated.
          stream_executor::KernelLoaderSpecProto::descriptor()->FindFieldByName(
              "ptx"),
          stream_executor::KernelLoaderSpecProto::descriptor()->FindFieldByName(
              "cubin"),
          ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
              "debug_options"),
          HloModuleConfigProto::descriptor()->FindFieldByName("debug_options"),
      });
}

absl::Status AOTInterceptionPjrtClient::CompareGoldenCPUExecutable(
    const HumanReadableAotExecutable& fresh,
    const HumanReadableAotExecutable& golden) {
  // The ignored fields hold compiled machine code, host-specific target details
  // and debug options, which are host- or run-specific noise.
  return CompareStructurally(
      fresh, golden,
      {
          cpu::CompilationResultProto::descriptor()->FindFieldByName(
              "object_files"),
          cpu::CompilationResultProto::descriptor()->FindFieldByName(
              "target_machine_options"),
          cpu::CompilationResultProto::descriptor()->FindFieldByName(
              "data_layout"),
          ExecutableBuildOptionsProto::descriptor()->FindFieldByName(
              "debug_options"),
          HloModuleConfigProto::descriptor()->FindFieldByName("debug_options"),
          // Host/process metadata: a global module-id counter and the intra-op
          // thread-pool size, both of which vary run to run.
          FieldByPath(cpu::CompilationResultProto::descriptor(),
                      {"hlo_module", "hlo_module", "id"}),
          FieldByPath(cpu::CompilationResultProto::descriptor(),
                      {"hlo_module", "config", "intra_op_parallelism_threads"}),
          FieldByPath(cpu::CompilationResultProto::descriptor(),
                      {"thunk_sequence", "thunks", "info", "module_id"}),
      });
}

absl::Status AOTInterceptionPjrtClient::VerifyAgainstGolden(
    const PjRtExecutable& fresh_executable) {
  ABSL_ASSIGN_OR_RETURN(const AOTTestPlatform platform,
                   PlatformFromName(inner_client_->platform_name()));

  ABSL_ASSIGN_OR_RETURN(std::string fresh_serialized,
                   fresh_executable.SerializeExecutable());
  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable fresh_unpacked,
                   DeserializeToHumanReadable(fresh_serialized, platform));
  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable golden,
                   LoadHumanReadableArtifact());

  absl::Status diff_status;
  switch (platform) {
    case AOTTestPlatform::kGpu:
      diff_status = CompareGPUExecutables(fresh_unpacked, golden);
      break;
    case AOTTestPlatform::kCpu:
      diff_status = CompareGoldenCPUExecutable(fresh_unpacked, golden);
      break;
  }

  if (!diff_status.ok()) {
    absl::string_view target_name =
        tsl::io::Basename(tsl::io::Dirname(tsl::io::Dirname(artifact_path_)));
    return absl::Status(
        diff_status.code(),
        absl::StrCat(
            "Golden artifact verification failed for ", artifact_path_, ".\n",
            diff_status.message(), "\n\nRegenerate the goldens for target `",
            target_name,
            "` and add the newly generated v<N+1> directory to the change "
            "after reviewing the differences."));
  }

  return absl::OkStatus();
}

void AOTInterceptionPjrtClient::ApplyAotDeterminismDebugOptions(
    DebugOptions& debug_options) {
  debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);
}

absl::Status AOTInterceptionPjrtClient::DumpGolden(
    const PjRtExecutable& fresh_executable) {
  ABSL_ASSIGN_OR_RETURN(std::string serialized,
                   fresh_executable.SerializeExecutable());
  ABSL_ASSIGN_OR_RETURN(AOTTestPlatform platform,
                   PlatformFromName(inner_client_->platform_name()));
  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable unpacked,
                   DeserializeToHumanReadable(serialized, platform));
  return WriteGoldenTextProto(unpacked, artifact_path_);
}

absl::Status AOTInterceptionPjrtClient::WriteGoldenTextProto(
    const HumanReadableAotExecutable& unpacked, absl::string_view path) {
  tsl::Env* env = tsl::Env::Default();
  ABSL_RETURN_IF_ERROR(
      env->RecursivelyCreateDir(std::string(tsl::io::Dirname(path))));
  // The text-format schema header lets editors and formatters recognize it.
  const tsl::protobuf::Descriptor* descriptor = unpacked.GetDescriptor();
  std::string body;
  if (!tsl::protobuf::TextFormat::PrintToString(unpacked, &body)) {
    return absl::InternalError(
        absl::StrCat("Failed to serialize HumanReadableAotExecutable to text "
                     "format for ",
                     path));
  }
  std::string out =
      absl::StrCat("# proto-file: ", descriptor->file()->name(), "\n",
                   "# proto-message: ", descriptor->full_name(), "\n", body);
  // Write to a sibling temporary and rename, so an interrupted or failed write
  // cannot leave a truncated golden behind for someone to check in.
  const std::string tmp_path = absl::StrCat(path, ".tmp");
  ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(env, tmp_path, out));
  absl::Status renamed = env->RenameFile(tmp_path, std::string(path));
  if (!renamed.ok()) {
    env->DeleteFile(tmp_path).IgnoreError();
    return renamed;
  }
  return absl::OkStatus();
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

// Reads (human-readable) artifact from the storage and re-packs it into the
// ExecutableAndOptions: riegeli-split for GPU, plain proto binary for CPU.
absl::StatusOr<std::string>
AOTInterceptionPjrtClient::PackArtifactForInnerClient() {
  ABSL_ASSIGN_OR_RETURN(HumanReadableAotExecutable unpacked,
                   LoadHumanReadableArtifact());

  ExecutableAndOptionsProto executable_and_options =
      std::move(*unpacked.mutable_executable_and_options());

  // Derived from the artifact, not the live client, to keep un/packing pure. An
  // unset oneof means a malformed golden, not one of the platforms.
  AOTTestPlatform platform;
  switch (unpacked.backend_executable_case()) {
    case HumanReadableAotExecutable::kCpuExecutable:
      platform = AOTTestPlatform::kCpu;
      break;
    case HumanReadableAotExecutable::kGpuExecutable:
      platform = AOTTestPlatform::kGpu;
      break;
    case HumanReadableAotExecutable::BACKEND_EXECUTABLE_NOT_SET:
      return absl::InvalidArgumentError(absl::StrCat(
          "AOTInterceptionPjrtClient: golden artifact has no backend "
          "executable set: ",
          artifact_path_));
  }
  if (platform == AOTTestPlatform::kGpu) {
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
    VLOG(1)
        << "AOTInterceptionPjrtClient: Successfully packed GPU AOT artifact "
           "into ExecutableAndOptions ("
        << serialized.size() << " bytes).";
    return serialized;
  }

  // CPU: re-emit plain protos so the inner CPU client can load the executable.
  *executable_and_options.mutable_serialized_executable() =
      unpacked.cpu_executable().SerializeAsString();
  std::string serialized;
  if (!executable_and_options.SerializeToString(&serialized)) {
    return absl::InternalError(
        "AOTInterceptionPjrtClient: failed to serialize CPU "
        "ExecutableAndOptionsProto.");
  }
  VLOG(1) << "AOTInterceptionPjrtClient: Successfully packed CPU AOT artifact "
             "into ExecutableAndOptions ("
          << serialized.size() << " bytes).";
  return serialized;
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::Compile(const XlaComputation& computation,
                                   CompileOptions options) {
  switch (mode_) {
    case AOTTestMode::kBackwardsCompatibility: {
      VLOG(1) << "AOTInterceptionPjrtClient: Intercepting Compile in "
                 "kBackwardsCompatibility mode for computation ["
              << computation.name()
              << "]. Deserializing executable instead of recompiling.";
      ABSL_ASSIGN_OR_RETURN(std::string serialized, PackArtifactForInnerClient());
      VLOG(1) << "AOTInterceptionPjrtClient: Calling "
                 "inner_client_->DeserializeExecutable.";
      return inner_client_->DeserializeExecutable(serialized,
                                                  std::move(options));
    }
    case AOTTestMode::kUpdateGolden:
    case AOTTestMode::kGoldenVerification: {
      ApplyAotDeterminismDebugOptions(
          *options.executable_build_options.mutable_debug_options());
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> exec,
                       inner_client_->Compile(computation, std::move(options)));
      TF_RET_CHECK(exec != nullptr) << "Compile() returned nullptr";
      ABSL_RETURN_IF_ERROR(mode_ == AOTTestMode::kUpdateGolden
                          ? DumpGolden(*exec)
                          : VerifyAgainstGolden(*exec));
      return exec;
    }
  }
  CHECK(false) << "Unsupported AOT test mode!";
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::CompileAndLoad(const XlaComputation& computation,
                                          CompileOptions options) {
  switch (mode_) {
    case AOTTestMode::kBackwardsCompatibility: {
      VLOG(1) << "AOTInterceptionPjrtClient: Intercepting CompileAndLoad in "
                 "kBackwardsCompatibility mode for computation ["
              << computation.name()
              << "]. Deserializing and loading executable instead of "
                 "recompiling.";
      ABSL_ASSIGN_OR_RETURN(std::string serialized, PackArtifactForInnerClient());
      VLOG(1) << "AOTInterceptionPjrtClient: Calling "
                 "inner_client_->LoadSerializedExecutable.";
      return inner_client_->LoadSerializedExecutable(
          serialized, std::move(options), LoadOptions());
    }
    case AOTTestMode::kUpdateGolden:
    case AOTTestMode::kGoldenVerification: {
      ApplyAotDeterminismDebugOptions(
          *options.executable_build_options.mutable_debug_options());
      ABSL_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtLoadedExecutable> exec,
          inner_client_->CompileAndLoad(computation, std::move(options)));
      PjRtExecutable* fresh_exec = exec->GetExecutable();
      TF_RET_CHECK(fresh_exec != nullptr) << "GetExecutable() returned nullptr";
      ABSL_RETURN_IF_ERROR(mode_ == AOTTestMode::kUpdateGolden
                          ? DumpGolden(*fresh_exec)
                          : VerifyAgainstGolden(*fresh_exec));
      return exec;
    }
  }
  CHECK(false) << "Unsupported AOT test mode!";
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
