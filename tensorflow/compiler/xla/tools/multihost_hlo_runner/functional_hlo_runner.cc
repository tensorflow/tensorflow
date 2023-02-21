/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/multihost_hlo_runner/functional_hlo_runner.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/tools/hlo_control_flow_flattening.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

namespace {
// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloTextToModule(
    absl::string_view hlo_text) {
  return ParseAndReturnUnverifiedModule(hlo_text);
}

// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloProtoToModule(
    const HloModuleProto& proto) {
  TF_ASSIGN_OR_RETURN(
      HloModuleConfig config,
      HloModule::CreateModuleConfigFromProto(proto, DebugOptions()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(proto, config));
  return std::move(module);
}

template <typename ElementType>
void PopulateWithSameValue(Literal* literal, ElementType val) {
  for (ElementType& element : literal->data<ElementType>()) {
    element = static_cast<ElementType>(val);
  }
}

StatusOr<Literal> MakeFakeLiteralWithSameValue(const Shape& shape, int value) {
  if (!shape.IsArray()) {
    return InvalidArgument(
        "MakeFakeLiteralWithSameValue does not support non-array type");
  }
  Shape new_shape = shape;
  new_shape.mutable_layout()->clear_tiles();
  Literal literal(new_shape);
  switch (new_shape.element_type()) {
    case BF16:
      PopulateWithSameValue(&literal, bfloat16(static_cast<float>(value)));
      break;
    case F16:
      PopulateWithSameValue(&literal, static_cast<half>(value));
      break;
    case F32:
      PopulateWithSameValue(&literal, static_cast<float>(value));
      break;
    case F64:
      PopulateWithSameValue(&literal, static_cast<double>(value));
      break;
    case S8:
      PopulateWithSameValue(&literal, static_cast<int8_t>(value));
      break;
    case U8:
      PopulateWithSameValue(&literal, static_cast<uint8_t>(value));
      break;
    case S16:
      PopulateWithSameValue(&literal, static_cast<int16_t>(value));
      break;
    case U16:
      PopulateWithSameValue(&literal, static_cast<uint16_t>(value));
      break;
    case S32:
      PopulateWithSameValue(&literal, static_cast<int32_t>(value));
      break;
    case U32:
      PopulateWithSameValue(&literal, static_cast<uint32_t>(value));
      break;
    case S64:
      PopulateWithSameValue(&literal, static_cast<int64_t>(value));
      break;
    case U64:
      PopulateWithSameValue(&literal, static_cast<uint64_t>(value));
      break;
    case C64:
      PopulateWithSameValue(&literal,
                            static_cast<complex64>(complex64(value, 0.0)));
      break;
    case C128:
      PopulateWithSameValue(&literal,
                            static_cast<complex128>(complex128(value, 0.0)));
      break;
    case PRED:
      PopulateWithSameValue(&literal, (value % 2) == 0);
      break;
    default:
      return Unimplemented("Unsupported type for fake literal generation: %s",
                           ShapeUtil::HumanString(shape));
  }
  return literal;
}

void AddShardingAnnotationsToSpmdPartitionedModule(HloModule* hlo_module) {
  auto set_manual_sharding = [](HloInstruction* hlo) {
    if (!hlo->has_sharding()) {
      hlo->set_sharding(
          HloSharding::Manual().NormalizeTupleSharding(hlo->shape()));
    }
  };
  for (int64_t i = 0; i < hlo_module->entry_computation()->num_parameters();
       ++i) {
    HloInstruction* param =
        hlo_module->entry_computation()->parameter_instruction(i);
    set_manual_sharding(param);
  }

  HloInstruction* entry_root =
      hlo_module->entry_computation()->root_instruction();
  set_manual_sharding(entry_root);
}

}  // namespace

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error) {
  if (text == "text") {
    *input_format = InputFormat::kText;
    return true;
  }
  if (text == "proto_text") {
    *input_format = InputFormat::kProtoText;
    return true;
  }
  if (text == "proto_binary") {
    *input_format = InputFormat::kProtoBinary;
    return true;
  }
  if (text == "snapshot_proto_binary") {
    *input_format = InputFormat::kSnapshotProtoBinary;
    return true;
  }

  *error = "unknown value for enumeration";
  return false;
}

std::string AbslUnparseFlag(InputFormat input_format) {
  switch (input_format) {
    case InputFormat::kText:
      return "text";
    case InputFormat::kProtoText:
      return "proto_text";
    case InputFormat::kProtoBinary:
      return "proto_binary";
    case InputFormat::kSnapshotProtoBinary:
      return "snapshot_proto_binary";
    default:
      return absl::StrCat(input_format);
  }
}

bool AbslParseFlag(absl::string_view text,
                   FunctionalHloRunner::ModuleArgumentMode* argument_mode,
                   std::string* error) {
  if (text == "use_device_id_as_input") {
    *argument_mode =
        FunctionalHloRunner::ModuleArgumentMode::kUseDeviceIdAsInput;
    return true;
  }
  if (text == "use_random_inputs") {
    *argument_mode = FunctionalHloRunner::ModuleArgumentMode::kUseRandomInputs;
    return true;
  }
  if (text == "use_shared_random_inputs") {
    *argument_mode =
        FunctionalHloRunner::ModuleArgumentMode::kUseSharedRandomInputs;
    return true;
  }
  if (text == "use_zeros_as_input") {
    *argument_mode = FunctionalHloRunner::ModuleArgumentMode::kUseZerosAsInput;
    return true;
  }
  if (text == "uninitialized") {
    *argument_mode = FunctionalHloRunner::ModuleArgumentMode::kUninitialized;
    return true;
  }
  *error =
      "Unrecognized module argument mode specified. Expect "
      "\"use_device_id_as_input\", \"use_random_inputs\", or "
      "\"use_shared_random_inputs\".";
  return false;
}

std::string AbslUnparseFlag(
    FunctionalHloRunner::ModuleArgumentMode argument_mode) {
  switch (argument_mode) {
    case FunctionalHloRunner::ModuleArgumentMode::kUseDeviceIdAsInput:
      return "use_device_id_as_input";
    case FunctionalHloRunner::ModuleArgumentMode::kUseRandomInputs:
      return "use_random_inputs";
    case FunctionalHloRunner::ModuleArgumentMode::kUseSharedRandomInputs:
      return "use_shared_random_inputs";
    case FunctionalHloRunner::ModuleArgumentMode::kUseZerosAsInput:
      return "use_zeros_as_input";
    case FunctionalHloRunner::ModuleArgumentMode::kUninitialized:
      return "uninitialized";
    default:
      LOG(FATAL) << "Unexpected argument mode.";
  }
}

StatusOr<std::unique_ptr<PjRtClient>> FunctionalHloRunner::CreateGpuClient() {
  return GetStreamExecutorGpuClient(
      /*asynchronous=*/true, GpuAllocatorConfig(),
      /*distributed_client=*/nullptr, /*node_id=*/0);
}

StatusOr<ExecutionOptions> FunctionalHloRunner::LoadExecutionOptions(
    absl::string_view path) {
  ExecutionOptions execution_options;
  TF_RETURN_IF_ERROR(tsl::ReadTextOrBinaryProto(
      tsl::Env::Default(), std::string(path), &execution_options));
  return execution_options;
}

StatusOr<CompileOptions> FunctionalHloRunner::CreateCompileOptions(
    const PjRtClient& client,
    const FunctionalHloRunner::RawCompileOptions& raw_options, int task_id) {
  CompileOptions compile_options;
  if (raw_options.execution_options.has_value()) {
    compile_options.executable_build_options =
        CreateExecutableBuildOptionsFromExecutionOptions(
            raw_options.execution_options.value());
  }

  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  ReplicasAndPartitions replicas_and_partitions =
      FunctionalHloRunner::GetReplicasAndPartitions(
          raw_options.execution_options, client.device_count(),
          raw_options.num_replicas, raw_options.num_partitions,
          raw_options.num_slices.value_or(1));
  build_options.set_num_replicas(replicas_and_partitions.replicas);
  build_options.set_num_partitions(replicas_and_partitions.partitions);
  if (raw_options.spmd_mode == SpmdMode::kUseSpmdPartitioning) {
    build_options.set_use_spmd_partitioning(true);
  }
  if (!build_options.has_device_assignment() &&
      !raw_options.num_slices.has_value()) {
    TF_ASSIGN_OR_RETURN(
        DeviceAssignment device_assignment,
        client.GetDefaultDeviceAssignment(replicas_and_partitions.replicas,
                                          replicas_and_partitions.partitions));
    build_options.set_device_assignment(device_assignment);
  }
  DebugOptions& debug_options = *build_options.mutable_debug_options();
  if (task_id == 0) {
    debug_options.set_xla_dump_to(raw_options.xla_dump_to);
    debug_options.set_xla_dump_hlo_as_text(raw_options.xla_text_dump_mode ==
                                           XlaTextDumpMode::kDumpAsText);
    debug_options.set_xla_dump_hlo_as_proto(raw_options.xla_proto_dump_mode ==
                                            XlaProtoDumpMode::kDumpAsProto);
  }

  switch (raw_options.hlo_passes_mode) {
    case HloPassesMode::kRunXLABackendOnly:
      build_options.set_run_backend_only(true);
      break;
    case HloPassesMode::kDisableAllHloPasses:
      debug_options.set_xla_disable_all_hlo_passes(true);
      break;
    case HloPassesMode::kStandardCompile:
      // Just use the default.
      break;
  }
  return compile_options;
}

FunctionalHloRunner::ReplicasAndPartitions
FunctionalHloRunner::GetReplicasAndPartitionsInternal(
    const std::optional<ExecutionOptions>& execution_options, int device_count,
    const std::optional<int>& num_replicas,
    const std::optional<int>& num_partitions, int num_slices) {
  if (num_replicas.has_value() && num_partitions.has_value()) {
    return ReplicasAndPartitions{num_replicas.value(), num_partitions.value()};
  }
  if (execution_options.has_value()) {
    return ReplicasAndPartitions{execution_options->num_replicas(),
                                 execution_options->num_partitions()};
  }
  if (num_replicas.has_value()) {
    return ReplicasAndPartitions{
        num_replicas.value(), device_count * num_slices / num_replicas.value()};
  }
  if (num_partitions.has_value()) {
    return ReplicasAndPartitions{
        device_count * num_slices / num_partitions.value(),
        num_partitions.value()};
  }
  return ReplicasAndPartitions{device_count * num_slices, 1};
}

FunctionalHloRunner::ReplicasAndPartitions
FunctionalHloRunner::GetReplicasAndPartitions(
    const std::optional<ExecutionOptions>& execution_options, int device_count,
    const std::optional<int>& num_replicas,
    const std::optional<int>& num_partitions, int num_slices) {
  CHECK_GE(num_slices, 1);
  ReplicasAndPartitions result = GetReplicasAndPartitionsInternal(
      execution_options, device_count, num_replicas, num_partitions,
      num_slices);
  VLOG(1) << "Calculated replicas: " << result.replicas
          << ", partitions: " << result.partitions;
  CHECK_GE(result.replicas, 1);
  CHECK_GE(result.partitions, 1);
  return result;
}

ExecutableBuildOptions
FunctionalHloRunner::CreateExecutableBuildOptionsFromExecutionOptions(
    const ExecutionOptions& execution_options) {
  ExecutableBuildOptions build_options;
  if (execution_options.has_debug_options()) {
    *build_options.mutable_debug_options() = execution_options.debug_options();
    build_options.mutable_debug_options()->set_xla_dump_to("");
  }
  if (execution_options.has_shape_with_output_layout()) {
    build_options.set_result_layout(
        Shape(execution_options.shape_with_output_layout()));
  }
  build_options.set_num_replicas(execution_options.num_replicas());
  build_options.set_num_partitions(execution_options.num_partitions());
  build_options.set_use_spmd_partitioning(
      execution_options.use_spmd_partitioning());
  build_options.set_use_auto_spmd_partitioning(
      execution_options.use_auto_spmd_partitioning());
  build_options.set_deduplicate_hlo(execution_options.deduplicate_hlo());
  build_options.set_allow_spmd_sharding_propagation_to_output(
      execution_options.allow_spmd_sharding_propagation_to_output());
  if (execution_options.has_device_assignment()) {
    StatusOr<std::unique_ptr<DeviceAssignment>> device_assignment =
        DeviceAssignment::Deserialize(execution_options.device_assignment());
    TF_CHECK_OK(device_assignment.status());
    build_options.set_device_assignment(**device_assignment);
  }
  build_options.set_alias_passthrough_params(
      execution_options.alias_passthrough_params());
  return build_options;
}

Status FunctionalHloRunner::DumpOutput(
    const FunctionalHloRunner::PerDeviceLiteralVecType& output,
    absl::string_view dump_output_to, int task_id) {
  std::vector<std::string> output_path_vec =
      absl::StrSplit(dump_output_to, '.');
  std::string suffix = output_path_vec.back();
  output_path_vec.pop_back();
  output_path_vec.push_back(absl::StrCat("task_", task_id));
  output_path_vec.push_back("");
  int device_id_index = output_path_vec.size() - 1;
  output_path_vec.push_back("");
  int literal_id_index = output_path_vec.size() - 1;
  output_path_vec.push_back(suffix);
  for (const auto& [device_id, literal_vec] : output) {
    output_path_vec[device_id_index] = absl::StrCat("device_", device_id);
    for (int literal_id = 0; literal_id < literal_vec.size(); ++literal_id) {
      output_path_vec[literal_id_index] = absl::StrCat("literal_", literal_id);
      std::string literal_path = absl::StrJoin(output_path_vec, ".");
      CHECK_EQ(suffix, std::string("txt"));
      Status write_status =
          tsl::WriteStringToFile(tsl::Env::Default(), literal_path,
                                 literal_vec[literal_id].ToString());
      if (!write_status.ok()) {
        return write_status;
      }
    }
  }
  return OkStatus();
}

absl::Span<PjRtDevice* const> FunctionalHloRunner::GetLocalDevices(
    const PjRtClient& client) {
  return client.addressable_devices();
}

StatusOr<FunctionalHloRunner::HloModuleAndArguments>
FunctionalHloRunner::LoadHloModuleAndArguments(absl::string_view hlo_file,
                                               InputFormat input_format) {
  HloModuleAndArguments hlo_module_and_arguments;
  switch (input_format) {
    case InputFormat::kText: {
      std::string hlo_text;
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                          ReadModuleFromHloTextFile(hlo_file));
    } break;
    case InputFormat::kProtoText: {
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                          ReadModuleFromTextProtoFile(hlo_file));
    } break;
    case InputFormat::kProtoBinary: {
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                          ReadModuleFromBinaryProtoFile(hlo_file));
    } break;
    case InputFormat::kSnapshotProtoBinary: {
      TF_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                          ReadModuleFromSnapshotBinaryProtoFile(hlo_file));
    } break;
    default:
      LOG(FATAL) << "Cannot process input format: "
                 << AbslUnparseFlag(input_format);
  }
  return hlo_module_and_arguments;
}

Status FunctionalHloRunner::LoadAndRunAndDump(
    PjRtClient& client,
    const xla::FunctionalHloRunner::PreprocessingOptions& preproc_options,
    const xla::FunctionalHloRunner::RawCompileOptions& raw_compile_options,
    const xla::FunctionalHloRunner::RunningOptions& running_options,
    absl::Span<const std::string> hlo_files, InputFormat input_format,
    std::string dump_output_to, int task_id) {
  TF_ASSIGN_OR_RETURN(CompileOptions compile_options,
                      FunctionalHloRunner::CreateCompileOptions(
                          client, raw_compile_options, task_id));
  TF_ASSIGN_OR_RETURN(FunctionalHloRunner::PerDeviceLiteralVecType output,
                      FunctionalHloRunner::LoadAndRun(
                          client, preproc_options, compile_options,
                          running_options, hlo_files, input_format));
  return dump_output_to.empty()
             ? OkStatus()
             : FunctionalHloRunner::DumpOutput(output, dump_output_to, task_id);
}

StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType>
FunctionalHloRunner::LoadAndRun(PjRtClient& client,
                                const PreprocessingOptions& preproc_options,
                                const CompileOptions& compile_options,
                                const RunningOptions& running_options,
                                absl::Span<const std::string> hlo_files,
                                InputFormat input_format,
                                const PerDeviceLiteralVecType& arguments) {
  // We only support SPMD as of now, i.e., all devices are supposed
  // to execute the same HLO module.
  // Currently there is no mechanism to map the loaded arguments to
  // proper device ID, so loading and executing from HLO snapshot might not
  // replay the original execution.
  HloModuleAndArguments hlo_module_and_arguments;
  PerDeviceLiteralVecType loaded_arguments;
  for (int i = 0; i < hlo_files.size(); ++i) {
    TF_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                        LoadHloModuleAndArguments(hlo_files[i], input_format));
    if (input_format == InputFormat::kSnapshotProtoBinary) {
      CHECK_GE(client.devices().size(), hlo_files.size());
      loaded_arguments[client.devices()[i]->id()] =
          std::move(hlo_module_and_arguments.arguments);
    }
  }
  if (!arguments.empty()) {
    return CompileAndRun(client, preproc_options, compile_options,
                         running_options,
                         hlo_module_and_arguments.hlo_module.get(), arguments);
  }
  return CompileAndRun(
      client, preproc_options, compile_options, running_options,
      hlo_module_and_arguments.hlo_module.get(), loaded_arguments);
}

StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType>
FunctionalHloRunner::LoadAndRun(
    PjRtClient& client, const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options,
    const RunningOptions& running_options,
    absl::Span<const std::string> hlo_files, InputFormat input_format,
    const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& per_device_index_vec) {
  CHECK(!hlo_files.empty());
  // We only support SPMD as of now, i.e., all devices are supposed
  // to execute the same HLO module.
  HloModuleAndArguments hlo_module_and_arguments;
  TF_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                      LoadHloModuleAndArguments(hlo_files[0], input_format));
  return CompileAndRun(client, preproc_options, compile_options,
                       running_options,
                       hlo_module_and_arguments.hlo_module.get(),
                       argument_literals, per_device_index_vec);
}

StatusOr<std::unique_ptr<HloModule>>
FunctionalHloRunner::ReadModuleFromHloTextFile(absl::string_view hlo_file) {
  std::string hlo_string;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                           std::string(hlo_file), &hlo_string));
  return ParseAndReturnUnverifiedModule(hlo_string);
}

StatusOr<std::unique_ptr<HloModule>>
FunctionalHloRunner::ReadModuleFromBinaryProtoFile(absl::string_view hlo_file) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  return HloProtoToModule(proto.hlo_module());
}

StatusOr<std::unique_ptr<HloModule>>
FunctionalHloRunner::ReadModuleFromTextProtoFile(absl::string_view hlo_file) {
  HloProto proto;
  TF_RETURN_IF_ERROR(
      tsl::ReadTextProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  return HloProtoToModule(proto.hlo_module());
}

StatusOr<FunctionalHloRunner::HloModuleAndArguments>
FunctionalHloRunner::ReadModuleFromSnapshotBinaryProtoFile(
    absl::string_view hlo_file) {
  HloSnapshot proto;
  HloModuleAndArguments hlo_module_and_arguments;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  hlo_module_and_arguments.arguments.resize(proto.arguments_size());
  for (int i = 0; i < proto.arguments_size(); i++) {
    TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.arguments[i],
                        Literal::CreateFromProto(proto.arguments()[i]));
  }
  TF_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                      HloProtoToModule(proto.hlo().hlo_module()));
  return hlo_module_and_arguments;
}

StatusOr<std::unique_ptr<HloModule>> FunctionalHloRunner::ReadModuleFromString(
    absl::string_view hlo_text) {
  return HloTextToModule(hlo_text);
}

StatusOr<std::unique_ptr<HloModule>> FunctionalHloRunner::ReadModuleFromProto(
    const HloModuleProto& proto) {
  return HloProtoToModule(proto);
}

StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType>
FunctionalHloRunner::CompileAndRun(PjRtClient& client,
                                   const PreprocessingOptions& preproc_options,
                                   const CompileOptions& compile_options,
                                   const RunningOptions& running_options,
                                   HloModule* hlo_module,
                                   const PerDeviceLiteralVecType& arguments) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      Compile(client, hlo_module, preproc_options, compile_options));

  return Run(client, executable.get(), arguments, running_options);
}

StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType>
FunctionalHloRunner::CompileAndRun(
    PjRtClient& client, const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options,
    const RunningOptions& running_options, HloModule* hlo_module,
    const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& argument_indices) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      Compile(client, hlo_module, preproc_options, compile_options));
  return Run(client, executable.get(), argument_literals, argument_indices,
             running_options);
}

namespace {

// Argument buffers are created on device at the first time an HLO module
// is executed. We reuse argument buffers in the following repeated
// executions whenever possible. We take the following strategy to
// maximally reuse on-device argument buffers which compiles and executes
// the HLO module differently depending on the number of parameters and the
// shape of the parameters of the HLO module. We have the following 3 cases.
// 1. The number of parameters is 1 and it has a shape of tuple of arrays.
// 2. The number of parameters is 1 or many and they are all arrays.
// 3. The rest: this should be rare and we don't expect this to happen with
// JAX.
//
// Case 1: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = true
// and the HLO module is executed with
// ExecuteOptions::arguments_are_tupled = false.
// This enables PjRtClient::Execute to assemble the tupled arguments from
// a flat list of buffers.
// Additionally, we set ExecuteOptions::untuple_result = true if the module's
// output is a tuple. Thus we can use the aliased output buffer as input
// arguments and reuse the non-aliased argument buffers. In this mode, users may
// provide the argument literals as a list of tuples (for the convenience of
// future use cases) or a tuple literal (to support existing use cases).
//
// Case 2: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = false
// and the HLO module is executed with
// ExecuteOptions::arguments_are_tupled = false.
// Same as above, we set ExecuteOptions::untuple_result = true if the module's
// output is a tuple. This allows us to reuse on-device buffers in the same way
// as case 1.
//
// Case 3: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = false
// and the HLO module is executed with
// ExecuteOptions::arguments_are_tupled = false.
// Additionally, we set ExecuteOptions::untuple_result = false.
// We will create new on-device buffers for each repeated execution.

enum class ParameterType {
  kOneTupleOfArrays = 0,
  kOneListOfArrays = 1,
  kOther = 2
};

ParameterType GetParameterType(const HloModule& module) {
  int num_parameters = module.entry_computation()->num_parameters();
  if (num_parameters == 1) {
    const Shape& shape =
        module.entry_computation()->parameter_instruction(0)->shape();
    if (shape.IsTuple()) {
      bool is_tuple_of_arrays = absl::c_all_of(
          shape.tuple_shapes(),
          [](const Shape& subshape) { return subshape.IsArray(); });
      if (is_tuple_of_arrays) {
        return ParameterType::kOneTupleOfArrays;
      }
      return ParameterType::kOther;
    }
  }
  bool is_list_of_arrays =
      absl::c_all_of(module.entry_computation()->parameter_instructions(),
                     [](const HloInstruction* parameter) {
                       return parameter->shape().IsArray();
                     });
  return is_list_of_arrays ? ParameterType::kOneListOfArrays
                           : ParameterType::kOther;
}

}  // namespace

Status FunctionalHloRunner::PrepareHloModuleForCompilation(
    HloModule* hlo_module, const PreprocessingOptions& preproc_options) {
  if (preproc_options.is_spmd_partitioned_module()) {
    // If the module has already been partitioned by SPMD, add sharding
    // annotations (replicated) to module parameters and result.
    AddShardingAnnotationsToSpmdPartitionedModule(hlo_module);
  }

  if (preproc_options.flatten_while_loop() ||
      preproc_options.remove_infeed_outfeed) {
    int while_execution_count =
        preproc_options.while_execution_count.value_or(0);
    HloControlFlowFlattening hlo_control_flow_flattening(
        HloControlFlowFlattening::Options{
            /*while_execution_count=*/while_execution_count,
            /*max_outer_loop_count=*/
            while_execution_count,
            /*max_loop_count=*/
            while_execution_count,
            /*remove_infeed_outfeed=*/preproc_options.remove_infeed_outfeed,
            /*flatten_while_loop=*/preproc_options.flatten_while_loop(),
            /*remove_comm=*/false, /*remove_host_transfer=*/true});
    TF_RETURN_IF_ERROR(hlo_control_flow_flattening.Run(hlo_module).status());
  }
  return OkStatus();
}

CompileOptions FunctionalHloRunner::CompleteCompileOptions(
    const HloModule& hlo_module, CompileOptions compile_options) {
  ParameterType parameter_type = GetParameterType(hlo_module);
  compile_options.parameter_is_tupled_arguments =
      (parameter_type == ParameterType::kOneTupleOfArrays);
  return compile_options;
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> FunctionalHloRunner::Compile(
    PjRtClient& client, HloModule* hlo_module,
    const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options) {
  TF_RETURN_IF_ERROR(
      PrepareHloModuleForCompilation(hlo_module, preproc_options));
  CompileOptions modified_compile_options =
      CompleteCompileOptions(*hlo_module, compile_options);
  XlaComputation computation(hlo_module->ToProto());
  VLOG(1) << "FunctionalHloRunner: compilation started.";
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      client.Compile(computation, modified_compile_options));
  VLOG(1) << "FunctionalHloRunner: compile succeeded.";
  return executable;
}

// Runs the executable and may repeat for multiple times.
// Since the input buffers may be donated by the PjrtClient, we re-create the
// input PjrtBuffers for each repetition.
StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType> FunctionalHloRunner::Run(
    PjRtClient& client, PjRtLoadedExecutable* executable,

    const PerDeviceLiteralVecType& arguments,
    const RunningOptions& running_options) {
  auto create_argument_buffers_on_device = [&client, &executable, &arguments,
                                            &running_options](
                                               bool flatten_tupled_arguments) {
    if (arguments.empty()) {
      return CreateArgumentsOnDevice(client, executable, running_options,
                                     flatten_tupled_arguments);
    }

    if (flatten_tupled_arguments && arguments.begin()->second.size() == 1 &&
        arguments.begin()->second.front().shape().IsTuple()) {
      PerDeviceLiteralVecType flattened_arguments;
      for (const auto& device_id_and_arguments : arguments) {
        Literal tupled_argument =
            device_id_and_arguments.second.front().Clone();
        LiteralVec flattened_argument = tupled_argument.DecomposeTuple();
        int device_id = device_id_and_arguments.first;
        flattened_arguments.insert({device_id, std::move(flattened_argument)});
      }
      return CopyArgumentsToDevice(client, executable->addressable_devices(),
                                   flattened_arguments,
                                   running_options.log_input_output());
    }
    // If the per-device argument is not a single tuple, we ignore the
    // flatten_tupled_arguments parameter and assume the provided arguments have
    // already been flattened.
    return CopyArgumentsToDevice(client, executable->addressable_devices(),
                                 arguments, running_options.log_input_output());
  };
  return RunInternal(client, executable, create_argument_buffers_on_device,
                     running_options);
}

// Runs the executable and may repeat for multiple times.
// Since the input buffers may be donated by the PjrtClient, we re-create the
// input PjrtBuffers for each repetition.
StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType> FunctionalHloRunner::Run(

    PjRtClient& client, PjRtLoadedExecutable* executable,
    const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& argument_indices,
    const RunningOptions& running_options) {
  auto create_argument_buffers_on_device = [&client, &executable,
                                            &argument_literals,
                                            &argument_indices,
                                            &running_options](
                                               bool flatten_arguments) {
    CHECK_GE(argument_literals.size(), 1);
    bool arguments_can_be_flattened = absl::c_all_of(
        argument_literals,
        [](const Literal& literal) { return literal.shape().IsTuple(); });
    arguments_can_be_flattened &= absl::c_all_of(
        argument_indices, [](PerDeviceIndexVecType::const_reference
                                 device_id_and_argument_indices) {
          return device_id_and_argument_indices.second.size() == 1;
        });
    if (flatten_arguments && arguments_can_be_flattened) {
      int tuple_shape_size =
          argument_literals.front().shape().tuple_shapes_size();
      LiteralVec flattened_argument_literals;
      for (const Literal& tupled_argument : argument_literals) {
        LiteralVec flattened_arguments =
            tupled_argument.Clone().DecomposeTuple();
        for (Literal& flattened_argument : flattened_arguments) {
          flattened_argument_literals.push_back(std::move(flattened_argument));
        }
      }
      PerDeviceIndexVecType flattened_per_device_index_vec;
      for (const auto& device_id_and_argument_indices : argument_indices) {
        std::vector<int> flattened_argument_indices(tuple_shape_size);
        int tupled_argument_index =
            device_id_and_argument_indices.second.front();
        for (int i = 0; i < tuple_shape_size; i++) {
          flattened_argument_indices[i] =
              tupled_argument_index * tuple_shape_size + i;
        }
        int device_id = device_id_and_argument_indices.first;
        flattened_per_device_index_vec.insert(
            {device_id, std::move(flattened_argument_indices)});
      }
      return CopyArgumentsToDevice(client, executable->addressable_devices(),
                                   flattened_argument_literals,
                                   flattened_per_device_index_vec,
                                   running_options.log_input_output());
    }
    return CopyArgumentsToDevice(client, executable->addressable_devices(),
                                 argument_literals, argument_indices,
                                 running_options.log_input_output());
  };
  return RunInternal(client, executable, create_argument_buffers_on_device,
                     running_options);
}

namespace {

std::vector<std::vector<PjRtBuffer*>> CreateArgumentPointersFromDeviceBuffers(
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> device_buffers) {
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs(device_buffers.size());
  for (int i = 0; i < device_buffers.size(); i++) {
    argument_ptrs[i].resize(device_buffers[i].size());
    for (int j = 0; j < device_buffers[i].size(); j++) {
      argument_ptrs[i][j] = device_buffers[i][j].get();
    }
  }
  return argument_ptrs;
}

std::vector<std::vector<PjRtBuffer*>> CreateArgumentPointersBasedOnAliasing(
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers,
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> input_buffers,
    std::function<std::optional<int64_t>(int64_t)> get_output_buffer_index) {
  int num_arguments = input_buffers.front().size();
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs(output_buffers.size());
  for (int i = 0; i < input_buffers.size(); i++) {
    argument_ptrs[i].resize(num_arguments);
    for (int argument_index = 0; argument_index < num_arguments;
         argument_index++) {
      std::optional<int> output_buffer_index =
          get_output_buffer_index(argument_index);
      if (!output_buffer_index.has_value()) {
        argument_ptrs[i][argument_index] =
            input_buffers[i][argument_index].get();
      } else {
        argument_ptrs[i][argument_index] =
            output_buffers[i][*output_buffer_index].get();
      }
    }
  }
  return argument_ptrs;
}

std::vector<Shape> GetArgumentShapes(const HloModule& module) {
  const std::vector<HloInstruction*>& params =
      module.entry_computation()->parameter_instructions();
  std::vector<Shape> argument_shapes;
  argument_shapes.reserve(params.size());
  for (int i = 0; i < static_cast<int>(params.size()); ++i) {
    const HloModuleConfig& module_config = module.config();
    argument_shapes.push_back((module_config.has_entry_computation_layout() &&
                               module_config.entry_computation_layout()
                                   .parameter_layout(i)
                                   .shape()
                                   .is_static())
                                  ? module_config.entry_computation_layout()
                                        .parameter_layout(i)
                                        .shape()
                                  : params[i]->shape());
  }
  return argument_shapes;
}

Status EnsureSingleTupleForFlattening(const HloModule& module) {
  if (module.entry_computation()->num_parameters() != 1) {
    return InvalidArgument(
        "Flattening arguments requires the number of parameters to be 1. "
        "The actual number of parameters is %d",
        module.entry_computation()->num_parameters());
  }
  if (!module.entry_computation()
           ->parameter_instructions()
           .front()
           ->shape()
           .IsTuple()) {
    return InvalidArgument(
        "Flattening arguments requires the module parameter to be a single "
        "tuple. But the actual parameter shape is %s",
        module.entry_computation()
            ->parameter_instructions()
            .front()
            ->shape()
            .ToString());
  }
  return OkStatus();
}

}  // namespace

StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType>
FunctionalHloRunner::RunInternal(
    PjRtClient& client, PjRtLoadedExecutable* executable,
    std::function<
        StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(bool)>
        create_argument_buffers_on_device,
    const RunningOptions& running_options) {
  ExecuteOptions execute_options;
  if (running_options.multi_slice_config != nullptr) {
    execute_options.multi_slice_config = running_options.multi_slice_config;
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->GetHloModules());
  CHECK_EQ(hlo_modules.size(), 1);
  const HloModule& module = *(hlo_modules.front());
  ParameterType parameter_type = GetParameterType(module);
  bool flatten_arguments = parameter_type == ParameterType::kOneTupleOfArrays;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> device_buffers,
      create_argument_buffers_on_device(flatten_arguments));
  auto get_output_index_for_one_tuple_of_arrays =
      [&module](int64_t parameter_index) -> std::optional<int64_t> {
    const HloInputOutputAliasConfig& alias_config =
        module.input_output_alias_config();
    std::optional<ShapeIndex> output_index =
        alias_config.GetAliasedOutput(0, {parameter_index});
    if (!output_index.has_value()) {
      return std::nullopt;
    }
    // If the HLO module output is a tuple, it should have been untupled by
    // PjRt. Therefore, we return the tuple index of the buffer.
    if (module.entry_computation()->root_instruction()->shape().IsTuple()) {
      return std::optional<int64_t>(output_index->front());
    }
    CHECK(output_index->empty());
    return 0;
  };
  auto get_output_index_for_one_list_of_arrays =
      [&module](int64_t parameter_index) -> std::optional<int64_t> {
    const HloInputOutputAliasConfig& alias_config =
        module.input_output_alias_config();
    std::optional<ShapeIndex> output_index =
        alias_config.GetAliasedOutput(parameter_index, {});
    if (!output_index.has_value()) {
      return std::nullopt;
    }
    if (module.entry_computation()->root_instruction()->shape().IsTuple()) {
      return std::optional<int64_t>(output_index->front());
    }
    CHECK(output_index->empty());
    return 0;
  };

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> output_buffers;
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs =
      CreateArgumentPointersFromDeviceBuffers(device_buffers);
  bool default_untuple_result = execute_options.untuple_result;
  switch (parameter_type) {
    case ParameterType::kOneTupleOfArrays:
      execute_options.arguments_are_tupled = false;
      execute_options.untuple_result =
          module.entry_computation()->root_instruction()->shape().IsTuple();
      break;
    case ParameterType::kOneListOfArrays:
      execute_options.arguments_are_tupled = false;
      execute_options.untuple_result =
          module.entry_computation()->root_instruction()->shape().IsTuple();
      break;
    case ParameterType::kOther:
      execute_options.arguments_are_tupled = false;
      execute_options.untuple_result = false;
      break;
  }
  for (int repeat = 0; repeat < running_options.num_repeats; ++repeat) {
    VLOG(1) << "FunctionalHloRunner: ExecuteOnDevices started (repeat = "
            << repeat << ").";
    if (repeat == running_options.num_repeats - 1) {
      execute_options.untuple_result = default_untuple_result;
    }
    TF_ASSIGN_OR_RETURN(output_buffers,
                        executable->Execute(argument_ptrs, execute_options));
    VLOG(1) << "FunctionalHloRunner: ExecuteOnDevices succeeded (repeat = "
            << repeat << ")";
    if (repeat < running_options.num_repeats - 1) {
      switch (parameter_type) {
        case ParameterType::kOneTupleOfArrays:
          argument_ptrs = CreateArgumentPointersBasedOnAliasing(
              output_buffers, device_buffers,
              get_output_index_for_one_tuple_of_arrays);
          break;
        case ParameterType::kOneListOfArrays:
          argument_ptrs = CreateArgumentPointersBasedOnAliasing(
              output_buffers, device_buffers,
              get_output_index_for_one_list_of_arrays);
          break;
        case ParameterType::kOther:
          argument_ptrs =
              CreateArgumentPointersFromDeviceBuffers(device_buffers);
          break;
      }
    }
  }
  TF_ASSIGN_OR_RETURN(PerDeviceLiteralVecType results,
                      FetchAndLogOutput(client, output_buffers,
                                        running_options.module_output_mode,
                                        running_options.log_input_output()));
  return results;
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
FunctionalHloRunner::CreateArgumentsOnDevice(
    PjRtClient& client, const PjRtLoadedExecutable* executable,
    const RunningOptions& running_options, bool flatten_arguments) {
  if (running_options.module_argument_mode ==
      ModuleArgumentMode::kUninitialized) {
    return CreateUninitializedArgumentsOnDevice(
        client, executable, running_options, flatten_arguments);
  }

  absl::Span<PjRtDevice* const> addressable_devices =
      executable->addressable_devices();
  size_t num_addressable_devices = addressable_devices.size();

  PerDeviceLiteralVecType per_device_argument_literals;
  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids =
          executable->addressable_device_logical_ids();
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->GetHloModules());
  VLOG(1) << "FunctionalHloRunner: local_executable count = "
          << hlo_modules.size();

  const bool kUseRandomInputs = running_options.module_argument_mode ==
                                    ModuleArgumentMode::kUseRandomInputs ||
                                running_options.module_argument_mode ==
                                    ModuleArgumentMode::kUseSharedRandomInputs;
  const bool kUseSharedInputs =
      running_options.module_argument_mode ==
          ModuleArgumentMode::kUseSharedRandomInputs ||
      running_options.module_argument_mode ==
          ModuleArgumentMode::kUseZerosAsInput;

  for (int i = 0; i < num_addressable_devices; ++i) {
    VLOG(3) << "Creating fake argument for device " << i;
    LiteralVec& argument_literals =
        per_device_argument_literals[addressable_devices[i]->id()];
    int executable_idx = hlo_modules.size() == 1
                             ? 0
                             : addressable_device_logical_ids[i].partition;
    HloModule* my_hlo_module = hlo_modules[executable_idx].get();
    if (flatten_arguments) {
      TF_RETURN_IF_ERROR(EnsureSingleTupleForFlattening(*my_hlo_module));
    }
    if (running_options.module_argument_mode ==
        ModuleArgumentMode::kUseDeviceIdAsInput) {
      const auto params =
          my_hlo_module->entry_computation()->parameter_instructions();
      if (flatten_arguments) {
        CHECK_EQ(params.size(), 1);
        CHECK(params.front()->shape().IsTuple());
        argument_literals.reserve(params.front()->shape().tuple_shapes_size());
      } else {
        argument_literals.reserve(params.size());
      }
      for (int j = 0; j < params.size(); ++j) {
        TF_ASSIGN_OR_RETURN(
            Literal argument_literal_j,
            MakeFakeLiteralWithSameValue(params[j]->shape(),
                                         addressable_devices[i]->id()));
        if (flatten_arguments) {
          std::vector<Literal> decomposed_argument_literals =
              argument_literal_j.DecomposeTuple();
          for (auto& literal : decomposed_argument_literals) {
            argument_literals.push_back(std::move(literal));
          }
        } else {
          argument_literals.push_back(std::move(argument_literal_j));
        }
      }
    } else {
      if (flatten_arguments) {
        TF_ASSIGN_OR_RETURN(LiteralVec tupled_argument_literals,
                            MakeFakeArguments(my_hlo_module, kUseRandomInputs));
        CHECK_EQ(tupled_argument_literals.size(), 1);
        CHECK(tupled_argument_literals.front().shape().IsTuple());
        argument_literals = tupled_argument_literals.front().DecomposeTuple();
      } else {
        TF_ASSIGN_OR_RETURN(argument_literals,
                            MakeFakeArguments(my_hlo_module, kUseRandomInputs));
      }
      if (kUseSharedInputs) {
        break;
      }
    }
  }

  if (kUseSharedInputs) {
    PerDeviceIndexVecType per_device_index_vec;
    std::vector<int> argument_indices;
    argument_indices.resize(
        per_device_argument_literals[addressable_devices[0]->id()].size());
    absl::c_iota(argument_indices, 0);
    for (int i = 0; i < num_addressable_devices; ++i) {
      per_device_index_vec[addressable_devices[i]->id()] = argument_indices;
    }
    return CopyArgumentsToDevice(
        client, addressable_devices,
        per_device_argument_literals[addressable_devices[0]->id()],
        per_device_index_vec, running_options.log_input_output());
  }
  return CopyArgumentsToDevice(client, addressable_devices,
                               per_device_argument_literals,
                               running_options.log_input_output());
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
FunctionalHloRunner::CreateUninitializedArgumentsOnDevice(
    PjRtClient& client, const PjRtLoadedExecutable* executable,
    const RunningOptions& running_options, bool flatten_arguments) {
  absl::Span<PjRtDevice* const> addressable_devices =
      executable->addressable_devices();
  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids =
          executable->addressable_device_logical_ids();
  TF_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                      executable->GetHloModules());
  VLOG(1) << "FunctionalHloRunner: local_executable count = "
          << hlo_modules.size();

  LOG(INFO) << "Starting argument buffer shape calculation.";
  PerDeviceShapeVecType argument_shapes_per_device;
  // This must be true, based on the comment on
  // PjRtLoadedExecutable::addressable_devices().
  CHECK_EQ(addressable_devices.size(), addressable_device_logical_ids.size());
  for (int i = 0; i < static_cast<int>(addressable_devices.size()); ++i) {
    VLOG(3) << "Calculating fake argument shapes for device " << i;
    PjRtDevice* device = addressable_devices[i];
    int executable_idx = hlo_modules.size() == 1
                             ? 0
                             : addressable_device_logical_ids[i].partition;
    const HloModule& hlo_module = *hlo_modules[executable_idx];

    std::vector<Shape> argument_shapes;
    if (flatten_arguments) {
      TF_RETURN_IF_ERROR(EnsureSingleTupleForFlattening(hlo_module));

      std::vector<Shape> original_argument_shapes =
          GetArgumentShapes(hlo_module);
      CHECK_EQ(original_argument_shapes.size(), 1);
      CHECK(original_argument_shapes.front().IsTuple());
      argument_shapes = original_argument_shapes.front().tuple_shapes();
    } else {
      argument_shapes = GetArgumentShapes(hlo_module);
    }

    argument_shapes_per_device[device->id()] = std::move(argument_shapes);
  }

  LOG(INFO) << "Starting argument buffer allocation.";
  int buffer_count = 0;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
      argument_buffers_per_device;
  argument_buffers_per_device.reserve(addressable_devices.size());
  for (int i = 0; i < static_cast<int>(addressable_devices.size()); ++i) {
    VLOG(3) << "Allocating fake arguments for device " << i;
    PjRtDevice* device = addressable_devices[i];

    CHECK(argument_shapes_per_device.contains(device->id()));
    const std::vector<Shape>& argument_shapes =
        argument_shapes_per_device.at(device->id());
    std::vector<std::unique_ptr<PjRtBuffer>> argument_buffers;
    argument_buffers.reserve(argument_shapes.size());

    for (const Shape& shape : argument_shapes) {
      if (running_options.log_input_output()) {
        LOG(INFO) << "device_id=" << device->id()
                  << ", input = " << shape.ToString();
      }

      TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> argument_buffer,
                          client.CreateUninitializedBuffer(shape, device));
      argument_buffers.push_back(std::move(argument_buffer));
      buffer_count += 1;
    }

    argument_buffers_per_device.push_back(std::move(argument_buffers));
  }
  LOG(INFO) << "Allocated argument buffers: " << buffer_count;

  for (const auto& argument_buffers : argument_buffers_per_device) {
    for (const auto& buffer : argument_buffers) {
      TF_RETURN_IF_ERROR(buffer->BlockHostUntilReady());
    }
  }
  LOG(INFO) << "Argument buffers are ready.";

  return argument_buffers_per_device;
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
FunctionalHloRunner::CopyArgumentsToDevice(
    PjRtClient& client, absl::Span<PjRtDevice* const> addressable_devices,
    const PerDeviceLiteralVecType& arguments, bool log_input) {
  size_t num_addressable_devices = addressable_devices.size();
  if (num_addressable_devices != arguments.size()) {
    return InvalidArgument(
        "The number of provided arguments does not match "
        "the number of logical devices.");
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffers;
  argument_buffers.resize(num_addressable_devices);

  for (int i = 0; i < num_addressable_devices; ++i) {
    PjRtDevice* curr_device = addressable_devices[i];
    int curr_device_id = curr_device->id();
    if (!arguments.contains(curr_device_id)) {
      return InvalidArgument(
          "The provided argument map does not contain arguments "
          "for device: %d",
          curr_device_id);
    }

    const std::vector<Literal>& curr_device_arguments =
        arguments.at(curr_device_id);

    argument_buffers[i].reserve(curr_device_arguments.size());
    for (const Literal& literal : curr_device_arguments) {
      if (log_input) {
        LOG(INFO) << "device_id=" << curr_device_id
                  << ", input = " << literal.ToString();
      }
      TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> argument_buffer,
                          client.BufferFromHostLiteral(literal, curr_device));
      argument_buffers[i].push_back(std::move(argument_buffer));
    }
  }
  for (const auto& device_argument_buffers : argument_buffers) {
    for (const auto& device_buffer : device_argument_buffers) {
      TF_RETURN_IF_ERROR(device_buffer->BlockHostUntilReady());
    }
  }
  return argument_buffers;
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
FunctionalHloRunner::CopyArgumentsToDevice(
    PjRtClient& client, absl::Span<PjRtDevice* const> addressable_devices,
    const LiteralVec& argument_literals,
    const PerDeviceIndexVecType& argument_indices, bool log_input) {
  size_t num_addressable_devices = addressable_devices.size();
  if (num_addressable_devices != argument_indices.size()) {
    return InvalidArgument(
        "The number of provided arguments does not match "
        "the number of logical devices.");
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffers;
  argument_buffers.resize(num_addressable_devices);

  for (int i = 0; i < num_addressable_devices; ++i) {
    PjRtDevice* curr_device = addressable_devices[i];
    int curr_device_id = curr_device->id();
    if (!argument_indices.contains(curr_device_id)) {
      return InvalidArgument(
          "The provided argument map does not contain arguments "
          "for device: %d",
          curr_device_id);
    }

    const std::vector<int> curr_device_arguments_indices =
        argument_indices.at(curr_device_id);

    argument_buffers[i].reserve(curr_device_arguments_indices.size());
    for (int index : curr_device_arguments_indices) {
      const Literal& literal = argument_literals[index];
      if (log_input) {
        LOG(INFO) << "device_id=" << curr_device_id
                  << ", input = " << literal.ToString();
      }
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> argument_buffer,
          client.BufferFromHostLiteral(literal, addressable_devices[i]));
      argument_buffers[i].push_back(std::move(argument_buffer));
    }
  }
  for (const auto& device_argument_buffers : argument_buffers) {
    for (const auto& device_buffer : device_argument_buffers) {
      TF_RETURN_IF_ERROR(device_buffer->BlockHostUntilReady());
    }
  }
  return argument_buffers;
}

StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType>
FunctionalHloRunner::FetchAndLogOutput(
    PjRtClient& client,
    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& output_buffers,
    ModuleOutputMode module_output_mode, bool log_output) {
  CHECK(!output_buffers.empty());
  absl::Mutex mu;
  Status status;
  size_t num_pending_transfers = 0;
  bool device_0_is_local = false;
  for (PjRtDevice* device : GetLocalDevices(client)) {
    if (device->id() == 0) {
      device_0_is_local = true;
    }
  }

  if (module_output_mode == ModuleOutputMode::kReturnDevice0Outputs &&
      device_0_is_local) {
    num_pending_transfers = output_buffers[0].size();
  } else if (module_output_mode == ModuleOutputMode::kReturnOutputs) {
    for (const auto& bs : output_buffers) {
      num_pending_transfers += bs.size();
    }
  }

  PerDeviceLiteralVecType outputs;
  for (int i = 0; i < output_buffers.size(); ++i) {
    if (output_buffers[i].empty()) {
      continue;
    }
    const int device_id = output_buffers[i][0]->device()->id();
    std::vector<Literal>& output_slice = outputs[device_id];
    if (module_output_mode == ModuleOutputMode::kReturnOutputs ||
        (module_output_mode == ModuleOutputMode::kReturnDevice0Outputs &&
         device_id == 0)) {
      output_slice.reserve(output_buffers[i].size());
      for (const auto& buffer : output_buffers[i]) {
        TF_RET_CHECK(buffer->device() == output_buffers[i][0]->device())
            << "All outputs from a given vector of outputs should be for the "
               "same device";
        output_slice.emplace_back(
            ShapeUtil::DeviceShapeToHostShape(buffer->on_device_shape()));
        buffer->ToLiteral(&output_slice.back(), [&](Status s) {
          absl::MutexLock lock(&mu);
          --num_pending_transfers;
          status.Update(s);
        });
      }
    } else {
      for (const auto& buffer : output_buffers[i]) {
        TF_RET_CHECK(buffer->device() == output_buffers[i][0]->device())
            << "All outputs from a given vector of outputs should be for the "
               "same device";
        TF_RETURN_IF_ERROR(buffer->BlockHostUntilReady());
      }
    }
  }
  if (module_output_mode == ModuleOutputMode::kReturnOutputs ||
      (module_output_mode == ModuleOutputMode::kReturnDevice0Outputs &&
       device_0_is_local)) {
    auto cond = [&]() { return !status.ok() || num_pending_transfers == 0; };
    absl::MutexLock lock(&mu);
    mu.Await(absl::Condition(&cond));
    TF_RETURN_IF_ERROR(status);
    if (log_output) {
      for (const PjRtDevice* device : GetLocalDevices(client)) {
        int device_id = device->id();
        if (module_output_mode == ModuleOutputMode::kReturnDevice0Outputs &&
            device_id != 0) {
          continue;
        }
        LOG(INFO) << "Outputs for device_id: " << device_id;
        const std::vector<Literal>& output_slice = outputs[device_id];
        for (int i = 0; i < output_slice.size(); ++i) {
          LOG(INFO) << "output[" << i << "]: " << output_slice[i].ToString();
        }
      }
    }
  }
  return outputs;
}

}  // namespace xla
