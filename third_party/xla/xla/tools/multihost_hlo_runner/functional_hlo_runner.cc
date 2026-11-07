/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/client/executable_build_options.h"
#include "xla/debug_options_flags.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/hlo/translate/hlo_to_mhlo/translate.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/large_hlo_snapshot_serialization/serialization.h"
#include "xla/service/computation_layout.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/slow_operation_alarm.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_control_flow_flattening.h"
#include "xla/tools/multihost_hlo_runner/hlo_input_output_format.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/file_system_helper.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/fixed_option_set_flag.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace FunctionalHloRunner {
namespace {
using HloModuleAndArguments = ::xla::FunctionalHloRunner::HloModuleAndArguments;

absl::Span<PjRtDevice* const> GetLocalDevices(const PjRtClient& client) {
  return client.addressable_devices();
}

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
// This enables PjRtClient::Execute to assemble the tupled arguments from
// a flat list of buffers.
//
// Case 2: the HLO module is compiled with
// CompileOptions::parameter_is_tupled_arguments = false
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

template <typename ElementType>
void PopulateWithSameValue(Literal* literal, ElementType val) {
  for (ElementType& element : literal->data<ElementType>()) {
    element = static_cast<ElementType>(val);
  }
}

absl::StatusOr<Literal> MakeFakeLiteralWithSameValue(const Shape& shape,
                                                     int value) {
  if (shape.IsArray()) {
    Shape new_shape = shape;
    new_shape.mutable_layout()->clear_tiles();
    return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
        [&](auto type) -> absl::StatusOr<Literal> {
          if constexpr (primitive_util::IsArrayType(type)) {
            using NativeT = primitive_util::NativeTypeOf<type>;

            Literal literal(new_shape);
            PopulateWithSameValue(
                &literal,
                static_cast<NativeT>(type == PRED ? (value % 2) == 0 : value));
            for (int i = 0; i < shape.dimensions().size(); i++) {
              if (shape.is_dynamic_dimension(i)) {
                // TODO(b/378917570): We might need to set the dynamic size to
                // the actual bound i.e., shape.dimensions(i) when HybridSim
                // supports SparseCore.
                literal.SetDynamicSize(i, 0);
              }
            }
            return literal;
          }
          return Unimplemented(
              "Unsupported type for fake literal generation: %s",
              ShapeUtil::HumanString(shape));
        },
        new_shape.element_type());
  } else if (shape.IsTuple()) {
    std::vector<Literal> subliterals;
    for (const Shape& subshape : shape.tuple_shapes()) {
      ABSL_ASSIGN_OR_RETURN(Literal subliteral,
                       MakeFakeLiteralWithSameValue(subshape, value));
      subliterals.push_back(std::move(subliteral));
    }
    return LiteralUtil::MakeTupleOwned(std::move(subliterals));
  }
  return InvalidArgument("Unsupported type for fake literal generation: %s",
                         ShapeUtil::HumanString(shape));
}

absl::StatusOr<HloModuleAndArguments> ReadModuleFromSnapshotBinaryProtoFile(
    absl::string_view hlo_file) {
  HloSnapshot proto;
  HloModuleAndArguments hlo_module_and_arguments;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  hlo_module_and_arguments.arguments.emplace_back();
  hlo_module_and_arguments.arguments.front().resize(proto.arguments_size());
  for (int i = 0; i < proto.arguments_size(); i++) {
    ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.arguments.front()[i],
                     Literal::CreateFromProto(proto.arguments()[i]));
  }
  ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                   CreateModuleFromProto(proto.hlo().hlo_module()));
  return hlo_module_and_arguments;
}

namespace {

// This function reads an HloUnoptimizedSnapshot from the file at the given
// path.
// It first tries to deserialize the snapshot using custom serialization.
// If that fails, it falls back to standard deserialization.
absl::StatusOr<HloUnoptimizedSnapshot> ReadHloUnoptimizedSnapshot(
    absl::string_view hlo_file) {
  tsl::Env* env = tsl::Env::Default();
  std::unique_ptr<tsl::RandomAccessFile> file;
  ABSL_RETURN_IF_ERROR(env->NewRandomAccessFile(std::string(hlo_file), &file));

  tsl::RandomAccessFileCopyingInputStream custom_deserialization_input_stream(
      file.get());
  tsl::protobuf::io::CopyingInputStreamAdaptor custom_deserialization_adaptor(
      &custom_deserialization_input_stream);

  // Try to deserialize snapshot with custom deserialization.
  auto proto_or_status =
      DeserializeHloUnoptimizedSnapshot(&custom_deserialization_adaptor);

  if (proto_or_status.ok()) {
    return proto_or_status.value();
  }

  // Fallback to standard deserialization.

  HloUnoptimizedSnapshot proto;
  // Fallback to standard deserialization. This requires creating a new input
  // stream because we need to read from the beginning of the file.
  tsl::RandomAccessFileCopyingInputStream fallback_input_stream(file.get());
  tsl::protobuf::io::CopyingInputStreamAdaptor fallback_adaptor(
      &fallback_input_stream);
  tsl::protobuf::io::CodedInputStream coded_stream(&fallback_adaptor);

  if (!proto.ParseFromCodedStream(&coded_stream)) {
    return Internal(
        "Failed to parse HloUnoptimizedSnapshot from the input stream with "
        "standard deserialization. Custom deserialization also failed with "
        "error: "
        "%s",
        proto_or_status.status().message());
  }
  return proto;
}

}  // namespace

absl::StatusOr<HloModuleAndArguments>
ReadModuleFromUnoptimizedSnapshotBinaryProtoFile(absl::string_view hlo_file) {
  HloModuleAndArguments hlo_module_and_arguments;

  ABSL_ASSIGN_OR_RETURN(HloUnoptimizedSnapshot proto,
                   ReadHloUnoptimizedSnapshot(hlo_file));

  ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                   CreateModuleFromProto(proto.hlo_module()));

  for (const auto& arguments : proto.partitions()) {
    hlo_module_and_arguments.arguments.emplace_back();
    hlo_module_and_arguments.arguments.back().reserve(
        arguments.arguments_size());
    for (const auto& argument : arguments.arguments()) {
      ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.arguments.back().emplace_back(),
                       Literal::CreateFromProto(argument));
    }
  }
  return hlo_module_and_arguments;
}

absl::StatusOr<HloModuleAndArguments>
ReadModuleFromUnoptimizedSnapshotTextProtoFile(absl::string_view hlo_file) {
  HloUnoptimizedSnapshot proto;
  HloModuleAndArguments hlo_module_and_arguments;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadTextProto(tsl::Env::Default(), std::string(hlo_file), &proto));
  ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                   CreateModuleFromProto(proto.hlo_module()));

  for (const auto& arguments : proto.partitions()) {
    hlo_module_and_arguments.arguments.emplace_back();
    hlo_module_and_arguments.arguments.back().reserve(
        arguments.arguments_size());
    for (const auto& argument : arguments.arguments()) {
      ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.arguments.back().emplace_back(),
                       Literal::CreateFromProto(argument));
    }
  }
  return hlo_module_and_arguments;
}

absl::StatusOr<std::unique_ptr<HloModule>> ReadStableHloModule(
    absl::string_view stablehlo_file) {
  std::string contents;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), stablehlo_file, &contents));

  mlir::MLIRContext ctx;
  ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                   ParseMlirModuleString(contents, ctx));
  return ConvertStablehloToHlo(*std::move(module));
}

ReplicasAndPartitions GetReplicasAndPartitionsInternal(
    const std::optional<ExecutionOptions>& execution_options, int device_count,
    const std::optional<int>& num_replicas,
    const std::optional<int>& num_partitions, int num_slices = 1) {
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

// Calculates the requested number of replicas and partitions.
// The explicit num_replicas and num_partitions options override
// execution_options.
// Regarding the num_slices parameter, see the comment on
// xla::MultiSliceConfig.
ReplicasAndPartitions GetReplicasAndPartitions(
    const std::optional<ExecutionOptions>& execution_options, int device_count,
    const std::optional<int>& num_replicas,
    const std::optional<int>& num_partitions, int num_slices = 1) {
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

absl::StatusOr<PerDeviceLiteralVecType> FetchAndLogOutput(
    PjRtClient& client,
    const std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>& output_buffers,
    ModuleOutputMode module_output_mode, bool log_output) {
  CHECK(!output_buffers.empty());
  bool device_0_is_local = false;
  for (PjRtDevice* device : GetLocalDevices(client)) {
    if (device->global_device_id().value() == 0) {
      device_0_is_local = true;
    }
  }

  PerDeviceLiteralVecType outputs;
  absl::Mutex mu;
  absl::Status status;
  size_t num_pending_transfers = 0;

  absl::Status issue_status = [&]() -> absl::Status {
    absl::Status ready_status;
    for (int i = 0; i < output_buffers.size(); ++i) {
      if (output_buffers[i].empty()) {
        continue;
      }
      const int device_id =
          output_buffers[i][0]->device()->global_device_id().value();
      std::vector<Literal>& output_slice = outputs[device_id];
      if (module_output_mode == ModuleOutputMode::kReturnOutputs ||
          (module_output_mode == ModuleOutputMode::kReturnDevice0Outputs &&
           device_id == 0)) {
        output_slice.reserve(output_buffers[i].size());
        for (const auto& buffer : output_buffers[i]) {
          if (buffer->device() != output_buffers[i][0]->device()) {
            return absl::InternalError(
                "All outputs from a given vector of outputs should be for the "
                "same device");
          }
          ABSL_ASSIGN_OR_RETURN(auto logical_shape,
                           buffer->logical_on_device_shape());
          output_slice.emplace_back(
              ShapeUtil::DeviceShapeToHostShape(logical_shape));
          {
            absl::MutexLock lock(mu);
            ++num_pending_transfers;
          }
          buffer->ToLiteral(&output_slice.back()).OnReady([&](absl::Status s) {
            absl::MutexLock lock(mu);
            --num_pending_transfers;
            status.Update(s);
          });
        }
      } else {
        for (const auto& buffer : output_buffers[i]) {
          if (buffer->device() != output_buffers[i][0]->device()) {
            return absl::InternalError(
                "All outputs from a given vector of outputs should be for the "
                "same device");
          }
          ready_status.Update(buffer->GetReadyFuture().Await());
        }
      }
    }
    return ready_status;
  }();

  // The ToLiteral callbacks above reference `outputs`, `mu`, `status` and
  // `num_pending_transfers`, so every issued transfer must complete before this
  // function returns, including when a transfer fails or issuing one fails.
  {
    auto all_transfers_done = [&]() { return num_pending_transfers == 0; };
    absl::MutexLock lock(mu);
    mu.Await(absl::Condition(&all_transfers_done));
    ABSL_RETURN_IF_ERROR(issue_status);
    ABSL_RETURN_IF_ERROR(status);
  }

  if (log_output &&
      (module_output_mode == ModuleOutputMode::kReturnOutputs ||
       (module_output_mode == ModuleOutputMode::kReturnDevice0Outputs &&
        device_0_is_local))) {
    for (const PjRtDevice* device : GetLocalDevices(client)) {
      int device_id = device->global_device_id().value();
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
  return outputs;
}

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
  const auto& params = module.entry_computation()->parameter_instructions();
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

absl::Status EnsureSingleTupleForFlattening(const HloModule& module) {
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
  return absl::OkStatus();
}

absl::StatusOr<PerDeviceLiteralVecType> RunInternal(
    PjRtClient& client, PjRtLoadedExecutable* executable,
    std::function<absl::StatusOr<
        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>(bool)>
        create_argument_buffers_on_device,
    const RunningOptions& running_options) {
  ExecuteOptions execute_options;
  if (running_options.multi_slice_config != nullptr) {
    execute_options.multi_slice_config = running_options.multi_slice_config;
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                   executable->GetHloModules());
  CHECK_EQ(hlo_modules.size(), 1);
  const HloModule& module = *(hlo_modules.front());
  ParameterType parameter_type = GetParameterType(module);
  bool flatten_arguments = parameter_type == ParameterType::kOneTupleOfArrays;
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
  std::optional<std::vector<Future<>>> futures;
  futures.emplace();
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> device_buffers;
  std::vector<std::vector<PjRtBuffer*>> argument_ptrs;
  size_t argument_preparation_step = 0;

  auto prepare_arguments_on_device = [&]() -> absl::Status {
    const size_t current_step = argument_preparation_step++;
    {
      XLA_SCOPED_LOGGING_TIMER("FunctionalHloRunner::CreateArgumentsOnDevice");
      device_buffers.clear();
      argument_ptrs.clear();
      ABSL_ASSIGN_OR_RETURN(device_buffers,
                       create_argument_buffers_on_device(flatten_arguments));
      absl::Status ready_status;
      for (const std::vector<std::unique_ptr<PjRtBuffer>>& per_device_buffers :
           device_buffers) {
        for (const std::unique_ptr<PjRtBuffer>& buffer : per_device_buffers) {
          if (buffer != nullptr) {
            ready_status.Update(buffer->GetReadyFuture().Await());
          }
        }
      }
      ABSL_RETURN_IF_ERROR(ready_status);
      argument_ptrs = CreateArgumentPointersFromDeviceBuffers(device_buffers);
    }
    if (running_options.after_arguments_ready_callback != nullptr) {
      XLA_SCOPED_LOGGING_TIMER(
          "FunctionalHloRunner::AfterArgumentsReadyCallback");
      ABSL_RETURN_IF_ERROR(
          running_options.after_arguments_ready_callback(current_step));
    }
    return absl::OkStatus();
  };

  auto refresh_argument_ptrs = [&]() -> absl::Status {
    if (running_options.recreate_buffers_between_repeats) {
      return prepare_arguments_on_device();
    }
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
        argument_ptrs = CreateArgumentPointersFromDeviceBuffers(device_buffers);
        break;
    }
    return absl::OkStatus();
  };

  if (running_options.total_run_time != nullptr) {
    *running_options.total_run_time = absl::ZeroDuration();
  }
  bool has_active_profiler_session = false;
  for (int repeat = 0; repeat < running_options.num_repeats; ++repeat) {
    const bool is_last_repeat = (repeat == running_options.num_repeats - 1);
    const bool profile_current_repeat =
        (running_options.profiler != nullptr) &&
        (repeat >= running_options.num_repeats -
                       running_options.num_repeats_with_profiler) &&
        (!running_options.multipass_profiling || is_last_repeat);

    if (repeat == 0 || running_options.recreate_buffers_between_repeats) {
      VLOG(1) << "Creating argument buffers. repeat = " << repeat;
      ABSL_RETURN_IF_ERROR(prepare_arguments_on_device());
    }

    VLOG(1) << "FunctionalHloRunner: ExecuteOnDevices started (repeat = "
            << repeat << ").";
    {
      XLA_SCOPED_LOGGING_TIMER("FunctionalHloRunner::ExecuteOnDevices");
      execute_options.launch_id = repeat + 1 + running_options.base_run_id;
      if (running_options.execution_profiles != nullptr) {
        execute_options.execution_profile =
            &running_options.execution_profiles->emplace_back();
        execute_options.execution_profile->set_warmup_run_executed(repeat > 0);
      }

      if (profile_current_repeat && !has_active_profiler_session) {
        running_options.profiler->CreateSession();
        has_active_profiler_session = true;
      }
      auto execute_once = [&]() -> absl::Status {
        futures->clear();
        const absl::Time execute_start = absl::Now();
        ABSL_ASSIGN_OR_RETURN(
            std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
                new_output_buffers,
            executable->Execute(argument_ptrs, execute_options, futures));
        absl::Status execute_status;
        for (Future<>& future : *futures) {
          execute_status.Update(future.Await());
        }
        if (running_options.total_run_time != nullptr) {
          *running_options.total_run_time += absl::Now() - execute_start;
        }
        ABSL_RETURN_IF_ERROR(execute_status);
        output_buffers = std::move(new_output_buffers);
        return absl::OkStatus();
      };

      if (running_options.multipass_profiling && profile_current_repeat) {
        bool executed_in_multipass = false;
        while (true) {
          ABSL_ASSIGN_OR_RETURN(bool need_more_pass,
                           running_options.profiler->NeedMorePasses());
          if (!need_more_pass) {
            break;
          }
          if (executed_in_multipass) {
            ABSL_RETURN_IF_ERROR(refresh_argument_ptrs());
          }
          running_options.profiler->StartPass();
          running_options.profiler->PushRange("hlo_runner_pass");
          absl::Cleanup end_pass = [&] {
            running_options.profiler->PopRange();
            running_options.profiler->StopPass();
          };
          ABSL_RETURN_IF_ERROR(execute_once());
          executed_in_multipass = true;
        }
        if (!executed_in_multipass) {
          ABSL_RETURN_IF_ERROR(execute_once());
        }
      } else {
        ABSL_RETURN_IF_ERROR(execute_once());
      }
      // Close any active profiler session before recreating argument buffers on
      // the next repeat so input generation and H2D/ICI buffer loading are
      // never recorded inside an active profile.
      const bool upload_active_profiler_session =
          running_options.recreate_profiler_session_between_repeats ||
          running_options.recreate_buffers_between_repeats || is_last_repeat;
      if (has_active_profiler_session && upload_active_profiler_session) {
        XLA_SCOPED_LOGGING_TIMER("FunctionalHloRunner::XProfUpload");
        running_options.profiler->UploadSession();
        has_active_profiler_session = false;
      }
    }
    VLOG(1) << "FunctionalHloRunner: ExecuteOnDevices succeeded (repeat = "
            << repeat << ")";
    if (!is_last_repeat && !running_options.recreate_buffers_between_repeats) {
      ABSL_RETURN_IF_ERROR(refresh_argument_ptrs());
    }
  }

  ABSL_ASSIGN_OR_RETURN(PerDeviceLiteralVecType results,
                   FetchAndLogOutput(client, output_buffers,
                                     running_options.module_output_mode,
                                     running_options.log_input_output()));
  return results;
}

// Returns true if `src_buffer` (created from `src_literal` at parameter index
// `src_arg_i`) has identical static array shape, minor-to-major host layout,
// executable parameter layout, memory space kind, and raw host payload bytes as
// `dst_literal`, making it safe to clone via `PjRtBuffer::CopyToMemorySpace`.
bool CanCloneArgumentBuffer(
    const Literal& src_literal, int src_arg_i, PjRtBuffer* src_buffer,
    const Literal& dst_literal, const Layout& dst_layout,
    PjRtMemorySpace* dst_memory_space,
    absl::Span<const Layout> executable_parameter_layouts) {
  const int64_t dst_byte_size = dst_literal.size_bytes();
  if (src_buffer == nullptr ||
      src_arg_i >= executable_parameter_layouts.size() ||
      !src_literal.shape().IsArrayExcludingBuffer() ||
      !src_literal.shape().is_static() ||
      src_literal.size_bytes() != dst_byte_size) {
    return false;
  }
  PjRtMemorySpace* src_memory_space = src_buffer->memory_space();
  if (src_memory_space == nullptr || dst_memory_space == nullptr ||
      src_memory_space->kind_id() != dst_memory_space->kind_id()) {
    return false;
  }
  if (executable_parameter_layouts[src_arg_i] != dst_layout ||
      !Shape::Equal().MinorToMajorOnlyInLayout()(src_literal.shape(),
                                                 dst_literal.shape())) {
    return false;
  }
  const char* src_ptr = static_cast<const char*>(src_literal.untyped_data());
  const char* dst_ptr = static_cast<const char*>(dst_literal.untyped_data());
  return src_ptr == dst_ptr ||
         std::memcmp(src_ptr, dst_ptr, static_cast<size_t>(dst_byte_size)) == 0;
}

// Attempts to create the device buffer for `(device_idx, arg_i)` by cloning an
// already-transferred buffer via `PjRtBuffer::CopyToMemorySpace`, avoiding
// redundant host-to-device DMA transfers and host-side layout relinearization.
//
// Search order:
//   1. Earlier parameters (`0 .. arg_i - 1`) on peer devices (`peer_idx !=
//      device_idx`) residing in a different `PjRtMemorySpace`. Rotating the
//      starting peer offset by `arg_i` forms a 1-to-1 ring across devices for
//      repeated layer weights while reading from buffers created in earlier
//      parameter iterations rather than chaining off in-flight copies at
//      `arg_i`.
//   2. Parameter `arg_i` on peer devices (`peer_idx != device_idx`) residing in
//      a different `PjRtMemorySpace`, covering the primary-to-replica fan-out
//      when a distinct tensor variant is first loaded at `arg_i`.
//   3. When `allow_same_memory_space_clone` is true (disabled on TPU, where
//      intra-chip `CopyToMemorySpace` bounces through host DRAM via D2H + H2D),
//      earlier parameters (`0 .. arg_i - 1`) on `device_idx` residing in the
//      same `dst_memory_space`.
//
// Returns `nullptr` if no matching source buffer exists or if the backend does
// not implement `CopyToMemorySpace` for the target memory space.
std::unique_ptr<PjRtBuffer> TryCloneArgumentBuffer(
    size_t device_idx, size_t arg_i, const Literal& dst_literal,
    const Layout& dst_layout, PjRtMemorySpace* dst_memory_space,
    absl::Span<const std::vector<Literal>* const> per_device_arguments,
    absl::Span<const std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffers,
    absl::Span<const Layout> executable_parameter_layouts,
    bool allow_same_memory_space_clone) {
  const size_t num_devices = argument_buffers.size();
  if (num_devices > 1) {
    const size_t peer_count = num_devices - 1;
    const size_t base_offset = (arg_i % peer_count) + 1;
    for (size_t prev_i = 0; prev_i < arg_i; ++prev_i) {
      for (size_t step = 0; step < peer_count; ++step) {
        const size_t offset = ((base_offset - 1 + step) % peer_count) + 1;
        const size_t peer_idx = (device_idx + offset) % num_devices;
        PjRtBuffer* src_buf = argument_buffers[peer_idx][prev_i].get();
        if (src_buf == nullptr || src_buf->memory_space() == dst_memory_space ||
            !CanCloneArgumentBuffer((*per_device_arguments[peer_idx])[prev_i],
                                    prev_i, src_buf, dst_literal, dst_layout,
                                    dst_memory_space,
                                    executable_parameter_layouts)) {
          continue;
        }
        absl::StatusOr<std::unique_ptr<PjRtBuffer>> cloned =
            src_buf->CopyToMemorySpace(dst_memory_space);
        if (!cloned.ok()) {
          return nullptr;
        }
        return *std::move(cloned);
      }
    }
    for (size_t offset = 1; offset < num_devices; ++offset) {
      const size_t peer_idx = (device_idx + offset) % num_devices;
      PjRtBuffer* src_buf = argument_buffers[peer_idx][arg_i].get();
      if (src_buf == nullptr || src_buf->memory_space() == dst_memory_space ||
          !CanCloneArgumentBuffer((*per_device_arguments[peer_idx])[arg_i],
                                  arg_i, src_buf, dst_literal, dst_layout,
                                  dst_memory_space,
                                  executable_parameter_layouts)) {
        continue;
      }
      absl::StatusOr<std::unique_ptr<PjRtBuffer>> cloned =
          src_buf->CopyToMemorySpace(dst_memory_space);
      if (!cloned.ok()) {
        return nullptr;
      }
      return *std::move(cloned);
    }
  }

  if (allow_same_memory_space_clone) {
    for (size_t prev_i = 0; prev_i < arg_i; ++prev_i) {
      PjRtBuffer* src_buf = argument_buffers[device_idx][prev_i].get();
      if (src_buf == nullptr || src_buf->memory_space() != dst_memory_space ||
          !CanCloneArgumentBuffer((*per_device_arguments[device_idx])[prev_i],
                                  prev_i, src_buf, dst_literal, dst_layout,
                                  dst_memory_space,
                                  executable_parameter_layouts)) {
        continue;
      }
      absl::StatusOr<std::unique_ptr<PjRtBuffer>> cloned =
          src_buf->CopyToMemorySpace(dst_memory_space);
      if (!cloned.ok()) {
        return nullptr;
      }
      return *std::move(cloned);
    }
  }
  return nullptr;
}

// Creates argument buffers based on the given arguments map. Note that the
// arguments might be invalid when arguments are destructed.
absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
CopyArgumentsToDevice(PjRtClient& client,
                      const PjRtLoadedExecutable* executable,
                      const PerDeviceLiteralVecType& arguments,
                      const RunningOptions& running_options,
                      bool flattened_arguments,
                      bool clone_device0_arguments = false) {
  const bool log_input = running_options.log_input_output();
  absl::Span<PjRtDevice* const> addressable_devices =
      executable->addressable_devices();
  size_t num_addressable_devices = addressable_devices.size();
  if (!clone_device0_arguments && num_addressable_devices != arguments.size()) {
    return InvalidArgument(
        "The number of provided arguments (%v) does not match the number of "
        "logical devices (%v).",
        arguments.size(), num_addressable_devices);
  }
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> argument_buffers;
  argument_buffers.resize(num_addressable_devices);

  auto argument_memory_space =
      [&flattened_arguments](const HloModule* module, PjRtDevice* device,
                             int arg_i) -> absl::StatusOr<PjRtMemorySpace*> {
    auto non_tuple_memory_space = [&device](const Shape& shape) {
      if (shape.has_layout() &&
          shape.layout().memory_space() == Layout::kHostMemorySpace) {
        return device->memory_space_by_kind(PinnedHostMemorySpace::kKind);
      }
      return device->default_memory_space();
    };

    const ComputationLayout& entry_layout = module->entry_computation_layout();
    TF_RET_CHECK(entry_layout.parameter_count() > 0);
    if (entry_layout.parameter_shape(0).IsTuple() && flattened_arguments) {
      TF_RET_CHECK(entry_layout.parameter_count() == 1)
          << "entry_layout.parameter_count(): "
          << entry_layout.parameter_count();
      TF_RET_CHECK(arg_i <
                   entry_layout.parameter_shape(0).tuple_shapes().size());
      const Shape& shape = entry_layout.parameter_shape(0).tuple_shapes(arg_i);
      TF_RET_CHECK(!shape.IsTuple()) << "Nested tuples are not supported";
      return non_tuple_memory_space(shape);
    }
    TF_RET_CHECK(arg_i < entry_layout.parameter_count());
    const Shape& shape = entry_layout.parameter_shape(arg_i);
    TF_RET_CHECK(!shape.IsTuple()) << "Param tuple without flattened_arguments";
    return non_tuple_memory_space(shape);
  };
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::shared_ptr<const PjRtLayout>>&
                       executable_parameter_pjrt_layouts,
                   executable->GetParameterLayouts());
  std::vector<Layout> executable_parameter_layouts;
  executable_parameter_layouts.reserve(
      executable_parameter_pjrt_layouts.size());
  for (const std::shared_ptr<const PjRtLayout>& pjrt_layout :
       executable_parameter_pjrt_layouts) {
    executable_parameter_layouts.push_back(pjrt_layout->xla_layout());
  }
  auto buffer_from_host_literal =
      [&client, &argument_memory_space, &executable_parameter_layouts](
          const HloModule* module, PjRtDevice* device, int arg_i,
          const Literal& literal)
      -> absl::StatusOr<std::unique_ptr<PjRtBuffer>> {
    // Use the layout as specified in the executable rather than the layout of
    // the host-side literal, as the former is the authoritative layout the
    // executable expects.
    const Layout* layout = &executable_parameter_layouts[arg_i];
    ABSL_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                     argument_memory_space(module, device, arg_i));
    auto device_buffers =
        client.BufferFromHostLiteral(literal, memory_space, layout);
    // Not all platforms support custom input device layouts. In such cases,
    // we use the only choice i.e. the default layout.
    if (absl::IsUnimplemented(device_buffers.status())) {
      return client.BufferFromHostLiteral(literal, memory_space,
                                          /*device_layout=*/nullptr);
    }
    return device_buffers;
  };

  ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                   executable->GetHloModules());
  CHECK_EQ(hlo_modules.size(), 1);
  HloModule* module = hlo_modules.front().get();

  absl::Status status = [&]() -> absl::Status {
    if (num_addressable_devices == 0) {
      return absl::OkStatus();
    }
    constexpr int64_t kMinDeviceCloneBytes = 64 * 1024;      // 64 KiB
    constexpr int64_t kMaxPendingTransferBytes = 2LL << 30;  // 2 GiB
    int64_t pending_transfer_bytes = 0;
    std::vector<PjRtBuffer*> pending_buffers;

    // On TPU, `CopyToMemorySpace` within the same `PjRtMemorySpace` has no
    // intra-chip HBM-to-HBM path and bounces through host DRAM (`D2H + H2D`),
    // which is strictly slower than transferring directly from the host
    // `Literal` while polluting device DMA queues. Cross-device TPU copies use
    // direct ICI.
    const bool allow_same_memory_space_clone =
        (client.platform_id() != TpuId());

    std::vector<const std::vector<Literal>*> per_device_arguments(
        num_addressable_devices, nullptr);
    for (size_t i = 0; i < num_addressable_devices; ++i) {
      PjRtDevice* curr_device = addressable_devices[i];
      // 'source_device' determines where we get the input literal from; look up
      // by `PjRtDevice::global_device_id()` (which may differ from the local
      // device index on multi-host slices).
      PjRtDevice* source_device =
          addressable_devices[clone_device0_arguments ? 0 : i];
      auto it = arguments.find(source_device->global_device_id().value());
      if (it == arguments.end()) {
        return InvalidArgument(
            "The provided argument map does not contain arguments "
            "for device: %d",
            curr_device->id());
      }
      per_device_arguments[i] = &it->second;
    }

    const size_t num_args = per_device_arguments[0]->size();
    for (size_t i = 0; i < num_addressable_devices; ++i) {
      if (per_device_arguments[i]->size() != num_args) {
        return InvalidArgument(
            "Argument count mismatch on device %d: expected %v, got %v.",
            addressable_devices[i]->id(), num_args,
            per_device_arguments[i]->size());
      }
      argument_buffers[i].resize(num_args);
    }

    // Iterate outer over parameters and rotate the starting primary device
    // whenever an uncloned host-to-device transfer is issued so PCIe H2D
    // bandwidth and outbound cross-device ICI fan-out are distributed evenly
    // across local devices rather than bottlenecked on device 0.
    size_t next_primary_device = 0;
    for (size_t arg_i = 0; arg_i < num_args; ++arg_i) {
      const size_t start_dev = next_primary_device % num_addressable_devices;
      for (size_t step = 0; step < num_addressable_devices; ++step) {
        const size_t i = (start_dev + step) % num_addressable_devices;
        PjRtDevice* curr_device = addressable_devices[i];
        const Literal& literal = (*per_device_arguments[i])[arg_i];
        if (log_input) {
          LOG(INFO) << "device_id=" << curr_device->id()
                    << ", input = " << literal.ToString();
        }
        std::unique_ptr<PjRtBuffer> argument_buffer;
        if (literal.shape().IsArrayExcludingBuffer() &&
            literal.shape().is_static() &&
            literal.size_bytes() >= kMinDeviceCloneBytes &&
            arg_i < executable_parameter_layouts.size()) {
          ABSL_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                           argument_memory_space(module, curr_device,
                                                 static_cast<int>(arg_i)));
          argument_buffer = TryCloneArgumentBuffer(
              i, arg_i, literal, executable_parameter_layouts[arg_i],
              memory_space, per_device_arguments, argument_buffers,
              executable_parameter_layouts, allow_same_memory_space_clone);
        }
        if (argument_buffer == nullptr) {
          ABSL_ASSIGN_OR_RETURN(
              argument_buffer,
              buffer_from_host_literal(module, curr_device,
                                       static_cast<int>(arg_i), literal));
          ++next_primary_device;
        }
        PjRtBuffer* created_ptr = argument_buffer.get();
        argument_buffers[i][arg_i] = std::move(argument_buffer);
        pending_buffers.push_back(created_ptr);
        if (literal.shape().IsArrayExcludingBuffer()) {
          pending_transfer_bytes += std::max<int64_t>(0, literal.size_bytes());
        }
        // Periodically drain in-flight transfers every 2 GiB so pinned host
        // staging buffers do not accumulate unboundedly on models with many
        // large parameters.
        if (pending_transfer_bytes >= kMaxPendingTransferBytes) {
          for (PjRtBuffer* pending_buf : pending_buffers) {
            ABSL_RETURN_IF_ERROR(pending_buf->GetReadyFuture().Await());
          }
          pending_buffers.clear();
          pending_transfer_bytes = 0;
        }
      }
    }
    return absl::OkStatus();
  }();

  // BufferFromHostLiteral transfers read directly from the host `Literal`s in
  // `arguments` until each buffer's ready future completes, so every created
  // buffer must be awaited before returning (including on error, since callers
  // such as CreateArgumentsOnDevice own `arguments` in their local stack
  // frame).
  for (const auto& device_argument_buffers : argument_buffers) {
    for (const auto& device_buffer : device_argument_buffers) {
      if (device_buffer != nullptr) {
        status.Update(device_buffer->GetReadyFuture().Await());
      }
    }
  }
  ABSL_RETURN_IF_ERROR(status);
  return argument_buffers;
}

// Creates uninitialized arguments to run the given executable.
absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
CreateUninitializedArgumentsOnDevice(PjRtClient& client,
                                     const PjRtLoadedExecutable* executable,
                                     const RunningOptions& running_options,
                                     bool flatten_arguments = false) {
  absl::Span<PjRtDevice* const> addressable_devices =
      executable->addressable_devices();
  ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                   executable->GetHloModules());
  VLOG(1) << "FunctionalHloRunner: local_executable count = "
          << hlo_modules.size();
  CHECK_EQ(hlo_modules.size(), 1);
  const HloModule& hlo_module = *hlo_modules.front();

  LOG(INFO) << "Starting argument buffer shape calculation.";
  PerDeviceShapeVecType argument_shapes_per_device;
  for (int i = 0; i < static_cast<int>(addressable_devices.size()); ++i) {
    VLOG(3) << "Calculating fake argument shapes for device " << i;
    PjRtDevice* device = addressable_devices[i];

    std::vector<Shape> argument_shapes;
    if (flatten_arguments) {
      ABSL_RETURN_IF_ERROR(EnsureSingleTupleForFlattening(hlo_module));

      std::vector<Shape> original_argument_shapes =
          GetArgumentShapes(hlo_module);
      CHECK_EQ(original_argument_shapes.size(), 1);
      CHECK(original_argument_shapes.front().IsTuple());
      argument_shapes = original_argument_shapes.front().tuple_shapes();
    } else {
      argument_shapes = GetArgumentShapes(hlo_module);
    }

    argument_shapes_per_device[device->global_device_id().value()] =
        std::move(argument_shapes);
  }

  LOG(INFO) << "Starting argument buffer allocation.";
  int buffer_count = 0;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
      argument_buffers_per_device;
  argument_buffers_per_device.reserve(addressable_devices.size());
  for (int i = 0; i < static_cast<int>(addressable_devices.size()); ++i) {
    VLOG(3) << "Allocating fake arguments for device " << i;
    PjRtDevice* device = addressable_devices[i];
    ABSL_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                     device->default_memory_space());

    const int device_id = device->global_device_id().value();
    CHECK(argument_shapes_per_device.contains(device_id));
    const std::vector<Shape>& argument_shapes =
        argument_shapes_per_device.at(device_id);
    std::vector<std::unique_ptr<PjRtBuffer>> argument_buffers;
    argument_buffers.reserve(argument_shapes.size());

    for (const Shape& shape : argument_shapes) {
      if (running_options.log_input_output()) {
        LOG(INFO) << "device_id=" << device_id
                  << ", input = " << shape.ToString();
      }

      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> argument_buffer,
                       client.CreateUninitializedBuffer(shape, memory_space));
      argument_buffers.push_back(std::move(argument_buffer));
      buffer_count += 1;
    }

    argument_buffers_per_device.push_back(std::move(argument_buffers));
  }
  LOG(INFO) << "Allocated argument buffers: " << buffer_count;

  absl::Status status;
  for (const auto& argument_buffers : argument_buffers_per_device) {
    for (const auto& buffer : argument_buffers) {
      status.Update(buffer->GetReadyFuture().Await());
    }
  }
  ABSL_RETURN_IF_ERROR(status);
  LOG(INFO) << "Argument buffers are ready.";

  return argument_buffers_per_device;
}

// Sums the host `Literal` byte sizes across all static array leaf subshapes of
// `module`'s entry computation parameters, using `entry_computation_layout()`
// when present (matching `MakeFakeArguments`).
int64_t ComputeModuleParameterHostBytes(const HloModule& module) {
  int64_t total_bytes = 0;
  const HloModuleConfig& module_config = module.config();
  const HloInstruction::InstructionVector& params =
      module.entry_computation()->parameter_instructions();
  for (int i = 0; i < params.size(); ++i) {
    const Shape& param_shape = (module_config.has_entry_computation_layout() &&
                                module_config.entry_computation_layout()
                                    .parameter_layout(i)
                                    .shape()
                                    .is_static())
                                   ? module_config.entry_computation_layout()
                                         .parameter_layout(i)
                                         .shape()
                                   : params[i]->shape();
    ShapeUtil::ForEachSubshape(param_shape, [&](const Shape& subshape,
                                                const ShapeIndex& /*index*/) {
      if (subshape.IsArrayExcludingBuffer() && subshape.is_static()) {
        const int64_t bytes =
            ShapeUtil::ByteSizeOf(ShapeUtil::DeviceShapeToHostShape(subshape));
        if (bytes > 0) {
          total_bytes += bytes;
        }
      }
    });
  }
  return total_bytes;
}

// Creates fake arguments to run the given executable.
absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
CreateArgumentsOnDevice(PjRtClient& client,
                        const PjRtLoadedExecutable* executable,
                        const RunningOptions& running_options,
                        bool flatten_arguments = false,
                        std::minstd_rand0* engine = nullptr) {
  if (running_options.module_argument_mode ==
      ModuleArgumentMode::kUninitialized) {
    return CreateUninitializedArgumentsOnDevice(
        client, executable, running_options, flatten_arguments);
  }

  SlowOperationAlarm alarm(
      absl::Seconds(5),
      absl::StrFormat("Argument initialization is slow. Consider changing "
                      "--hlo_argument_mode."));
  const absl::Time init_start = absl::Now();

  absl::Span<PjRtDevice* const> addressable_devices =
      executable->addressable_devices();
  size_t num_addressable_devices = addressable_devices.size();

  PerDeviceLiteralVecType per_device_argument_literals;
  ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<HloModule>> hlo_modules,
                   executable->GetHloModules());
  VLOG(1) << "FunctionalHloRunner: local_executable count = "
          << hlo_modules.size();
  CHECK_EQ(hlo_modules.size(), 1);
  HloModule* my_hlo_module = hlo_modules.front().get();

  const bool use_random_inputs = running_options.module_argument_mode ==
                                     ModuleArgumentMode::kUseRandomInputs ||
                                 running_options.module_argument_mode ==
                                     ModuleArgumentMode::kUseSharedRandomInputs;
  // Scale the minimum cached tensor size with total module parameter footprint
  // (up to 16 MiB on multi-GiB models) so small per-layer tensors such as MoE
  // router gates and quantization/norm scales stay independent across layers
  // while bulk FFN/attention weights remain cached.
  constexpr int64_t kMinCachedArgumentBytesFloor = 64LL << 10;  // 64 KiB
  constexpr int64_t kMaxCachedArgumentBytesFloor = 16LL << 20;  // 16 MiB
  const int64_t min_cached_argument_bytes =
      use_random_inputs
          ? std::clamp<int64_t>(
                ComputeModuleParameterHostBytes(*my_hlo_module) / 256,
                kMinCachedArgumentBytesFloor, kMaxCachedArgumentBytesFloor)
          : kMinCachedArgumentBytesFloor;
  // When engine == nullptr under kUseRandomInputs, MakeFakeArguments creates a
  // fresh default seeded std::minstd_rand0 on each device iteration, producing
  // identical literals across all addressable devices. Share device 0's
  // generated literals across addressable devices in that case.
  const bool use_shared_inputs =
      running_options.module_argument_mode ==
          ModuleArgumentMode::kUseSharedRandomInputs ||
      running_options.module_argument_mode ==
          ModuleArgumentMode::kUseZerosAsInput ||
      (running_options.module_argument_mode ==
           ModuleArgumentMode::kUseRandomInputs &&
       engine == nullptr);
  absl::BitGen rd;

  std::optional<std::normal_distribution<double>> random_dist;
  if (running_options.module_argument_mode ==
      ModuleArgumentMode::kUseRandomNormalInputs) {
    // Create a normal distribution
    // mean = 0.0, standard deviation = 1.0 (gaussian(0,1))
    random_dist.emplace(0.0, 1.0);
  }
  std::function<double(std::minstd_rand0*)> float_generator =
      [&](std::minstd_rand0* engine) { return (*random_dist)(*engine); };

  for (int i = 0; i < num_addressable_devices; ++i) {
    VLOG(3) << "Creating fake arguments for device " << i;
    LiteralVec& argument_literals = per_device_argument_literals
        [addressable_devices[i]->global_device_id().value()];
    if (flatten_arguments) {
      ABSL_RETURN_IF_ERROR(EnsureSingleTupleForFlattening(*my_hlo_module));
    }
    if (running_options.module_argument_mode ==
            ModuleArgumentMode::kUseRandomNormalInputs ||
        running_options.module_argument_mode ==
            ModuleArgumentMode::kUseDeviceIdAsInput) {
      const auto params =
          my_hlo_module->entry_computation()->parameter_instructions();
      if (flatten_arguments) {
        CHECK_EQ(params.size(), 1);
        CHECK(params.front()->shape().IsTuple());
        argument_literals.reserve(
            params.front()->shape().tuple_shapes().size());
      } else {
        argument_literals.reserve(params.size());
      }
      for (int j = 0; j < params.size(); ++j) {
        Literal argument_literal_j;
        if (running_options.module_argument_mode ==
            ModuleArgumentMode::kUseRandomNormalInputs) {
          // Create a random number engine
          // std::random_device provides a non-deterministic seed (from
          // hardware/OS)
          std::minstd_rand0 minstd(rd());

          ABSL_ASSIGN_OR_RETURN(
              argument_literal_j,
              xla::MakeFakeLiteral(
                  params[j]->shape(), &minstd, std::nullopt,
                  /*is_sorted=*/false,
                  /*no_duplicates=*/false, /*use_large_range=*/false,
                  /*max_bits_of_precision=*/std::nullopt,
                  /*index_alignment=*/std::nullopt,
                  /*index_known_zeroes=*/std::nullopt, float_generator));
        } else {
          ABSL_ASSIGN_OR_RETURN(
              argument_literal_j,
              MakeFakeLiteralWithSameValue(
                  params[j]->shape(),
                  addressable_devices[i]->global_device_id().value()));
        }
        ShapeUtil::ForEachSubshape(
            argument_literal_j.shape(),
            [&](const Shape& subshape, const ShapeIndex& index) {
              if (subshape.IsArray()) {
                for (int64_t k = 0; k < subshape.dimensions().size(); ++k) {
                  if (subshape.is_dynamic_dimension(k)) {
                    argument_literal_j.SetDynamicSize(k, index, 0);
                  }
                }
              }
            });
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
      FakeArgumentsOptions options;
      options.engine = engine;
      options.pseudo_random = use_random_inputs;
      options.cache_arguments = use_random_inputs;
      options.min_cached_argument_bytes = min_cached_argument_bytes;
      if (flatten_arguments) {
        ABSL_ASSIGN_OR_RETURN(LiteralVec tupled_argument_literals,
                         MakeFakeArguments(my_hlo_module, options));
        CHECK_EQ(tupled_argument_literals.size(), 1);
        CHECK(tupled_argument_literals.front().shape().IsTuple());
        argument_literals = tupled_argument_literals.front().DecomposeTuple();
      } else {
        ABSL_ASSIGN_OR_RETURN(argument_literals,
                         MakeFakeArguments(my_hlo_module, options));
      }
      if (use_shared_inputs) {
        break;
      }
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> device_buffers,
      CopyArgumentsToDevice(client, executable, per_device_argument_literals,
                            running_options, flatten_arguments,
                            /*clone_device0_arguments=*/use_shared_inputs));
  const double elapsed_secs = absl::ToDoubleSeconds(absl::Now() - init_start);
  LOG(INFO) << "Initialized and loaded arguments on " << num_addressable_devices
            << " addressable device(s) in " << elapsed_secs << "s.";
  return device_buffers;
}

// Creates an ExecutableBuildOptions using the specified ExecutionOptions.
ExecutableBuildOptions CreateExecutableBuildOptionsFromExecutionOptions(
    const ExecutionOptions& execution_options) {
  ExecutableBuildOptions build_options;
  if (execution_options.has_debug_options()) {
    *build_options.mutable_debug_options() = execution_options.debug_options();
  }
  if (execution_options.has_shape_with_output_layout()) {
    absl::StatusOr<Shape> shape =
        Shape::FromProto(execution_options.shape_with_output_layout());
    CHECK_OK(shape.status());
    build_options.set_result_layout(*shape);
  }
  build_options.set_num_replicas(execution_options.num_replicas());
  build_options.set_num_partitions(execution_options.num_partitions());
  build_options.set_use_spmd_partitioning(
      execution_options.use_spmd_partitioning());
  build_options.set_use_shardy_partitioner(
      execution_options.use_shardy_partitioner());
  build_options.set_use_auto_spmd_partitioning(
      execution_options.use_auto_spmd_partitioning());
  build_options.set_deduplicate_hlo(execution_options.deduplicate_hlo());
  build_options.set_allow_spmd_sharding_propagation_to_parameters(
      execution_options.allow_spmd_sharding_propagation_to_parameters());
  build_options.set_allow_spmd_sharding_propagation_to_output(
      execution_options.allow_spmd_sharding_propagation_to_output());
  if (execution_options.has_device_assignment()) {
    absl::StatusOr<std::unique_ptr<DeviceAssignment>> device_assignment =
        DeviceAssignment::Deserialize(execution_options.device_assignment());
    CHECK_OK(device_assignment.status());
    build_options.set_device_assignment(**device_assignment);
  }
  build_options.set_alias_passthrough_params(
      execution_options.alias_passthrough_params());
  return build_options;
}

}  // namespace

absl::StatusOr<ExecutionOptions> LoadExecutionOptions(absl::string_view path) {
  ExecutionOptions execution_options;
  ABSL_RETURN_IF_ERROR(tsl::ReadTextOrBinaryProto(
      tsl::Env::Default(), std::string(path), &execution_options));
  return execution_options;
}

namespace {

// Shared implementation of the `CreateCompileOptions` overloads. A client or a
// topology is only needed for `device_count` (used to infer the number of
// replicas and partitions) and `get_default_device_assignment`.
absl::StatusOr<CompileOptions> CreateCompileOptionsInternal(
    int device_count,
    absl::FunctionRef<absl::StatusOr<DeviceAssignment>(int num_replicas,
                                                       int num_partitions)>
        get_default_device_assignment,
    const FunctionalHloRunner::RawCompileOptions& raw_options, int task_id,
    int num_nodes, std::shared_ptr<xla::KeyValueStoreInterface> kv_store) {
  CompileOptions compile_options;
  if (raw_options.execution_options.has_value()) {
    compile_options.executable_build_options =
        CreateExecutableBuildOptionsFromExecutionOptions(
            raw_options.execution_options.value());
    if (compile_options.executable_build_options.has_debug_options()) {
      // Preserve the xla_dump_to path from other sources (e.g. via
      // RawCompileOptions.xla_dump_to or DebugOptions parsed from XLA_FLAGS).
      // This will also reset it if xla_dump_to is not specified.
      std::string xla_dump_to = raw_options.debug_options.xla_dump_to().empty()
                                    ? raw_options.xla_dump_to
                                    : raw_options.debug_options.xla_dump_to();
      compile_options.executable_build_options.mutable_debug_options()
          ->set_xla_dump_to(xla_dump_to);
      // Also preserve other dump related flags.
      if (raw_options.debug_options.has_xla_dump_hlo_as_text()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_as_text(
                raw_options.debug_options.xla_dump_hlo_as_text());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_as_proto()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_as_proto(
                raw_options.debug_options.xla_dump_hlo_as_proto());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_as_dot()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_as_dot(
                raw_options.debug_options.xla_dump_hlo_as_dot());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_as_url()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_as_url(
                raw_options.debug_options.xla_dump_hlo_as_url());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_as_riegeli()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_as_riegeli(
                raw_options.debug_options.xla_dump_hlo_as_riegeli());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_as_html()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_as_html(
                raw_options.debug_options.xla_dump_hlo_as_html());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_pass_re()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_pass_re(
                raw_options.debug_options.xla_dump_hlo_pass_re());
      }
      if (raw_options.debug_options.has_xla_dump_hlo_module_re()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_hlo_module_re(
                raw_options.debug_options.xla_dump_hlo_module_re());
      }
      if (raw_options.debug_options.has_xla_dump_emitter_re()) {
        compile_options.executable_build_options.mutable_debug_options()
            ->set_xla_dump_emitter_re(
                raw_options.debug_options.xla_dump_emitter_re());
      }
    }
  }

  // Fall back to `raw_options.debug_options` if necessary.
  if (!compile_options.executable_build_options.has_debug_options()) {
    // Guard against `raw_options.debug_options` being uninitialized.
    *compile_options.executable_build_options.mutable_debug_options() =
        GetDebugOptionsFromFlags();
    compile_options.executable_build_options.mutable_debug_options()->MergeFrom(
        raw_options.debug_options);
  }

  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  ReplicasAndPartitions replicas_and_partitions =
      FunctionalHloRunner::GetReplicasAndPartitions(
          raw_options.execution_options, device_count, raw_options.num_replicas,
          raw_options.num_partitions, raw_options.num_slices.value_or(1));
  build_options.set_num_replicas(replicas_and_partitions.replicas);
  build_options.set_num_partitions(replicas_and_partitions.partitions);
  build_options.set_process_index(task_id);
  build_options.set_process_count(num_nodes);
  build_options.set_key_value_store(kv_store);
  if (raw_options.spmd_mode == SpmdMode::kUseSpmdPartitioning ||
      raw_options.spmd_mode == SpmdMode::kUseShardyPartitioning) {
    build_options.set_use_spmd_partitioning(true);
    if (raw_options.spmd_mode == SpmdMode::kUseShardyPartitioning) {
      build_options.set_use_shardy_partitioner(true);
    }
  }
  if (!build_options.has_device_assignment() &&
      !raw_options.num_slices.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        DeviceAssignment device_assignment,
        get_default_device_assignment(replicas_and_partitions.replicas,
                                      replicas_and_partitions.partitions));
    build_options.set_device_assignment(device_assignment);
  }
  DebugOptions& debug_options = *build_options.mutable_debug_options();
  if (task_id == 0) {
    // Overwrite xla_dump_to only if it's not empty, to preserve `xla_dump_to`
    // from parsed XLA_FLAGS env (already populated in debug_options).
    if (!raw_options.xla_dump_to.empty()) {
      debug_options.set_xla_dump_to(raw_options.xla_dump_to);
      debug_options.set_xla_dump_hlo_as_text(raw_options.xla_text_dump_mode ==
                                             XlaTextDumpMode::kDumpAsText);
      debug_options.set_xla_dump_hlo_as_proto(raw_options.xla_proto_dump_mode ==
                                              XlaProtoDumpMode::kDumpAsProto);
    }
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

}  // namespace

absl::StatusOr<CompileOptions> CreateCompileOptions(
    const PjRtClient& client,
    const FunctionalHloRunner::RawCompileOptions& raw_options, int task_id,
    int num_nodes, std::shared_ptr<xla::KeyValueStoreInterface> kv_store) {
  return CreateCompileOptionsInternal(
      client.device_count(),
      [&](int num_replicas, int num_partitions) {
        return client.GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      raw_options, task_id, num_nodes, std::move(kv_store));
}

absl::StatusOr<CompileOptions> CreateCompileOptions(
    const PjRtTopologyDescription& topology,
    const FunctionalHloRunner::RawCompileOptions& raw_options, int task_id,
    int num_nodes, std::shared_ptr<xla::KeyValueStoreInterface> kv_store) {
  return CreateCompileOptionsInternal(
      topology.DeviceDescriptions().size(),
      [&](int num_replicas, int num_partitions) {
        return topology.GetDefaultDeviceAssignment(
            /*process_index=*/task_id, num_replicas,
            /*num_replicas_per_slice=*/std::nullopt, num_partitions,
            /*multi_slice_config=*/nullptr);
      },
      raw_options, task_id, num_nodes, std::move(kv_store));
}

// Dumps the output literals to the specified path.
// Example:
//  `dump_output_to` filepath: /tmp/output.pb
//  Actual output filepath: /tmp/output_task_0.device_0.literal_0.pb
absl::Status DumpOutput(
    const FunctionalHloRunner::PerDeviceLiteralVecType& output,
    absl::string_view dump_output_to, int task_id, OutputFormat output_format) {
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
  std::vector<std::function<absl::Status()>> write_tasks;
  for (const auto& [device_id, literal_vec] : output) {
    output_path_vec[device_id_index] = absl::StrCat("device_", device_id);
    for (int literal_id = 0; literal_id < literal_vec.size(); ++literal_id) {
      output_path_vec[literal_id_index] = absl::StrCat("literal_", literal_id);
      std::string literal_path = absl::StrJoin(output_path_vec, ".");
      switch (output_format) {
        case OutputFormat::kText: {
          CHECK_EQ(suffix, std::string("txt"));
          auto& literal = literal_vec[literal_id];
          write_tasks.push_back([literal_path, &literal]() {
            return tsl::WriteStringToFile(tsl::Env::Default(), literal_path,
                                          literal.ToString());
          });
          break;
        }
        case OutputFormat::kProtoBinary: {
          CHECK_EQ(suffix, std::string("pb"));
          auto& literal = literal_vec[literal_id];
          write_tasks.push_back([literal_path, &literal]() {
            return tsl::WriteBinaryProto(tsl::Env::Default(), literal_path,
                                         literal.ToProto());
          });
          break;
        }
        case OutputFormat::kProtoText: {
          CHECK(suffix == "pbtxt" || suffix == "textproto");
          auto& literal = literal_vec[literal_id];
          write_tasks.push_back([literal_path, &literal]() {
            return tsl::WriteTextProto(tsl::Env::Default(), literal_path,
                                       literal.ToProto());
          });
          break;
        }
      }
    }
  }
  std::vector<absl::Status> results;
  results.resize(write_tasks.size());
  {
    tsl::Env* env = tsl::Env::Default();
    tsl::thread::ThreadPool thread_pool(env, "XlaHloRunner_DumpOutput", 16);
    for (int i = 0; i < write_tasks.size(); ++i) {
      thread_pool.Schedule(
          [&write_tasks, &results, i]() { results[i] = write_tasks[i](); });
    }
  }
  for (const auto& result : results) {
    ABSL_RETURN_IF_ERROR(result) << "Failed to dump output";
  }
  LOG(INFO) << "Dumped output to " << write_tasks.size() << " files.";
  return absl::OkStatus();
}

absl::StatusOr<HloModuleAndArguments> LoadHloModuleAndArguments(
    absl::string_view hlo_file, InputFormat input_format) {
  HloModuleAndArguments hlo_module_and_arguments;
  switch (input_format) {
    case InputFormat::kText: {
      ABSL_ASSIGN_OR_RETURN(
          hlo_module_and_arguments.hlo_module,
          ReadModuleFromHloTextFile(
              hlo_file, DebugOptions::default_instance(),
              HloParserOptions().set_keep_module_auto_layouts(true)));
      break;
    }
    case InputFormat::kProtoText: {
      ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                       ReadModuleFromTextProtoFile(hlo_file));
      break;
    }
    case InputFormat::kProtoBinary: {
      ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                       ReadModuleFromBinaryProtoFile(hlo_file));
      break;
    }
    case InputFormat::kSnapshotProtoBinary: {
      ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments,
                       ReadModuleFromSnapshotBinaryProtoFile(hlo_file));
      break;
    }
    case InputFormat::kUnoptimizedSnapshotProtoBinary: {
      ABSL_ASSIGN_OR_RETURN(
          hlo_module_and_arguments,
          ReadModuleFromUnoptimizedSnapshotBinaryProtoFile(hlo_file));
      break;
    }
    case InputFormat::kUnoptimizedSnapshotProtoText: {
      ABSL_ASSIGN_OR_RETURN(
          hlo_module_and_arguments,
          ReadModuleFromUnoptimizedSnapshotTextProtoFile(hlo_file));
      break;
    }
    case InputFormat::kSerializedPjRtExecutable: {
      LOG(INFO) << "Skipping loading HLO module and arguments for serialized "
                   "PjRtExecutable.";
      return hlo_module_and_arguments;
      break;
    }
    case InputFormat::kStableHlo: {
      ABSL_ASSIGN_OR_RETURN(hlo_module_and_arguments.hlo_module,
                       ReadStableHloModule(hlo_file));
      break;
    }
    default:
      LOG(FATAL) << "Cannot process input format: "
                 << AbslUnparseFlag(input_format);
  }
  return hlo_module_and_arguments;
}

absl::Status LoadAndRunAndDump(
    PjRtClient& client,
    const xla::FunctionalHloRunner::PreprocessingOptions& preproc_options,
    const xla::FunctionalHloRunner::RawCompileOptions& raw_compile_options,
    const xla::FunctionalHloRunner::RunningOptions& running_options,
    absl::string_view hlo_file, InputFormat input_format,
    std::string dump_output_to, int task_id, int num_nodes,
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
    std::minstd_rand0* engine) {
  ABSL_ASSIGN_OR_RETURN(
      CompileOptions compile_options,
      FunctionalHloRunner::CreateCompileOptions(client, raw_compile_options,
                                                task_id, num_nodes, kv_store));
  ABSL_ASSIGN_OR_RETURN(
      FunctionalHloRunner::PerDeviceLiteralVecType output,
      FunctionalHloRunner::LoadAndRun(client, preproc_options, compile_options,
                                      running_options, hlo_file, input_format,
                                      /*arguments=*/{}, engine));
  LOG(INFO) << "Dump output to: " << dump_output_to;
  return dump_output_to.empty()
             ? absl::OkStatus()
             : FunctionalHloRunner::DumpOutput(output, dump_output_to, task_id);
}

absl::Status LoadAndCompileAndDump(
    PjRtClient& client,
    const xla::FunctionalHloRunner::PreprocessingOptions& preproc_options,
    const xla::FunctionalHloRunner::RawCompileOptions& raw_compile_options,
    absl::string_view hlo_file, InputFormat input_format,
    std::string dump_executable_to, int task_id, int num_nodes,
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
    bool use_gpu_count_workaround) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      FunctionalHloRunner::LoadAndCompile(
          client, preproc_options, raw_compile_options, hlo_file, input_format,
          task_id, num_nodes, kv_store, use_gpu_count_workaround));

  if (!dump_executable_to.empty()) {
    absl::StrAppend(&dump_executable_to, ".task_", task_id);
    ABSL_ASSIGN_OR_RETURN(std::string serialized, executable->SerializeExecutable());
    return tsl::WriteStringToFile(tsl::Env::Default(), dump_executable_to,
                                  serialized);
  }
  return absl::OkStatus();
}

absl::StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType> LoadAndRun(
    PjRtClient& client, const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options,
    const RunningOptions& running_options, absl::string_view hlo_file,
    InputFormat input_format, const PerDeviceLiteralVecType& arguments,
    std::minstd_rand0* engine) {
  // We only support SPMD as of now, i.e., all devices are supposed
  // to execute the same HLO module.
  // Currently there is no mechanism to map the loaded arguments to
  // proper device ID, so loading and executing from HLO snapshot might not
  // replay the original execution.

  const PerDeviceLiteralVecType* final_arguments = nullptr;
  std::unique_ptr<HloModule> hlo_module;
  PerDeviceLiteralVecType loaded_arguments;
  if (!arguments.empty()) {
    final_arguments = &arguments;
  } else {
    ABSL_ASSIGN_OR_RETURN(HloModuleAndArguments hlo_module_and_arguments,
                     LoadHloModuleAndArguments(hlo_file, input_format));

    // Check that the number of shards is not greater than the number of
    // devices.
    if (hlo_module_and_arguments.arguments.size() > client.devices().size()) {
      return absl::InvalidArgumentError(
          "The number of shards in the given input file is greater than the "
          "number of devices available on the host.");
    }

    for (int i = 0; i < hlo_module_and_arguments.arguments.size(); ++i) {
      loaded_arguments[client.devices()[i]->global_device_id().value()] =
          std::move(hlo_module_and_arguments.arguments[i]);
    }
    hlo_module = std::move(hlo_module_and_arguments.hlo_module);
    final_arguments = &loaded_arguments;
  }

  if (input_format == InputFormat::kSerializedPjRtExecutable) {
    std::string serialized_executable;
    ABSL_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(), std::string(hlo_file), &serialized_executable));
    std::unique_ptr<PjRtLoadedExecutable> executable;
    {
      XLA_SCOPED_LOGGING_TIMER("FunctionalHloRunner::LoadSerializedExecutable");
      ABSL_ASSIGN_OR_RETURN(executable, client.LoadSerializedExecutable(
                                       serialized_executable, compile_options,
                                       LoadOptions()));
    }
    return Run(client, executable.get(), *final_arguments, running_options,
               engine);
  }
  if (!hlo_module) {
    // Load hlo module.
    ABSL_ASSIGN_OR_RETURN(HloModuleAndArguments hlo_module_and_arguments,
                     LoadHloModuleAndArguments(hlo_file, input_format));
    hlo_module = std::move(hlo_module_and_arguments.hlo_module);
  }

  return CompileAndRun(client, preproc_options, compile_options,
                       running_options, hlo_module.get(), *final_arguments,
                       engine);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LoadAndCompile(
    PjRtClient& client, const PreprocessingOptions& preproc_options,
    const RawCompileOptions& raw_compile_options, absl::string_view hlo_file,
    InputFormat input_format, int task_id, int num_nodes,
    std::shared_ptr<xla::KeyValueStoreInterface> kv_store,
    bool use_gpu_count_workaround) {
  ABSL_ASSIGN_OR_RETURN(
      CompileOptions compile_options,
      FunctionalHloRunner::CreateCompileOptions(client, raw_compile_options,
                                                task_id, num_nodes, kv_store));

  int num_replicas = compile_options.executable_build_options.num_replicas();
  int num_partitions =
      compile_options.executable_build_options.num_partitions();
  int needed_devices = num_replicas * num_partitions;
  if (client.addressable_device_count() < needed_devices &&
      use_gpu_count_workaround) {
    LOG(INFO) << "Applying a workaround to allow compiling multi-device HLOs "
                 "on machines with fewer devices.";
    DeviceAssignment assignment(num_replicas, num_partitions);
    assignment.Fill(0);
    compile_options.executable_build_options.set_device_assignment(assignment);
  }

  ABSL_ASSIGN_OR_RETURN(HloModuleAndArguments hlo_module_and_arguments,
                   LoadHloModuleAndArguments(hlo_file, input_format));

  return FunctionalHloRunner::Compile(client,
                                      hlo_module_and_arguments.hlo_module.get(),
                                      preproc_options, compile_options);
}

absl::Status VerifyExecutableFingerprintAcrossProcesses(
    KeyValueStoreInterface& kv_store, absl::string_view key_prefix,
    int process_index, int process_count, absl::string_view fingerprint,
    absl::Duration timeout) {
  if (process_index < 0 || process_index >= process_count) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Process index %d is out of range for %d processes.",
                        process_index, process_count));
  }
  ABSL_RETURN_IF_ERROR(
      kv_store.Set(absl::StrCat(key_prefix, process_index), fingerprint));

  // Maps each fingerprint to the processes that reported it.
  absl::btree_map<std::string, std::vector<int>> processes_by_fingerprint;
  const absl::Time deadline = absl::Now() + timeout;
  for (int i = 0; i < process_count; ++i) {
    if (i == process_index) {
      processes_by_fingerprint[std::string(fingerprint)].push_back(i);
      continue;
    }
    absl::StatusOr<std::string> other_fingerprint =
        kv_store.Get(absl::StrCat(key_prefix, i),
                     std::max(deadline - absl::Now(), absl::ZeroDuration()));
    if (!other_fingerprint.ok()) {
      return absl::Status(
          other_fingerprint.status().code(),
          absl::StrFormat("Failed to get the executable fingerprint of process "
                          "%d, which might have failed to compile: %s",
                          i, other_fingerprint.status().message()));
    }
    processes_by_fingerprint[*std::move(other_fingerprint)].push_back(i);
  }

  if (processes_by_fingerprint.size() == 1) {
    return absl::OkStatus();
  }
  if (processes_by_fingerprint.contains("")) {
    LOG(WARNING) << "Not all processes could fingerprint their executable, so "
                    "we can't tell whether they compiled the same executable.";
    return absl::OkStatus();
  }
  std::vector<std::string> fingerprint_descriptions;
  fingerprint_descriptions.reserve(processes_by_fingerprint.size());
  for (const auto& [other_fingerprint, processes] : processes_by_fingerprint) {
    fingerprint_descriptions.push_back(
        absl::StrFormat("%s on process(es) %s", other_fingerprint,
                        absl::StrJoin(processes, ", ")));
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "The processes compiled different executables (fingerprints: %s). "
      "Running them would likely hang because the processes can issue "
      "collectives in a different order. A common cause is autotuning with "
      "--xla_gpu_shard_autotuning=false, since then each process picks its "
      "own configs. Compare the optimized HLO of the processes (e.g. with "
      "--xla_dump_to) to find the difference. This check can be disabled with "
      "verify_executable_fingerprint_across_processes.",
      absl::StrJoin(fingerprint_descriptions, "; ")));
}

namespace {

// Processes can finish compiling at very different times (e.g. because of
// autotuning), so we use the same generous timeout as the exchange of sharded
// autotuning results.
constexpr absl::Duration kExecutableFingerprintTimeout = absl::Hours(24);

// Verifies that all processes compiled the same executable if this is enabled
// and the compile options contain a key-value store and more than one process.
//
// This assumes that all processes compile the same modules in the same order
// (and not concurrently), so that the n-th verification refers to the same
// module on all processes.
absl::Status MaybeVerifyExecutableFingerprintAcrossProcesses(
    const PjRtClient& client, const PjRtLoadedExecutable& executable,
    const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options) {
  const ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  if (!preproc_options.verify_executable_fingerprint_across_processes ||
      build_options.key_value_store() == nullptr ||
      build_options.process_count() <= 1) {
    return absl::OkStatus();
  }
  // Keys can't be overwritten, so every verification uses its own keys.
  static std::atomic<int64_t> next_verification_id{0};
  const int64_t verification_id = next_verification_id.fetch_add(1);

  std::string fingerprint;
  if (absl::StatusOr<std::string> result = executable.FingerprintExecutable();
      result.ok()) {
    fingerprint = *std::move(result);
  } else {
    LOG(WARNING) << "Failed to fingerprint the executable: " << result.status();
  }
  LOG(INFO) << "Executable fingerprint of process " << client.process_index()
            << ": " << fingerprint;
  // Callers don't always set the process index in the build options, so we
  // take it from the client.
  return VerifyExecutableFingerprintAcrossProcesses(
      *build_options.key_value_store(),
      absl::StrCat("functional_hlo_runner/executable_fingerprint/",
                   verification_id, "/"),
      client.process_index(), build_options.process_count(), fingerprint,
      kExecutableFingerprintTimeout);
}

}  // namespace

absl::StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType> CompileAndRun(
    PjRtClient& client, const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options,
    const RunningOptions& running_options, HloModule* hlo_module,
    const PerDeviceLiteralVecType& arguments, std::minstd_rand0* engine) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      Compile(client, hlo_module, preproc_options, compile_options));

  return Run(client, executable.get(), arguments, running_options, engine);
}

absl::StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType> CompileAndRun(
    PjRtClient& client, const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options,
    const RunningOptions& running_options, MaybeOwningMlirModule module,
    const PerDeviceLiteralVecType& arguments, std::minstd_rand0* engine) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                   client.CompileAndLoad(std::move(module), compile_options));
  ABSL_RETURN_IF_ERROR(MaybeVerifyExecutableFingerprintAcrossProcesses(
      client, *executable, preproc_options, compile_options));

  return Run(client, executable.get(), arguments, running_options, engine);
}

absl::Status PrepareHloModuleForCompilation(
    HloModule* hlo_module, const DebugOptions& debug_options,
    const PreprocessingOptions& preproc_options) {
  hlo_module->mutable_config().set_debug_options(debug_options);

  if (preproc_options.is_spmd_partitioned_module()) {
    // If the module has already been partitioned by SPMD, add sharding
    // annotations (replicated) to module parameters and result.
    AddShardingAnnotationsToSpmdPartitionedModule(hlo_module);
  }

  if (preproc_options.flatten_while_loop() ||
      preproc_options.remove_infeed_outfeed ||
      preproc_options.flatten_conditional) {
    // The pipeline will check for the presence of
    // debug_options().xla_disable_hlo_passes().
    HloPassPipeline pipeline("control-flow-flattening-pipeline");
    int while_execution_count =
        preproc_options.while_execution_count.value_or(0);
    pipeline.AddPass<HloControlFlowFlattening>(
        HloControlFlowFlattening::Options{
            /*while_execution_count=*/while_execution_count,
            /*max_outer_loop_count=*/
            while_execution_count,
            /*max_loop_count=*/
            while_execution_count,
            /*remove_infeed_outfeed=*/preproc_options.remove_infeed_outfeed,
            /*flatten_while_loop=*/preproc_options.flatten_while_loop(),
            /*remove_comm=*/false,
            /*remove_host_transfer=*/true,
            /*remove_id=*/false,
            /*flatten_conditional=*/
            preproc_options.flatten_conditional,
            /*conditional_value=*/
            preproc_options.conditional_value});
    if (preproc_options.annotate_while_loop_trip_count) {
      pipeline.AddPass<WhileLoopTripCountAnnotator>();
    }
    ABSL_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }
  return absl::OkStatus();
}

absl::StatusOr<CompileOptions> CompleteCompileOptions(
    const HloModule& hlo_module, CompileOptions compile_options,
    const PreprocessingOptions& preproc_options) {
  ParameterType parameter_type = GetParameterType(hlo_module);
  compile_options.parameter_is_tupled_arguments =
      (parameter_type == ParameterType::kOneTupleOfArrays);
  if (preproc_options.force_auto_layout) {
    XlaComputation computation(hlo_module.ToProto());
    ABSL_ASSIGN_OR_RETURN(ProgramShape program_shape, computation.GetProgramShape());
    LayoutUtil::ClearLayout(&program_shape);
    compile_options.argument_layouts = program_shape.parameters();
    compile_options.executable_build_options.set_result_layout(
        program_shape.result());
    compile_options.executable_build_options.mutable_debug_options()
        ->set_xla_pjrt_allow_auto_layout_in_hlo(true);
  } else if (preproc_options.use_layouts_from_hlo_module) {
    const ComputationLayout& layout = hlo_module.entry_computation_layout();
    std::vector<Shape> parameter_shapes;
    parameter_shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& shape_layout : layout.parameter_layouts()) {
      parameter_shapes.push_back(shape_layout.shape());
    }
    compile_options.argument_layouts = std::move(parameter_shapes);
    compile_options.executable_build_options.set_result_layout(
        layout.result_shape());
    compile_options.executable_build_options.mutable_debug_options()
        ->set_xla_pjrt_allow_auto_layout_in_hlo(true);
  }
  return compile_options;
}

namespace {

// Depending on the compile_as_stablehlo flag, convert the HLO module either to
// StableHLO mlir::Module or to XlaComputation and calls the compile_function
// which should take either of these as input.
template <typename R, typename T>
absl::StatusOr<std::unique_ptr<R>> ConvertAndCallCompiler(
    bool compile_as_stablehlo, HloModule* hlo_module, T&& compile_function) {
  auto compile_and_log =
      [&](auto&& module) -> absl::StatusOr<std::unique_ptr<R>> {
    VLOG(1) << "FunctionalHloRunner: compilation started.";
    ABSL_ASSIGN_OR_RETURN(auto result,
                     compile_function(std::forward<decltype(module)>(module)));
    VLOG(1) << "FunctionalHloRunner: compile succeeded.";
    return result;
  };

  if (compile_as_stablehlo) {
    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);
    auto context = std::make_unique<mlir::MLIRContext>(registry);
    ABSL_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> stablehlo_module,
                     ConvertHloToStablehlo(*context, hlo_module));
    return compile_and_log(
        MaybeOwningMlirModule(std::move(context), std::move(stablehlo_module)));
  }
  XlaComputation computation(hlo_module->ToProto());
  return compile_and_log(computation);
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
    PjRtClient& client, HloModule* hlo_module,
    const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options) {
  TF_RET_CHECK(compile_options.executable_build_options.has_debug_options())
      << "debug_options must be set in compile_options";
  const DebugOptions& debug_options =
      compile_options.executable_build_options.debug_options();
  ABSL_RETURN_IF_ERROR(PrepareHloModuleForCompilation(hlo_module, debug_options,
                                                 preproc_options));
  ABSL_ASSIGN_OR_RETURN(
      CompileOptions modified_compile_options,
      CompleteCompileOptions(*hlo_module, compile_options, preproc_options));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      ConvertAndCallCompiler<PjRtLoadedExecutable>(
          preproc_options.compile_as_stablehlo, hlo_module, [&](auto&& module) {
            return client.CompileAndLoad(std::forward<decltype(module)>(module),
                                         modified_compile_options);
          }));
  ABSL_RETURN_IF_ERROR(MaybeVerifyExecutableFingerprintAcrossProcesses(
      client, *executable, preproc_options, modified_compile_options));
  return executable;
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
    PjRtClient& client, HloModule* hlo_module,
    const PreprocessingOptions& preproc_options,
    const CompileOptions& compile_options,
    const PjRtTopologyDescription& topology) {
  TF_RET_CHECK(compile_options.executable_build_options.has_debug_options())
      << "debug_options must be set in compile_options";
  const DebugOptions& debug_options =
      compile_options.executable_build_options.debug_options();
  ABSL_RETURN_IF_ERROR(PrepareHloModuleForCompilation(hlo_module, debug_options,
                                                 preproc_options));
  ABSL_ASSIGN_OR_RETURN(
      CompileOptions modified_compile_options,
      CompleteCompileOptions(*hlo_module, compile_options, preproc_options));

  return ConvertAndCallCompiler<PjRtExecutable>(
      preproc_options.compile_as_stablehlo, hlo_module, [&](auto&& module) {
        return PjRtCompile(modified_compile_options,
                           std::forward<decltype(module)>(module), topology,
                           &client);
      });
}

// Runs the executable and may repeat for multiple times.
absl::StatusOr<FunctionalHloRunner::PerDeviceLiteralVecType> Run(
    PjRtClient& client, PjRtLoadedExecutable* executable,
    const PerDeviceLiteralVecType& arguments,
    const RunningOptions& running_options, std::minstd_rand0* engine) {
  auto create_argument_buffers_on_device = [&client, &executable, &arguments,
                                            &running_options, engine](
                                               bool flatten_tupled_arguments) {
    if (arguments.empty()) {
      return CreateArgumentsOnDevice(client, executable, running_options,
                                     flatten_tupled_arguments, engine);
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
      return CopyArgumentsToDevice(client, executable, flattened_arguments,
                                   running_options,
                                   /*flattened_arguments=*/true);
    }
    // If the per-device argument is not a single tuple, we ignore the
    // flatten_tupled_arguments parameter and assume the provided arguments have
    // already been flattened.
    return CopyArgumentsToDevice(client, executable, arguments, running_options,
                                 /*flattened_arguments=*/false);
  };
  return RunInternal(client, executable, create_argument_buffers_on_device,
                     running_options);
}

namespace {
const FixedOptionSetFlagParser<FunctionalHloRunner::ModuleOutputMode>&
GetModuleOutputModeParser() {
  using ModuleOutputMode = FunctionalHloRunner::ModuleOutputMode;
  static const auto& parser = GetFixedOptionSetFlagParser<ModuleOutputMode>(
      {{"return_outputs", ModuleOutputMode::kReturnOutputs},
       {"not_return_outputs", ModuleOutputMode::kNotReturnOutputs},
       {"return_device_0_outputs", ModuleOutputMode::kReturnDevice0Outputs}});
  return parser;
}
const FixedOptionSetFlagParser<FunctionalHloRunner::ModuleArgumentMode>&
GetModuleArgumentModeParser() {
  using ModuleArgumentMode = FunctionalHloRunner::ModuleArgumentMode;
  static const auto& parser = GetFixedOptionSetFlagParser<ModuleArgumentMode>(
      {{"use_device_id_as_input", ModuleArgumentMode::kUseDeviceIdAsInput},
       {"use_random_inputs", ModuleArgumentMode::kUseRandomInputs},
       {"use_shared_random_inputs", ModuleArgumentMode::kUseSharedRandomInputs},
       {"use_zeros_as_input", ModuleArgumentMode::kUseZerosAsInput},
       {"uninitialized", ModuleArgumentMode::kUninitialized},
       {"use_random_normal_inputs",
        ModuleArgumentMode::kUseRandomNormalInputs}});
  return parser;
}
}  // namespace

bool AbslParseFlag(absl::string_view text, ModuleArgumentMode* argument_mode,
                   std::string* error) {
  return GetModuleArgumentModeParser().Parse(text, argument_mode, error);
}

std::string AbslUnparseFlag(ModuleArgumentMode argument_mode) {
  return GetModuleArgumentModeParser().Unparse(argument_mode);
}

bool AbslParseFlag(absl::string_view text, ModuleOutputMode* output_mode,
                   std::string* error) {
  return GetModuleOutputModeParser().Parse(text, output_mode, error);
}

std::string AbslUnparseFlag(ModuleOutputMode output_mode) {
  return GetModuleOutputModeParser().Unparse(output_mode);
}

absl::StatusOr<ResolveTopologyResult> ResolveTopology(
    std::optional<int> num_replicas, std::optional<int> num_partitions,
    absl::string_view hlo_file, InputFormat input_format,
    const HloModule* already_loaded_module) {
  ResolveTopologyResult result;
  const HloModule* module = already_loaded_module;
  if (module == nullptr && !hlo_file.empty()) {
    ABSL_ASSIGN_OR_RETURN(result.loaded_module,
                     LoadHloModuleAndArguments(hlo_file, input_format));
    module = result.loaded_module->hlo_module.get();
  }

  int resolved_replicas = 1;
  int resolved_partitions = 1;

  if (num_replicas.has_value() && *num_replicas >= 0) {
    resolved_replicas = *num_replicas;
  } else if (module != nullptr) {
    resolved_replicas = module->config().replica_count();
  }

  if (num_partitions.has_value() && *num_partitions >= 0) {
    resolved_partitions = *num_partitions;
  } else if (module != nullptr) {
    resolved_partitions = module->config().num_partitions();
  }

  result.topology.num_replicas = resolved_replicas;
  result.topology.num_partitions = resolved_partitions;
  result.topology.num_nodes = resolved_replicas * resolved_partitions;

  return result;
}

}  // namespace FunctionalHloRunner

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

}  // namespace xla
