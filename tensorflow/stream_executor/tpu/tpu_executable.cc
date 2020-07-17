/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/tpu/tpu_executable.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_execute_c_api.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"

namespace xla {
namespace tpu {
namespace {

using HostCommandHandler = TpuExecutable::HostCommandHandler;

// Write the tuple index buffers (arrays of pointers).
static Status PopulateResultTupleBuffers(const ShapedBuffer& result,
                                         se::Stream* stream,
                                         se::Stream* transfer_stream) {
  TF_ASSIGN_OR_RETURN(auto transfer_manager, TransferManager::GetForPlatform(
                                                 stream->parent()->platform()));
  if (transfer_manager->CanShapedBufferBeAccessedNow(stream->parent(),
                                                     result)) {
    TF_RETURN_IF_ERROR(transfer_manager->WriteTupleIndexTablesAsync(
        transfer_stream ? transfer_stream : stream, result));
    if (transfer_stream && transfer_stream != stream) {
      stream->ThenWaitFor(transfer_stream);
    }
    return Status::OK();
  } else {
    return transfer_manager->WriteTupleIndexTablesAsync(stream, result);
  }
}

xla::Shape HostShapeToDeviceShape(const xla::Shape& host_shape) {
  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;
  TpuConversions::XlaShapeToCShape(host_shape, &c_host_shape);
  tensorflow::tpu::ExecutorApiFn()->HardwareLayout_HostShapeToDeviceShapeFn(
      &c_host_shape, &c_device_shape);
  xla::Shape device_shape = TpuConversions::CShapeToXlaShape(&c_device_shape);
  TpuConversions::CShapeCleanup(&c_host_shape);
  TpuConversions::CShapeCleanup(&c_device_shape);
  return device_shape;
}

int64 ShapeSize(const xla::Shape& shape) {
  XLA_Shape c_shape;
  TpuConversions::XlaShapeToCShape(shape, &c_shape);
  int64 size =
      tensorflow::tpu::ExecutorApiFn()->HardwareLayout_ShapeSizeFn(&c_shape);
  TpuConversions::CShapeCleanup(&c_shape);
  return size;
}

}  // namespace

TpuExecutable::TpuExecutable(const XLA_TpuProgram* core_program,
                             std::unique_ptr<HloModule> hlo_module,
                             HostCommandHandler host_command_handler)
    : Executable(std::move(hlo_module), /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr),
      core_program_(core_program),
      host_command_handler_(std::move(host_command_handler)) {}

StatusOr<ExecutionOutput> TpuExecutable::AllocateOutputMemoryWithInputReuse(
    const Shape& host_shape, const HloInputOutputAliasConfig& alias_config,
    se::DeviceMemoryAllocator* allocator,
    std::vector<ExecutionInput>* arguments, se::Stream* stream,
    se::Stream* transfer_stream) {
  auto stream_exec = stream->parent();
  auto device_ordinal = stream_exec->device_ordinal();
  VLOG(3) << "AllocateOutputMemoryWithInputReuse, device = " << device_ordinal
          << " host_shape = " << ShapeUtil::HumanStringWithLayout(host_shape);
  Shape device_shape = HostShapeToDeviceShape(host_shape);

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "AllocateOutputMemoryWithInputReuse, device = " << device_ordinal
            << " host_shape = " << ShapeUtil::HumanStringWithLayout(host_shape);
    if (!Shape::Equal().MinorToMajorOnlyInLayout()(host_shape, device_shape)) {
      VLOG(3) << "Rewrote host_shape to device_shape: "
              << ShapeUtil::HumanStringWithLayout(host_shape) << " -> "
              << ShapeUtil::HumanStringWithLayout(device_shape);
    }
  }

  ExecutionOutput result(host_shape, std::move(device_shape), allocator,
                         device_ordinal);
  // Iterate through and allocate a buffer for each shape index, checking for
  // possible input buffer reuse.
  int64 reused_buffer_bytes = 0;
  int64 total_result_buffer_bytes = 0;
  for (auto& pair : result.MutableResult()->buffers()) {
    const ShapeIndex& result_index = pair.first;
    se::DeviceMemoryBase& result_buffer = pair.second;
    int64 allocation_bytes = ShapeSize(ShapeUtil::GetSubshape(
        result.Result().on_device_shape(), result_index));
    total_result_buffer_bytes += allocation_bytes;

    // Return an InternalError if result_index is invalid. This avoids failing
    // the CHECK when calling GetAliasedParameter
    if (!ShapeUtil::IndexIsValid(alias_config.shape(), result_index)) {
      return InternalError("result_index is invalid: %s",
                           result_index.ToString());
    }

    absl::optional<HloInputOutputAliasConfig::Alias> alias =
        alias_config.GetAliasedParameter(result_index);
    if (alias) {
      TF_RET_CHECK(alias->parameter_number < arguments->size());
      ExecutionInput& input = (*arguments)[alias->parameter_number];
      MaybeOwningDeviceMemory* device_memory =
          input.MutableBuffer(alias->parameter_index);
      if (auto owning = device_memory->Release()) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        se::DeviceMemoryBase device_memory_base = owning->Release();
        *device_memory = device_memory_base;
        result_buffer = device_memory_base;
        reused_buffer_bytes += allocation_bytes;
        // The caller is giving us the input buffer, but in case of error of the
        // execute call, we should not be releasing it as it contains valid data
        // (for example, it is a parameter which the user wants us to alias, in
        // a gradient update computation). So we store the index into the result
        // in the aliased vactor, which will be fed to the ExecutionOutput,
        // which will be using the indices to drop the addresses from its own
        // ScopedShapedBuffer result, if the ExecutionOutput is not committed.
        result.AddAliasedIndex(result_index);
      }
    }

    // We need to allocate a new output buffer for two cases:
    // - There is no alias between this output and any input.
    // - There is an alias, but the xla doesn't own the input memory so it can't
    // donate buffer to the computation.
    if (result_buffer.is_null()) {
      const Shape& on_device_shape = result.Result().on_device_shape();
      const Shape& on_device_subshape =
          ShapeUtil::GetSubshape(on_device_shape, result_index);
      TF_ASSIGN_OR_RETURN(
          auto allocated_buffer,
          allocator->Allocate(device_ordinal, allocation_bytes,
                              /*retry_on_failure=*/true,
                              on_device_subshape.layout().memory_space()));
      // Store the allocated buffer in our ScopedShapedBuffer, which takes
      // ownership.
      result_buffer = allocated_buffer.Release();
    }
    TF_RET_CHECK(allocation_bytes == 0 || result_buffer != nullptr);
  }

  VLOG(1) << "Reused " << reused_buffer_bytes
          << " parameter buffers (total result buffer size: "
          << total_result_buffer_bytes << ")";

  TF_RETURN_IF_ERROR(
      PopulateResultTupleBuffers(result.Result(), stream, transfer_stream));
  return std::move(result);
}

StatusOr<ExecutionOutput> TpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* /*hlo_execution_profile*/) {
  std::vector<se::DeviceMemoryBase> memory_bases;
  memory_bases.reserve(arguments.size());
  for (auto& argument : arguments) {
    memory_bases.push_back(argument.Buffer({}).AsDeviceMemoryBase());
  }
  se::Stream* stream = run_options->stream();

  CHECK_NE(run_options->allocator(), nullptr);
  const Shape& shape =
      hlo_module_ == nullptr ? ShapeUtil::MakeNil() : result_shape();
  const HloInputOutputAliasConfig& alias_config =
      hlo_module_ == nullptr ? HloInputOutputAliasConfig()
                             : hlo_module_->input_output_alias_config();
  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      AllocateOutputMemoryWithInputReuse(
          shape, alias_config, run_options->allocator(), &arguments, stream,
          run_options->run_options().host_to_device_stream()));

  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);

  // Address of the buffer in TPU memory that is being speculated.
  absl::optional<se::DeviceMemoryBase> cross_program_prefetch_addr;
  if (hlo_module_) {
    for (const auto& [parameter, index] :
         hlo_module_->CrossProgramPrefetches()) {
      CHECK_LT(parameter, arguments.size());
      // Ensure the cross program prefetched buffer doesn't alias with any
      // program outputs. If the input and output aliased, the buffer could be
      // invalidated during program execution and the program could read stale
      // data instead of fresh data.
      auto it = arguments[parameter].MutableBuffers()->find({index});
      CHECK(it != arguments[parameter].MutableBuffers()->end());
      if (absl::c_none_of(result.Result().buffers(), [&](auto index_addr_pair) {
            return index_addr_pair.second.IsSameAs(
                it->second.AsDeviceMemoryBase());
          })) {
        // Supports only one cross-program prefetch address.
        cross_program_prefetch_addr = it->second.AsDeviceMemoryBase();
      }
    }
  }

  TF_RETURN_IF_ERROR(LoadProgramAndEnqueueToStream(
      *run_options, memory_bases, result.Result().root_buffer(),
      cross_program_prefetch_addr));
  return std::move(result);
}

Status TpuExecutable::LoadProgramAndEnqueueToStream(
    const ServiceExecutableRunOptions& run_options,
    absl::Span<const se::DeviceMemoryBase> arguments,
    se::DeviceMemoryBase result,
    absl::optional<se::DeviceMemoryBase> cross_program_prefetch_addr) {
  SE_DeviceMemoryBase* arguments_bases = nullptr;
  if (!arguments.empty()) {
    arguments_bases = new SE_DeviceMemoryBase[arguments.size()];
    for (int i = 0; i < arguments.size(); i++) {
      arguments_bases[i] =
          SE_DeviceMemoryBase{const_cast<void*>(arguments[i].opaque()),
                              arguments[i].size(), arguments[i].payload()};
    }
  }

  SE_DeviceMemoryBase result_base{result.opaque(), result.size(),
                                  result.payload()};
  SE_DeviceMemoryBase prefetch_base;
  if (cross_program_prefetch_addr.has_value()) {
    prefetch_base = SE_DeviceMemoryBase{cross_program_prefetch_addr->opaque(),
                                        cross_program_prefetch_addr->size(),
                                        cross_program_prefetch_addr->payload()};
  }
  int32 rng_seed = run_options.run_options().rng_seed();

  XLA_DeviceAssignment c_dev_assign{/*bytes=*/nullptr, /*size=*/0};
  auto dev_assign = run_options.run_options().device_assignment();
  stream_executor::tpu::SerializedProto dev_assign_serialized;
  if (dev_assign != nullptr) {
    xla::DeviceAssignmentProto dev_assign_proto;
    TF_RETURN_IF_ERROR(dev_assign->Serialize(&dev_assign_proto));
    dev_assign_serialized =
        stream_executor::tpu::SerializeProto(dev_assign_proto);
    c_dev_assign.bytes = dev_assign_serialized.bytes;
    c_dev_assign.size = dev_assign_serialized.size;
  }

  auto stream =
      tensorflow::TpuPlatform::GetRegisteredPlatform()->stream_map()->at(
          run_options.run_options().stream()->implementation());
  StatusHelper status;

  TpuExecutable_LoadProgramAndEnqueueToStream(
      core_program_, arguments_bases, arguments.size(), &result_base,
      (cross_program_prefetch_addr.has_value() ? &prefetch_base : nullptr),
      rng_seed, &c_dev_assign, stream, status.c_status);

  if (dev_assign != nullptr) {
    stream_executor::tpu::SerializedProto_Free(dev_assign_serialized);
  }
  delete[] arguments_bases;
  return status.status();
}

}  // namespace tpu
}  // namespace xla
