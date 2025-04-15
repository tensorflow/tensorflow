/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/stream_executor/tpu/tpu_executable_interface.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/layout_util.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

// Write the tuple index buffers (arrays of pointers).
static absl::Status PopulateResultTupleBuffers(const ShapedBuffer& result,
                                               se::Stream* stream,
                                               se::Stream* transfer_stream) {
  TF_ASSIGN_OR_RETURN(
      auto transfer_manager,
      TransferManager::GetForPlatform(stream->parent()->GetPlatform()));
  if (transfer_manager->CanShapedBufferBeAccessedNow(stream->parent(),
                                                     result)) {
    TF_RETURN_IF_ERROR(transfer_manager->WriteTupleIndexTablesAsync(
        transfer_stream ? transfer_stream : stream, result));
    if (transfer_stream && transfer_stream != stream) {
      TF_RETURN_IF_ERROR(stream->WaitFor(transfer_stream));
    }
    return absl::OkStatus();
  } else {
    return transfer_manager->WriteTupleIndexTablesAsync(stream, result);
  }
}

}  // namespace

absl::StatusOr<ExecutionOutput>
TpuExecutableInterface::AllocateOutputMemoryWithInputReuse(
    const Shape& shape, const HloInputOutputAliasConfig& alias_config,
    se::DeviceMemoryAllocator* allocator,
    std::vector<ExecutionInput>* arguments, se::Stream* stream,
    se::Stream* transfer_stream) {
  auto stream_exec = stream->parent();
  auto platform = stream_exec->GetPlatform();
  TF_ASSIGN_OR_RETURN(auto transfer_manager,
                      TransferManager::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));
  auto shape_size_fn = compiler->ShapeSizeBytesFunction();
  auto device_ordinal = stream_exec->device_ordinal();
  VLOG(3) << "AllocateOutputMemoryWithInputReuse, device = " << device_ordinal
          << " shape = " << ShapeUtil::HumanStringWithLayout(shape);
  auto update_layout = [&transfer_manager](xla::Shape* subshape,
                                           const xla::ShapeIndex& index) {
    if (subshape->IsArray()) {
      CHECK(subshape->has_layout());
      if (!subshape->layout().tiles().empty()) {
        // Already in device shape.
        return;
      }
      *subshape = transfer_manager->HostShapeToDeviceShape(*subshape);
    }
  };
  Shape device_shape = shape;
  xla::ShapeUtil::ForEachMutableSubshape(&device_shape, update_layout);

  TF_RETURN_IF_ERROR(alias_config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          std::optional<HloInputOutputAliasConfig::Alias> alias)
          -> absl::Status {
        if (alias && alias->must_alias()) {
          VLOG(1) << alias->ToString();
          const MaybeOwningDeviceMemory& original_input =
              (*arguments)[alias->parameter_number].Buffers().element(
                  alias->parameter_index);
          if (!original_input.HasOwnership()) {
            return InvalidArgument(
                "An input was configured to be must-alias at "
                "compile time but not donated at runtime: %s",
                alias->ToString());
          }
        }
        return absl::OkStatus();
      }));

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "AllocateOutputMemoryWithInputReuse, device = " << device_ordinal
            << " shape = " << ShapeUtil::HumanStringWithLayout(shape);
    if (!Shape::Equal().MinorToMajorOnlyInLayout()(shape, device_shape)) {
      VLOG(3) << "Rewrote shape to device_shape: "
              << ShapeUtil::HumanStringWithLayout(shape) << " -> "
              << ShapeUtil::HumanStringWithLayout(device_shape);
    }
  }

  ExecutionOutput result(std::move(device_shape), allocator, device_ordinal);
  // Iterate through and allocate a buffer for each shape index, checking for
  // possible input buffer reuse.
  int64_t reused_buffer_bytes = 0;
  int64_t total_result_buffer_bytes = 0;
  for (auto& pair : result.MutableResult()->buffers()) {
    const ShapeIndex& result_index = pair.first;
    se::DeviceMemoryBase& result_buffer = pair.second;
    int64_t allocation_bytes = shape_size_fn(ShapeUtil::GetSubshape(
        result.Result().on_device_shape(), result_index));
    total_result_buffer_bytes += allocation_bytes;

    // Return an InternalError if result_index is invalid. This avoids failing
    // the CHECK when calling GetAliasedParameter
    if (!ShapeUtil::IndexIsValid(alias_config.shape(), result_index)) {
      return Internal("result_index is invalid: %s", result_index.ToString());
    }

    std::optional<HloInputOutputAliasConfig::Alias> alias =
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
      } else {
        VLOG(2) << "An input was not reused since it is not donated "
                << alias->ToString();
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
                              LayoutUtil::MemorySpace(on_device_subshape)));
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

absl::StatusOr<ExecutionOutput> TpuExecutableInterface::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
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

  // Address of the buffer in TPU memory that is being speculated.
  std::vector<se::DeviceMemoryBase> cross_program_prefetch_addrs;
  std::vector<uint32_t> cross_program_prefetch_offsets;
  if (hlo_module_) {
    for (const auto& [parameter, index, offset] :
         hlo_module_->CrossProgramPrefetches()) {
      CHECK_LT(parameter, arguments.size());
      // Ensure the cross program prefetched buffer doesn't alias with any
      // program outputs. If the input and output aliased, the buffer could be
      // invalidated during program execution and the program could read stale
      // data from fast memory instead of fresh data in large memory.
      auto it = arguments[parameter].MutableBuffers()->find({index});
      CHECK(it != arguments[parameter].MutableBuffers()->end());
      CHECK(!it->second.AsDeviceMemoryBase().is_null());
      CHECK(offset);
      bool is_prefetch_output_alias =
          absl::c_any_of(result.Result().buffers(), [&](auto index_addr_pair) {
            return index_addr_pair.second.IsSameAs(
                it->second.AsDeviceMemoryBase());
          });
      cross_program_prefetch_addrs.emplace_back(
          is_prefetch_output_alias ? stream_executor::DeviceMemoryBase()
                                   : it->second.AsDeviceMemoryBase());
      cross_program_prefetch_offsets.emplace_back(
          is_prefetch_output_alias ? std::numeric_limits<uint32_t>::max()
                                   : *offset);
    }
  }

  // MarkToBeReleasedArguments may std::move some elements of arguments, so it
  // must run after the cross program prefetch address is calculated from the
  // arguments.
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);

  TF_RETURN_IF_ERROR(LoadProgramAndEnqueueToStream(
      *run_options, memory_bases, result.Result().root_buffer(),
      cross_program_prefetch_addrs, cross_program_prefetch_offsets));
  return std::move(result);
}

}  // namespace xla
