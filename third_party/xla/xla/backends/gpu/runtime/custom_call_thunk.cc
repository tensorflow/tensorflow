/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/custom_call_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/debugging/symbolize.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/record_ffi.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/ffi/record_ffi.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/primitive_util.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

std::string GetSymbolName(const void* ptr) {
  char buf[512];
  if (absl::Symbolize(ptr, buf, sizeof(buf))) {
    return std::string(buf);
  }
  return "unknown";
}

struct CustomCallRecordState : public CommandState {
  std::vector<const XLA_FFI_Command*> commands;
};

}  // namespace

using xla::ffi::CallFrame;
using xla::ffi::CallFrameBuilder;
using xla::ffi::InvokeContext;

// Builds a call frame prototype for typed-FFI custom calls with dummy device
// memory addresses. This is called once when creating the CustomCall thunk,
// then the thunk will need to update the addresses at runtime.
static absl::StatusOr<ffi::CallFrame> BuildCallFramePrototype(
    absl::Span<const NullableShapedSlice> operands,
    absl::Span<const NullableShapedSlice> results,
    ffi::AttributesMap attributes) {
  CallFrameBuilder builder(
      /*num_args=*/operands.size(),
      /*num_rets=*/results.size());

  for (int i = 0; i < operands.size(); ++i) {
    auto& operand = operands[i];

    if (!operand.has_value()) {
      builder.AddTokenArg();
      continue;
    }

    auto elements = absl::c_accumulate(operand->shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(operand->shape.element_type());
    se::DeviceAddressBase placeholder_arg(nullptr, elements * dtype_bytes);
    builder.AddBufferArg(placeholder_arg, operand->shape.element_type(),
                         operand->shape.dimensions());
  }

  for (int i = 0; i < results.size(); ++i) {
    auto& result = results[i];

    if (!result.has_value()) {
      builder.AddTokenRet();
      continue;
    }

    auto elements = absl::c_accumulate(result->shape.dimensions(), 1ULL,
                                       std::multiplies<int64_t>());
    auto dtype_bytes = primitive_util::ByteWidth(result->shape.element_type());
    se::DeviceAddressBase placeholder_ret(nullptr, elements * dtype_bytes);
    builder.AddBufferRet(placeholder_ret, result->shape.element_type(),
                         result->shape.dimensions());
  }

  if (!attributes.empty()) {
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(std::move(attributes));
    builder.AddAttributes(attrs.Build());
  }

  return builder.Build();
}

static InvokeContext BuildInstantiateInvokeContext(
    ffi::ExecutionState* execution_state,
    const se::GpuComputeCapability* gpu_compute_capability,
    const xla::cpu::TargetMachineOptions* cpu_target_machine_options) {
  InvokeContext context{};
  context.state_context = {execution_state};
  context.backend_context = InvokeContext::GpuContext{
      /*.stream=*/nullptr,
      /*.allocator=*/nullptr,
      /*.collective_params=*/nullptr,
      /*.collective_clique_requests=*/nullptr,
      /*.collective_memory_requests=*/nullptr,
      /*.collective_cliques=*/nullptr,
      /*.collective_memory=*/nullptr,
      /*.gpu_target_config=*/gpu_compute_capability,
      /*.cpu_target_machine_options=*/cpu_target_machine_options,
  };
  return context;
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, ffi::AttributesMap attributes,
    const HloComputation* called_computation, absl::string_view platform_name,
    const se::GpuComputeCapability& gpu_compute_capability,
    std::unique_ptr<ffi::ExecutionState> execution_state,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
    bool use_pdl) {
  ABSL_ASSIGN_OR_RETURN(ffi::HandlerRegistration registration,
                   ffi::FindHandler(target_name, platform_name));

  return Create(thunk_info, std::move(target_name),
                std::move(registration.bundle), std::move(operands),
                std::move(results), std::move(attributes), called_computation,
                gpu_compute_capability, std::move(execution_state),
                std::move(cpu_target_machine_options), use_pdl);
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name,
    XLA_FFI_Handler_Bundle bundle, std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, ffi::AttributesMap attributes,
    const HloComputation* called_computation,
    const se::GpuComputeCapability& gpu_compute_capability,
    std::unique_ptr<ffi::ExecutionState> execution_state,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
    bool use_pdl) {
  VLOG(1) << "CustomCallThunk::Create called for: " << target_name
          << ", execution_state is "
          << (execution_state == nullptr ? "NULL" : "NON-NULL")
          << ", bundle.instantiate is "
          << (bundle.instantiate == nullptr ? "NULL" : "NON-NULL");
  // Initialize FFI handler state if it has an instantiate callback.
  if (execution_state == nullptr) {
    execution_state = std::make_unique<ffi::ExecutionState>();
    if (bundle.instantiate) {
      // Build a call frame with placeholder buffers so the instantiate handler
      // can read operand/result types and shapes. Data pointers are nullptr.
      ABSL_ASSIGN_OR_RETURN(CallFrame call_frame,
                       BuildCallFramePrototype(operands, results, attributes));

      if (!cpu_target_machine_options.has_value()) {
        cpu_target_machine_options = xla::cpu::TargetMachineOptions();
      }
      InvokeContext call_options = BuildInstantiateInvokeContext(
          execution_state.get(), &gpu_compute_capability,
          &*cpu_target_machine_options);
      ABSL_RETURN_IF_ERROR(Invoke(ffi::GetXlaFfiApi(), bundle.instantiate,
                             call_frame, call_options,
                             XLA_FFI_ExecutionStage_INSTANTIATE));
    }
  }

  ABSL_ASSIGN_OR_RETURN(CallFrame call_frame,
                   BuildCallFramePrototype(operands, results, attributes));
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(bundle),
      std::move(operands), std::move(results), std::move(call_frame),
      std::move(attributes), std::move(execution_state), called_computation,
      cpu_target_machine_options, use_pdl));
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::Create(
    ThunkInfo thunk_info, std::string target_name, OwnedHandlerBundle bundle,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results,
    xla::ffi::AttributesMap attributes,
    const HloComputation* called_computation,
    const se::GpuComputeCapability& gpu_compute_capability,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options) {
  if (!bundle.execute) {
    return absl::InvalidArgumentError(
        "Execute handler is required for a CustomCallThunk");
  }

  auto execution_state = std::make_unique<ffi::ExecutionState>();

  if (bundle.instantiate) {
    // Build a call frame with placeholder buffers so the instantiate handler
    // can read operand/result types and shapes. Data pointers are nullptr.
    ABSL_ASSIGN_OR_RETURN(CallFrame call_frame,
                     BuildCallFramePrototype(operands, results, attributes));

    if (!cpu_target_machine_options.has_value()) {
      cpu_target_machine_options = xla::cpu::TargetMachineOptions();
    }
    InvokeContext context = BuildInstantiateInvokeContext(
        execution_state.get(), &gpu_compute_capability,
        &*cpu_target_machine_options);
    ABSL_RETURN_IF_ERROR(Invoke(ffi::GetXlaFfiApi(), *bundle.instantiate, call_frame,
                           context, xla::ffi::ExecutionStage::kInstantiate));
  }

  ABSL_ASSIGN_OR_RETURN(CallFrame call_frame,
                   BuildCallFramePrototype(operands, results, attributes));
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(bundle),
      std::move(operands), std::move(results), std::move(call_frame),
      std::move(attributes), std::move(execution_state), called_computation,
      cpu_target_machine_options));
}

CustomCallThunk::CustomCallThunk(
    ThunkInfo thunk_info, std::string target_name,
    std::variant<XLA_FFI_Handler_Bundle, OwnedHandlerBundle> bundle,
    std::vector<NullableShapedSlice> operands,
    std::vector<NullableShapedSlice> results, CallFrame call_frame,
    ffi::AttributesMap attributes,
    std::unique_ptr<ffi::ExecutionState> execution_state,
    const HloComputation* called_computation,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options,
    bool use_pdl)
    : TracedCommand(Thunk::kCustomCall, thunk_info),
      target_name_(std::move(target_name)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      bundle_(std::move(bundle)),
      use_pdl_(use_pdl),
      attributes_(std::move(attributes)),
      call_frame_(std::move(call_frame)),
      call_frames_([this] { return call_frame_->Copy(); }),
      execution_state_(std::move(execution_state)),
      called_computation_(called_computation),
      cpu_target_machine_options_(std::move(cpu_target_machine_options)) {}

absl::StatusOr<ObjectPool<CallFrame>::BorrowedObject>
CustomCallThunk::BuildCallFrame(
    const BufferAllocations* absl_nullable buffer_allocations) {
  auto device_memory = [&](BufferAllocation::Slice slice) {
    return buffer_allocations ? buffer_allocations->GetDeviceAddress(slice)
                              : se::DeviceAddressBase{};
  };

  absl::InlinedVector<se::DeviceAddressBase, 8> arguments;
  arguments.reserve(operands_.size());
  for (auto& operand : operands_) {
    if (!operand.has_value()) {
      arguments.push_back(se::DeviceAddressBase{});
    } else {
      arguments.push_back(device_memory(operand->slice));
    }
  }

  absl::InlinedVector<se::DeviceAddressBase, 4> results;
  results.reserve(results_.size());
  for (auto& result : results_) {
    if (!result.has_value()) {
      results.push_back(se::DeviceAddressBase{});
    } else {
      results.push_back(device_memory(result->slice));
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  ABSL_RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));
  return call_frame;
}

InvokeContext CustomCallThunk::BuildInvokeContext(
    RunId run_id, se::Stream* absl_nullable stream,
    Thunk::ExecutionScopedState* absl_nullable execution_scoped_state,
    const BufferAllocations* absl_nullable buffer_allocations,
    const CollectiveParams* absl_nullable collective_params,
    CollectiveCliqueRequests* absl_nullable collective_clique_requests,
    CollectiveMemoryRequests* absl_nullable collective_memory_requests,
    const CollectiveCliques* absl_nullable collective_cliques,
    const CollectiveMemory* absl_nullable collective_memory,
    const ffi::ExecutionContext* absl_nullable execution_context,
    absl::Span<se::Stream* const> computation_streams) {
  int32_t device_ordinal = -1;
  se::DeviceAddressAllocator* allocator = nullptr;
  if (buffer_allocations != nullptr) {
    device_ordinal = buffer_allocations->device_ordinal();
    allocator = buffer_allocations->memory_allocator();
  }

  const se::GpuComputeCapability* gpu_compute_capability = nullptr;
  if (stream != nullptr) {
    gpu_compute_capability =
        &stream->parent()->GetDeviceDescription().gpu_compute_capability();
  }

  ffi::ExecutionState* prepare_state = nullptr;
  ffi::ExecutionState* initialize_state = nullptr;

  if (execution_scoped_state) {
    auto [it, _] = execution_scoped_state->try_emplace(
        this->thunk_info().thunk_id, std::in_place_type<PrepareAndInitState>);
    PrepareAndInitState& prepare_and_init =
        tsl::any_cast<PrepareAndInitState>(it->second);
    prepare_state = &prepare_and_init.prepare;
    initialize_state = &prepare_and_init.init;
  }

  // `called_computation_` is forwarded to the FFI handler both for direct
  // ExecuteOnStream and for TracedCommand::Record (which traces ExecuteOnStream
  // onto a command-buffer trace stream). The old CustomCallCmd path hard-coded
  // nullptr here with a TODO(b/342285364); this unified path resolves that
  // TODO so handlers see the real called computation under command buffers.
  return InvokeContext{
      run_id,
      device_ordinal,
      InvokeContext::GpuContext{
          stream, allocator, collective_params, collective_clique_requests,
          collective_memory_requests, collective_cliques, collective_memory,
          gpu_compute_capability,
          cpu_target_machine_options_ ? &*cpu_target_machine_options_ : nullptr,
          computation_streams,
          collective_params ? absl::MakeSpan(collective_params->async_streams)
                            : absl::Span<se::Stream* const>()},
      InvokeContext::StateContext{execution_state_.get(), prepare_state,
                                  initialize_state},
      called_computation_,
      execution_context};
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    RunId run_id, XLA_FFI_Handler* handler, XLA_FFI_ExecutionStage stage,
    se::Stream* stream, Thunk::ExecutionScopedState* execution_scoped_state,
    const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations,
    const CollectiveParams* absl_nullable collective_params,
    CollectiveCliqueRequests* absl_nullable collective_clique_requests,
    CollectiveMemoryRequests* absl_nullable collective_memory_requests,
    const CollectiveCliques* absl_nullable collective_cliques,
    const CollectiveMemory* absl_nullable collective_memory,
    absl::Span<se::Stream* const> computation_streams,
    const XLA_FFI_Extension* extension_start) {
  if (handler == nullptr) {
    return absl::InternalError("FFI execute handler is not set");
  }
  if (stage != XLA_FFI_ExecutionStage_PREPARE &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  ABSL_ASSIGN_OR_RETURN(auto call_frame, BuildCallFrame(buffer_allocations));
  InvokeContext context = BuildInvokeContext(
      run_id, stream, execution_scoped_state, buffer_allocations,
      collective_params, collective_clique_requests, collective_memory_requests,
      collective_cliques, collective_memory, execution_context,
      computation_streams);
  context.extension_start = extension_start;
  return Invoke(ffi::GetXlaFfiApi(), handler, *call_frame, context, stage);
}

absl::Status CustomCallThunk::ExecuteFfiHandler(
    RunId run_id, xla::ffi::Ffi& handler, xla::ffi::ExecutionStage stage,
    se::Stream* stream, Thunk::ExecutionScopedState* execution_scoped_state,
    const ffi::ExecutionContext* execution_context,
    const BufferAllocations* buffer_allocations,
    const CollectiveParams* absl_nullable collective_params,
    CollectiveCliqueRequests* absl_nullable collective_clique_requests,
    CollectiveMemoryRequests* absl_nullable collective_memory_requests,
    const CollectiveCliques* absl_nullable collective_cliques,
    const CollectiveMemory* absl_nullable collective_memory,
    absl::Span<se::Stream* const> computation_streams,
    const XLA_FFI_Extension* extension_start) {
  if (stage != xla::ffi::ExecutionStage::kPrepare &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  ABSL_ASSIGN_OR_RETURN(auto call_frame, BuildCallFrame(buffer_allocations));
  InvokeContext context = BuildInvokeContext(
      run_id, stream, execution_scoped_state, buffer_allocations,
      collective_params, collective_clique_requests, collective_memory_requests,
      collective_cliques, collective_memory, execution_context,
      computation_streams);
  context.extension_start = extension_start;
  return Invoke(ffi::GetXlaFfiApi(), handler, *call_frame, context, stage);
}

absl::Status CustomCallThunk::Prepare(const PrepareParams& params) {
  // Run the prepare stage at most once per execution. When lowered into a
  // command buffer, CommandBufferThunk drives this same object via both its
  // command sequence and its fallback thunk sequence; see PrepareAndInitState.
  if (params.execution_scoped_state != nullptr) {
    auto [it, _] = params.execution_scoped_state->try_emplace(
        thunk_info().thunk_id, std::in_place_type<PrepareAndInitState>);
    PrepareAndInitState& state = tsl::any_cast<PrepareAndInitState>(it->second);
    if (state.prepared) {
      return absl::OkStatus();
    }
    state.prepared = true;
  }

  const RunId run_id =
      params.collective_params ? params.collective_params->run_id : RunId{-1};

  if (const auto* c_bundle = std::get_if<XLA_FFI_Handler_Bundle>(&bundle_);
      c_bundle && c_bundle->prepare) {
    return ExecuteFfiHandler(
        run_id, c_bundle->prepare, XLA_FFI_ExecutionStage_PREPARE,
        /*stream=*/nullptr,
        /*execution_scoped_state=*/params.execution_scoped_state,
        /*execution_context=*/nullptr,
        /*buffer_allocations=*/params.buffer_allocations,
        /*collective_params=*/params.collective_params,
        /*collective_clique_requests=*/params.collective_clique_requests,
        /*collective_memory_requests=*/params.collective_memory_requests,
        /*collective_cliques=*/nullptr,
        /*collective_memory=*/nullptr,
        /*computation_streams=*/{});
  }
  if (const auto* owned_bundle = std::get_if<OwnedHandlerBundle>(&bundle_);
      owned_bundle && owned_bundle->prepare) {
    return ExecuteFfiHandler(
        run_id, *owned_bundle->prepare, xla::ffi::ExecutionStage::kPrepare,
        /*stream=*/nullptr,
        /*execution_scoped_state=*/params.execution_scoped_state,
        /*execution_context=*/nullptr,
        /*buffer_allocations=*/params.buffer_allocations,
        /*collective_params=*/params.collective_params,
        /*collective_clique_requests=*/params.collective_clique_requests,
        /*collective_memory_requests=*/params.collective_memory_requests,
        /*collective_cliques=*/nullptr,
        /*collective_memory=*/nullptr,
        /*computation_streams=*/{});
  }

  return absl::OkStatus();
}

absl::Status CustomCallThunk::Initialize(const InitializeParams& params) {
  // Run the initialize stage at most once per execution. When lowered into a
  // command buffer, CommandBufferThunk drives this same object via both its
  // command sequence and its fallback thunk sequence; see PrepareAndInitState.
  if (params.execution_scoped_state != nullptr) {
    auto [it, _] = params.execution_scoped_state->try_emplace(
        thunk_info().thunk_id, std::in_place_type<PrepareAndInitState>);
    PrepareAndInitState& state = tsl::any_cast<PrepareAndInitState>(it->second);
    if (state.initialized) {
      return absl::OkStatus();
    }
    state.initialized = true;
  }

  const RunId run_id =
      params.collective_params ? params.collective_params->run_id : RunId{-1};

  if (const auto* c_bundle = std::get_if<XLA_FFI_Handler_Bundle>(&bundle_);
      c_bundle && c_bundle->initialize) {
    return ExecuteFfiHandler(
        run_id, *c_bundle->initialize, XLA_FFI_ExecutionStage_INITIALIZE,
        params.stream, params.execution_scoped_state,
        params.ffi_execution_context, params.buffer_allocations,
        params.collective_params,
        /*collective_clique_requests=*/nullptr,
        /*collective_memory_requests=*/nullptr, params.collective_cliques,
        params.collective_memory,
        /*computation_streams=*/{});
  }
  if (const auto* owned_bundle = std::get_if<OwnedHandlerBundle>(&bundle_);
      owned_bundle && owned_bundle->initialize) {
    return ExecuteFfiHandler(
        run_id, *owned_bundle->initialize,
        xla::ffi::ExecutionStage::kInitialize, params.stream,
        params.execution_scoped_state, params.ffi_execution_context,
        params.buffer_allocations, params.collective_params,
        /*collective_clique_requests=*/nullptr,
        /*collective_memory_requests=*/nullptr, params.collective_cliques,
        params.collective_memory,
        /*computation_streams=*/{});
  }
  return absl::OkStatus();
}

absl::Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;

  const RunId run_id =
      params.collective_params ? params.collective_params->run_id : RunId{-1};

  if (const auto* c_bundle = std::get_if<XLA_FFI_Handler_Bundle>(&bundle_)) {
    return ExecuteFfiHandler(
        run_id, c_bundle->execute, XLA_FFI_ExecutionStage_EXECUTE, stream,
        params.execution_scoped_state, params.ffi_execution_context,
        params.buffer_allocations, params.collective_params,
        /*collective_clique_requests=*/nullptr,
        /*collective_memory_requests=*/nullptr, params.collective_cliques,
        params.collective_memory, params.additional_compute_streams);
  }
  if (const auto* owned_bundle = std::get_if<OwnedHandlerBundle>(&bundle_)) {
    if (!owned_bundle->execute) {
      return absl::InternalError("FFI execute handler is not set");
    }
    return ExecuteFfiHandler(
        run_id, *owned_bundle->execute, xla::ffi::ExecutionStage::kExecute,
        stream, params.execution_scoped_state, params.ffi_execution_context,
        params.buffer_allocations, params.collective_params,
        /*collective_clique_requests=*/nullptr,
        /*collective_memory_requests=*/nullptr, params.collective_cliques,
        params.collective_memory, params.additional_compute_streams);
  }

  return absl::InternalError("No FFI handler bundle set");
}

absl::StatusOr<const se::CommandBuffer::Command*> CustomCallThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::StreamExecutor* executor = execute_params.stream->parent();
  ABSL_ASSIGN_OR_RETURN(auto call_frame,
                   BuildCallFrame(execute_params.buffer_allocations));

  // Retrieve or create the state for recording.
  auto* state = record_params.state.GetOrCreate<CustomCallRecordState>(
      this, command_buffer);

  constexpr int64_t kMaxCommands = 16;
  const XLA_FFI_Command* commands_storage[kMaxCommands] = {nullptr};
  int64_t num_commands = 0;

  const bool is_record_create =
      std::holds_alternative<RecordCreate>(record_action);
  const bool is_record_update =
      std::holds_alternative<RecordUpdate>(record_action);
  if (is_record_update) {  // Copy over commands from state to inline storage.
    TF_RET_CHECK(state->commands.size() <= kMaxCommands)
        << "Too many commands to fit in inline storage";
    std::copy(state->commands.begin(), state->commands.end(), commands_storage);
    num_commands = state->commands.size();
  }

  XLA_FFI_RecordAction action_to_pass = is_record_create
                                            ? XLA_FFI_RecordAction_Create
                                            : XLA_FFI_RecordAction_Update;

  // Record directly into the primary command buffer.
  std::vector<const se::CommandBuffer::Command*> deps;
  if (is_record_create) {
    auto record_create = std::get<RecordCreate>(record_action);
    deps.assign(record_create.dependencies.begin(),
                record_create.dependencies.end());
  }
  XLA_FFI_RecordContext record_ctx = {command_buffer, executor, std::move(deps),
                                      use_pdl_};

  XLA_FFI_RecordFrame record_frame{&record_ctx,    GetXlaFfiRecordApi(),
                                   action_to_pass, commands_storage,
                                   &num_commands,  kMaxCommands};
  ffi::RecordExtension::CExtension record_extension =
      ffi::BuildRecordCExtension(&record_frame);

  const RunId run_id = execute_params.collective_params
                           ? execute_params.collective_params->run_id
                           : RunId{-1};

  // Invoke FFI handler
  bool attempted_record = false;
  auto const execute_handler = [&](auto& handler, auto stage) {
    return ExecuteFfiHandler(
        run_id, handler, stage, execute_params.stream,
        /*execution_scoped_state=*/nullptr,
        execute_params.ffi_execution_context, execute_params.buffer_allocations,
        execute_params.collective_params,
        /*collective_clique_requests=*/nullptr,
        /*collective_memory_requests=*/nullptr,
        execute_params.collective_cliques, execute_params.collective_memory,
        execute_params.additional_compute_streams,
        &record_extension.extension_base);
  };
  absl::Status status = absl::OkStatus();
  if (const auto* c_bundle = std::get_if<XLA_FFI_Handler_Bundle>(&bundle_);
      c_bundle && c_bundle->record != nullptr) {
    attempted_record = true;
    VLOG(5) << "CustomCallThunk::Record: c_bundle->record symbol: "
            << GetSymbolName(reinterpret_cast<const void*>(c_bundle->record));
    status = execute_handler(c_bundle->record, XLA_FFI_ExecutionStage_RECORD);
  } else if (const auto* owned_bundle =
                 std::get_if<OwnedHandlerBundle>(&bundle_)) {
    if (owned_bundle->record) {
      attempted_record = true;
      status = execute_handler(*owned_bundle->record,
                               xla::ffi::ExecutionStage::kRecord);
    }
  }
  ABSL_RETURN_IF_ERROR(status);

  // Fallback to tracing if requested or if no record handler was present
  if (!attempted_record || record_ctx.stream_capture_requested) {
    VLOG(3) << "FFI handler requested or required fallback to stream capture.";
    TF_RET_CHECK(num_commands == 0)
        << "Stream capture requested but commands were recorded.";
    return TracedCommand::Record(execute_params, record_params, record_action,
                                 command_buffer);
  }

  // Save newly recorded commands to state if this is the Create action
  // Must be done after returning from the FFI handler.
  if (is_record_create) {
    state->commands.assign(commands_storage, commands_storage + num_commands);
  }

  // Return the last command in the chain for dependency tracking.
  // If more than one command was recorded, and they are independent, a dummy
  // node must be added to the command graph by the FFI client so that XLA
  // can track a single dependency for the entire chain.
  if (num_commands > 0 && commands_storage[num_commands - 1] != nullptr) {
    return reinterpret_cast<const se::CommandBuffer::Command*>(
        commands_storage[num_commands - 1]);
  }
  // No commands were recorded.
  return nullptr;
}

absl::StatusOr<ThunkProto> CustomCallThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  proto.mutable_custom_call_thunk()->set_target_name(target_name_);
  proto.mutable_custom_call_thunk()->set_api_version(
      CustomCallApiVersion::API_VERSION_TYPED_FFI);
  if (called_computation_ != nullptr) {
    proto.mutable_custom_call_thunk()->set_called_computation(
        called_computation_->name());
  }

  for (const NullableShapedSlice& operand : operands_) {
    ABSL_ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_operands(),
                     operand.ToProto());
  }

  for (const NullableShapedSlice& result : results_) {
    ABSL_ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_results(),
                     result.ToProto());
  }

  if (attributes_.has_value()) {
    *proto.mutable_custom_call_thunk()->mutable_attributes() =
        attributes_->ToProto();
  }

  if (execution_state_ && execution_state_->IsSerializable()) {
    ABSL_ASSIGN_OR_RETURN(
        *proto.mutable_custom_call_thunk()->mutable_execution_state(),
        execution_state_->ToProto());
  }
  proto.mutable_custom_call_thunk()->set_use_pdl(use_pdl_);
  return proto;
}

absl::StatusOr<std::unique_ptr<CustomCallThunk>> CustomCallThunk::FromProto(
    ThunkInfo thunk_info, const CustomCallThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const HloModule* absl_nullable hlo_module, absl::string_view platform_name,
    const se::GpuComputeCapability& gpu_compute_capability,
    std::optional<xla::cpu::TargetMachineOptions> cpu_target_machine_options) {
  if (hlo_module == nullptr && proto.has_called_computation()) {
    return absl::InvalidArgumentError(
        "HloModule is required to deserialize a CustomCallThunk with a "
        "called computation");
  }

  std::vector<NullableShapedSlice> operands, results;
  for (const auto& operand_proto : proto.operands()) {
    ABSL_ASSIGN_OR_RETURN(
        NullableShapedSlice operand,
        NullableShapedSlice::FromProto(operand_proto, buffer_allocations));
    operands.push_back(std::move(operand));
  }
  for (const auto& result_proto : proto.results()) {
    ABSL_ASSIGN_OR_RETURN(
        NullableShapedSlice result,
        NullableShapedSlice::FromProto(result_proto, buffer_allocations));
    results.push_back(std::move(result));
  }

  ABSL_ASSIGN_OR_RETURN(ffi::AttributesMap attributes,
                   ffi::AttributesMap::FromProto(proto.attributes()));

  HloComputation* called_computation = nullptr;
  if (proto.has_called_computation()) {
    CHECK(hlo_module != nullptr);
    called_computation =
        hlo_module->GetComputationWithName(proto.called_computation());
    if (called_computation == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "HloComputation '", proto.called_computation(),
          "' not found in the HloModule with name '", hlo_module->name(), "'"));
    }
  }
  std::unique_ptr<ffi::ExecutionState> execution_state;
  if (proto.has_execution_state()) {
    auto state = ffi::ExecutionState::FromProto(proto.execution_state());
    if (state.ok()) {
      execution_state =
          std::make_unique<ffi::ExecutionState>(std::move(state.value()));
    } else {
      LOG(WARNING)
          << "Failed to deserialize the custom call execution state. Falling "
             "back to runtime instantiation of the execution state. Reason: "
          << state.status();
    }
  }

  return CustomCallThunk::Create(
      std::move(thunk_info), proto.target_name(), std::move(operands),
      std::move(results), std::move(attributes), called_computation,
      platform_name, gpu_compute_capability, std::move(execution_state),
      std::move(cpu_target_machine_options), proto.use_pdl());
}

}  // namespace xla::gpu
