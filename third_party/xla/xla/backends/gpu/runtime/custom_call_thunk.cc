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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/debugging/symbolize.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/ffi/invoke.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/primitive_util.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_slice.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"

namespace se = ::stream_executor;

struct XLA_FFI_RecordContext {
  se::CommandBuffer* command_buffer;
  se::StreamExecutor* executor;
  std::vector<const se::CommandBuffer::Command*> dependencies;
  bool use_pdl = false;
  bool stream_capture_requested = false;
};

namespace xla::gpu {
namespace {

template <typename T, std::size_t kAlignment>
struct alignas(kAlignment) AlignedWrapper {
  T value;

  template <typename... Args>
  explicit AlignedWrapper(Args&&... args)
      : value(std::forward<Args>(args)...) {}

  explicit operator T&() { return value; }
  explicit operator const T&() { return value; }
};

class CustomKernelArgs : public se::KernelArgsPackedArrayBase {
  static constexpr size_t kAlignment = 64;

 public:
  explicit CustomKernelArgs(size_t num_args) {
    argument_addresses_.reserve(num_args);
  }

  void AddDeviceAddress(void* opaque) {
    auto ptr = std::make_unique<void*>(opaque);
    argument_addresses_.push_back(ptr.get());
    device_ptrs_.push_back(std::move(ptr));
  }

  void add_argument(const void* data, size_t size) {
    using AlignedStorage = AlignedWrapper<std::byte, kAlignment>;
    const size_t num_blocks =
        (size + sizeof(AlignedStorage) - 1) / sizeof(AlignedStorage);
    auto block_vector =
        std::make_unique<std::vector<AlignedStorage>>(num_blocks);
    std::memcpy(block_vector->data(), data, size);
    argument_addresses_.push_back(block_vector->data());
    allocated_args_.push_back(std::move(block_vector));
  }

  void add_shared_bytes(size_t bytes) { shared_memory_bytes_ += bytes; }

  absl::Span<const void* const> argument_addresses() const override {
    return argument_addresses_;
  }
  size_t number_of_arguments() const override {
    return argument_addresses_.size() + (shared_memory_bytes_ > 0);
  }
  uint64_t number_of_shared_bytes() const override {
    return shared_memory_bytes_;
  }

 private:
  std::vector<std::unique_ptr<void*>> device_ptrs_;
  std::vector<
      std::unique_ptr<std::vector<AlignedWrapper<std::byte, kAlignment>>>>
      allocated_args_;
  std::vector<const void*> argument_addresses_;
  size_t shared_memory_bytes_ = 0;
};

std::string GetSymbolName(const void* ptr) {
  char buf[512];
  if (absl::Symbolize(ptr, buf, sizeof(buf))) {
    return std::string(buf);
  }
  return "unknown";
}

struct FfiLaunchParams {
  LaunchDimensions launch_dimensions;
  std::optional<stream_executor::ClusterDim> cluster_dims;

  se::Kernel* kernel;
  uint32_t shared_mem_bytes;
};

template <typename Sink>
[[maybe_unused]] void AbslStringify(Sink& sink, const FfiLaunchParams& params) {
  absl::Format(&sink,
               "FfiLaunchParams(launch_dimensions=%s, cluster_dims=%s, "
               "kernel=%s, shared_mem_bytes=%u)",
               params.launch_dimensions.ToString(),
               params.cluster_dims.value_or(se::ClusterDim{0, 0, 0}).ToString(),
               GetSymbolName(params.kernel), params.shared_mem_bytes);
}

class FfiKernelCache : public se::CommandBuffer::Resource {
 public:
  absl::StatusOr<se::Kernel*> GetOrCreateKernel(
      se::StreamExecutor* executor, absl::string_view kernel_name_view,
      const void* kernel_data, size_t kernel_size, XLA_FFI_SourceFormat format,
      size_t num_args) {
    std::string kernel_name(kernel_name_view);
    auto key = std::make_pair(kernel_name, kernel_data);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
      return it->second.get();
    }
    bool is_ptx = (format == XLA_FFI_SourceFormat_PTX);
    se::KernelLoaderSpec spec =
        is_ptx
            ? se::KernelLoaderSpec::CreateCudaPtxInMemorySpec(
                  absl::string_view(reinterpret_cast<const char*>(kernel_data),
                                    kernel_size),
                  kernel_name, num_args)
            : se::KernelLoaderSpec::CreateCudaCubinInMemorySpec(
                  absl::Span<const uint8_t>(
                      reinterpret_cast<const uint8_t*>(kernel_data),
                      kernel_size),
                  kernel_name, num_args);

    ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                     executor->LoadKernel(spec));

    se::Kernel* kernel_ptr = kernel.get();
    kernels_[key] = std::move(kernel);
    return kernel_ptr;
  }

  void SaveLaunchParams(const se::CommandBuffer::Command* cmd,
                        FfiLaunchParams params) {
    launch_params_[cmd] = params;
  }

  const FfiLaunchParams* GetLaunchParams(
      const se::CommandBuffer::Command* cmd) const {
    auto it = launch_params_.find(cmd);
    if (it != launch_params_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  FfiLaunchParams* GetMutableLaunchParams(
      const se::CommandBuffer::Command* cmd) {
    auto it = launch_params_.find(cmd);
    if (it != launch_params_.end()) {
      return &it->second;
    }
    return nullptr;
  }

 private:
  absl::flat_hash_map<std::pair<std::string, const void*>,
                      std::unique_ptr<se::Kernel>>
      kernels_;
  absl::flat_hash_map<const se::CommandBuffer::Command*, FfiLaunchParams>
      launch_params_;
};

XLA_FFI_Error* FfiCreateLaunch(
    XLA_FFI_RecordContext* ctx, const char* kernel_name,
    const void* kernel_data, size_t kernel_size, XLA_FFI_SourceFormat format,
    XLA_FFI_LaunchDims launch_dims, uint32_t shared_mem_bytes,
    const XLA_FFI_KernelArgs* args, const XLA_FFI_Command* const* dependencies,
    uint32_t num_dependencies, const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;

  auto* cache = cmd_buffer->GetOrConstructResource<FfiKernelCache>();
  auto kernel_or =
      cache->GetOrCreateKernel(ctx->executor, kernel_name, kernel_data,
                               kernel_size, format, args->num_args);
  if (!kernel_or.ok()) {
    return new XLA_FFI_Error{kernel_or.status()};
  }
  se::Kernel* kernel = *kernel_or;
  kernel->set_use_pdl(ctx->use_pdl);

  CustomKernelArgs packed_args(args->num_args);
  packed_args.add_shared_bytes(shared_mem_bytes);
  for (size_t i = 0; i < args->num_args; ++i) {
    if (args->args != nullptr && args->args[i].size > 0) {
      packed_args.add_argument(args->args[i].arg_address, args->args[i].size);
    } else {
      packed_args.AddDeviceAddress(
          // FFI API is const we need to cast away constness.
          // NOLINTNEXTLINE
          const_cast<void*>(args->args[i].arg_address));
    }
  }

  // Map dependencies
  std::vector<const se::CommandBuffer::Command*> deps;
  if (num_dependencies == 0) {
    deps = ctx->dependencies;
  } else {
    deps.reserve(num_dependencies);
    for (uint32_t i = 0; i < num_dependencies; ++i) {
      deps.push_back(
          reinterpret_cast<const se::CommandBuffer::Command*>(dependencies[i]));
    }
  }

  se::ThreadDim threads(launch_dims.block.x, launch_dims.block.y,
                        launch_dims.block.z);
  se::BlockDim blocks(launch_dims.grid.x, launch_dims.grid.y,
                      launch_dims.grid.z);
  std::optional<se::ClusterDim> cluster_dims = std::nullopt;
  if (launch_dims.cluster != nullptr) {
    cluster_dims = se::ClusterDim(
        launch_dims.cluster->x, launch_dims.cluster->y, launch_dims.cluster->z);
  }

  FfiLaunchParams params{
      /*.launch_dimensions = */ LaunchDimensions(blocks, threads),
      /*.cluster_dims =*/cluster_dims,
      /*.kernel =*/kernel,
      /*.shared_mem_bytes = */ shared_mem_bytes,
  };
  VLOG(3) << "FfiCreateLaunch for kernel: " << kernel_name
          << ", use_pdl from ctx: " << ctx->use_pdl
          << ", num_dependencies passed: " << num_dependencies
          << ", resolved deps size: " << deps.size()
          << ", launch_params: " << params;

  auto status_or_cmd = cmd_buffer->CreateLaunch(threads, blocks, cluster_dims,
                                                *kernel, packed_args, deps);

  if (!status_or_cmd.ok()) {
    return new XLA_FFI_Error{status_or_cmd.status()};
  }

  const se::CommandBuffer::Command* cmd = *status_or_cmd;
  VLOG(3) << "FfiCreateLaunch: created command ptr: " << cmd;
  cache->SaveLaunchParams(cmd, std::move(params));

  *out_command = reinterpret_cast<const XLA_FFI_Command*>(cmd);
  return nullptr;
}

XLA_FFI_Error* FfiUpdateLaunch(XLA_FFI_RecordContext* ctx,
                               const XLA_FFI_Command* command,
                               const XLA_FFI_KernelArgs* args) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  auto* cmd = reinterpret_cast<const se::CommandBuffer::Command*>(command);

  auto* cache = cmd_buffer->GetOrConstructResource<FfiKernelCache>();
  FfiLaunchParams* params = cache->GetMutableLaunchParams(cmd);
  if (params == nullptr) {
    return new XLA_FFI_Error{absl::InternalError(
        "Failed to find cached launch parameters for update")};
  }
  CustomKernelArgs packed_args(args->num_args);
  packed_args.add_shared_bytes(params->shared_mem_bytes);
  for (size_t i = 0; i < args->num_args; ++i) {
    if (args->args != nullptr && args->args[i].size > 0) {
      packed_args.add_argument(args->args[i].arg_address, args->args[i].size);
    } else {
      packed_args.AddDeviceAddress(
          // FFI API is const we need to cast away constness.
          // NOLINTNEXTLINE
          const_cast<void*>(args->args[i].arg_address));
    }
  }

  absl::Status status = cmd_buffer->UpdateLaunch(
      cmd, params->launch_dimensions.thread_counts_per_block(),
      params->launch_dimensions.block_counts(), params->cluster_dims,
      *params->kernel, packed_args);

  if (!status.ok()) {
    return new XLA_FFI_Error{status};
  }
  return nullptr;
}

XLA_FFI_Error* FfiCreateMemcpyD2D(XLA_FFI_RecordContext* ctx, void* dst,
                                  void* src, size_t size,
                                  const XLA_FFI_Command* const* dependencies,
                                  uint32_t num_dependencies,
                                  const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;

  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(
        reinterpret_cast<const se::CommandBuffer::Command*>(dependencies[i]));
  }

  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  auto status_or_cmd =
      cmd_buffer->CreateMemcpyD2D(&dst_addr, src_addr, size, deps);

  if (!status_or_cmd.ok()) {
    return new XLA_FFI_Error{status_or_cmd.status()};
  }

  *out_command = reinterpret_cast<const XLA_FFI_Command*>(*status_or_cmd);
  return nullptr;
}

XLA_FFI_Error* FfiUpdateMemcpyD2D(XLA_FFI_RecordContext* ctx,
                                  const XLA_FFI_Command* command, void* dst,
                                  void* src, size_t size) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  auto* cmd = reinterpret_cast<const se::CommandBuffer::Command*>(command);

  se::DeviceAddressBase dst_addr(dst, size);
  se::DeviceAddressBase src_addr(src, size);
  absl::Status status =
      cmd_buffer->UpdateMemcpyD2D(cmd, &dst_addr, src_addr, size);
  if (!status.ok()) {
    return new XLA_FFI_Error{status};
  }
  return nullptr;
}

XLA_FFI_Error* FfiRequestStreamCapture(XLA_FFI_RecordContext* ctx) {
  ctx->stream_capture_requested = true;
  return nullptr;
}

XLA_FFI_Error* FfiCreateEmptyCommand(XLA_FFI_RecordContext* ctx,
                                     const XLA_FFI_Command* const* dependencies,
                                     uint32_t num_dependencies,
                                     const XLA_FFI_Command** out_command) {
  se::CommandBuffer* cmd_buffer = ctx->command_buffer;
  std::vector<const se::CommandBuffer::Command*> deps;
  deps.reserve(num_dependencies);
  for (uint32_t i = 0; i < num_dependencies; ++i) {
    deps.push_back(
        reinterpret_cast<const se::CommandBuffer::Command*>(dependencies[i]));
  }
  auto status_or_cmd = cmd_buffer->CreateEmptyCmd(deps);
  if (!status_or_cmd.ok()) {
    return new XLA_FFI_Error{status_or_cmd.status()};
  }
  *out_command = reinterpret_cast<const XLA_FFI_Command*>(*status_or_cmd);
  return nullptr;
}

static constexpr XLA_FFI_RecordApi ffi_record_api = {
    /*.create_launch =*/FfiCreateLaunch,
    /*.update_launch =*/FfiUpdateLaunch,
    /*.create_memcpy_d2d =*/FfiCreateMemcpyD2D,
    /*.update_memcpy_d2d =*/FfiUpdateMemcpyD2D,
    /*.request_stream_capture =*/FfiRequestStreamCapture,
    /*.create_empty_command =*/FfiCreateEmptyCommand,
};

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
  ASSIGN_OR_RETURN(ffi::HandlerRegistration registration,
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
    VLOG(1)
        << "CustomCallThunk::Create: execution_state is null, initializing...";
    execution_state = std::make_unique<ffi::ExecutionState>();
    if (bundle.instantiate) {
      VLOG(1) << "CustomCallThunk::Create: calling bundle.instantiate...";
      // Build a call frame with placeholder buffers so the instantiate handler
      // can read operand/result types and shapes. Data pointers are nullptr.
      ASSIGN_OR_RETURN(CallFrame call_frame,
                       BuildCallFramePrototype(operands, results, attributes));

      if (!cpu_target_machine_options.has_value()) {
        cpu_target_machine_options = xla::cpu::TargetMachineOptions();
      }
      InvokeContext call_options = BuildInstantiateInvokeContext(
          execution_state.get(), &gpu_compute_capability,
          &*cpu_target_machine_options);
      RETURN_IF_ERROR(Invoke(ffi::GetXlaFfiApi(), bundle.instantiate,
                             call_frame, call_options,
                             XLA_FFI_ExecutionStage_INSTANTIATE));
    }
  }

  ASSIGN_OR_RETURN(CallFrame call_frame,
                   BuildCallFramePrototype(operands, results, attributes));
  XLA_FFI_Handler* record_handler = bundle.record;
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(bundle), record_handler,
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
  if (!bundle.execute && !bundle.record) {
    return absl::InvalidArgumentError(
        "Execute or Record handler is required for a CustomCallThunk");
  }

  auto execution_state = std::make_unique<ffi::ExecutionState>();

  if (bundle.instantiate) {
    // Build a call frame with placeholder buffers so the instantiate handler
    // can read operand/result types and shapes. Data pointers are nullptr.
    ASSIGN_OR_RETURN(CallFrame call_frame,
                     BuildCallFramePrototype(operands, results, attributes));

    if (!cpu_target_machine_options.has_value()) {
      cpu_target_machine_options = xla::cpu::TargetMachineOptions();
    }
    InvokeContext context = BuildInstantiateInvokeContext(
        execution_state.get(), &gpu_compute_capability,
        &*cpu_target_machine_options);
    RETURN_IF_ERROR(Invoke(ffi::GetXlaFfiApi(), *bundle.instantiate, call_frame,
                           context, xla::ffi::ExecutionStage::kInstantiate));
  }

  ASSIGN_OR_RETURN(CallFrame call_frame,
                   BuildCallFramePrototype(operands, results, attributes));
  return absl::WrapUnique(new CustomCallThunk(
      thunk_info, std::move(target_name), std::move(bundle),
      /*record_handler=*/nullptr, std::move(operands), std::move(results),
      std::move(call_frame), std::move(attributes), std::move(execution_state),
      called_computation, cpu_target_machine_options));
}

CustomCallThunk::CustomCallThunk(
    ThunkInfo thunk_info, std::string target_name,
    std::variant<XLA_FFI_Handler_Bundle, OwnedHandlerBundle> bundle,
    XLA_FFI_Handler* record_handler, std::vector<NullableShapedSlice> operands,
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
      record_handler_(record_handler),
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

  ASSIGN_OR_RETURN(auto call_frame, call_frames_->GetOrCreate());
  RETURN_IF_ERROR(call_frame->UpdateWithBuffers(arguments, results));
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
    absl::AnyInvocable<void(XLA_FFI_CallFrame*) &&> configure_call_frame) {
  if (handler == nullptr) {
    return absl::InternalError("FFI execute handler is not set");
  }
  if (stage != XLA_FFI_ExecutionStage_PREPARE &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  ASSIGN_OR_RETURN(auto call_frame, BuildCallFrame(buffer_allocations));
  InvokeContext context = BuildInvokeContext(
      run_id, stream, execution_scoped_state, buffer_allocations,
      collective_params, collective_clique_requests, collective_memory_requests,
      collective_cliques, collective_memory, execution_context,
      computation_streams);
  return Invoke(ffi::GetXlaFfiApi(), handler, *call_frame, context, stage,
                std::move(configure_call_frame));
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
    absl::AnyInvocable<void(XLA_FFI_CallFrame*) &&> configure_call_frame) {
  if (stage != xla::ffi::ExecutionStage::kPrepare &&
      !(buffer_allocations && stream)) {
    return absl::InternalError("buffer allocations and stream are required");
  }

  ASSIGN_OR_RETURN(auto call_frame, BuildCallFrame(buffer_allocations));
  InvokeContext context = BuildInvokeContext(
      run_id, stream, execution_scoped_state, buffer_allocations,
      collective_params, collective_clique_requests, collective_memory_requests,
      collective_cliques, collective_memory, execution_context,
      computation_streams);
  return Invoke(ffi::GetXlaFfiApi(), handler, *call_frame, context, stage,
                std::move(configure_call_frame));
}

absl::Status CustomCallThunk::Prepare(const PrepareParams& params) {
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
  ASSIGN_OR_RETURN(auto call_frame,
                   BuildCallFrame(execute_params.buffer_allocations));

  // Retrieve or create the state for recording
  auto* state = record_params.state.GetOrNull<CustomCallRecordState>(
      this, command_buffer);
  if (state == nullptr) {
    auto new_state = std::make_unique<CustomCallRecordState>();
    state = record_params.state.GetOrCreate<CustomCallRecordState>(
        this, command_buffer, [&] { return std::move(new_state); });
  }

  constexpr size_t kMaxCommands = 16;
  const XLA_FFI_Command* commands_storage[kMaxCommands] = {nullptr};
  size_t num_commands = 0;

  const bool is_record_update =
      std::holds_alternative<RecordUpdate>(record_action);
  if (is_record_update) {
    TF_RET_CHECK(state->commands.size() <= kMaxCommands)
        << "Too many commands to fit in inline storage";
    std::copy(state->commands.begin(), state->commands.end(), commands_storage);
    num_commands = state->commands.size();
  }

  const bool is_record_create =
      std::holds_alternative<RecordCreate>(record_action);
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

  XLA_FFI_RecordFrame_Extension record_frame_ext;
  XLA_FFI_RecordFrame record_frame;

  auto configure_call_frame = [&](XLA_FFI_CallFrame* raw_call_frame) {
    record_frame = {raw_call_frame, &record_ctx,      &ffi_record_api,
                    action_to_pass, commands_storage, &num_commands,
                    kMaxCommands};
    record_frame_ext.extension_base.struct_size =
        XLA_FFI_RecordFrame_Extension_STRUCT_SIZE;
    record_frame_ext.extension_base.type = XLA_FFI_Extension_RecordFrame;
    record_frame_ext.extension_base.next = nullptr;
    record_frame_ext.record_frame = &record_frame;
    raw_call_frame->extension_start = &record_frame_ext.extension_base;
    raw_call_frame->ctx->record_frame = &record_frame;
  };

  const RunId run_id = execute_params.collective_params
                           ? execute_params.collective_params->run_id
                           : RunId{-1};

  // Invoke FFI handler
  bool attempted_record = false;
  absl::Status status = absl::OkStatus();
  // If record_handler_ is present it means the FFI client registered their
  // record handler.
  if (record_handler_ != nullptr) {
    attempted_record = true;
    VLOG(5) << "CustomCallThunk::Record: record_handler_ symbol: "
            << GetSymbolName(reinterpret_cast<const void*>(record_handler_));
    status = ExecuteFfiHandler(
        run_id, record_handler_, XLA_FFI_ExecutionStage_RECORD,
        execute_params.stream, /*execution_scoped_state=*/nullptr,
        execute_params.ffi_execution_context, execute_params.buffer_allocations,
        execute_params.collective_params,
        /*collective_clique_requests=*/nullptr,
        /*collective_memory_requests=*/nullptr,
        execute_params.collective_cliques, execute_params.collective_memory,
        execute_params.additional_compute_streams,
        std::move(configure_call_frame));
  } else if (const auto* owned_bundle =
                 std::get_if<OwnedHandlerBundle>(&bundle_)) {
    if (owned_bundle->record) {
      attempted_record = true;
      status = ExecuteFfiHandler(
          run_id, *owned_bundle->record, xla::ffi::ExecutionStage::kRecord,
          execute_params.stream, /*execution_scoped_state=*/nullptr,
          execute_params.ffi_execution_context,
          execute_params.buffer_allocations, execute_params.collective_params,
          /*collective_clique_requests=*/nullptr,
          /*collective_memory_requests=*/nullptr,
          execute_params.collective_cliques, execute_params.collective_memory,
          execute_params.additional_compute_streams,
          std::move(configure_call_frame));
    }
  }

  // Fallback to tracing if requested or if no record handler was present
  if (!attempted_record || record_ctx.stream_capture_requested) {
    VLOG(3) << "FFI handler requested or required fallback to stream capture.";
    return TracedCommand::Record(execute_params, record_params, record_action,
                                 command_buffer);
  }
  RETURN_IF_ERROR(status);

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

  return absl::InternalError("No commands recorded by FFI handler");
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
    ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_operands(),
                     operand.ToProto());
  }

  for (const NullableShapedSlice& result : results_) {
    ASSIGN_OR_RETURN(*proto.mutable_custom_call_thunk()->add_results(),
                     result.ToProto());
  }

  if (attributes_.has_value()) {
    *proto.mutable_custom_call_thunk()->mutable_attributes() =
        attributes_->ToProto();
  }

  if (execution_state_ && execution_state_->IsSerializable()) {
    ASSIGN_OR_RETURN(
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
    ASSIGN_OR_RETURN(
        NullableShapedSlice operand,
        NullableShapedSlice::FromProto(operand_proto, buffer_allocations));
    operands.push_back(std::move(operand));
  }
  for (const auto& result_proto : proto.results()) {
    ASSIGN_OR_RETURN(
        NullableShapedSlice result,
        NullableShapedSlice::FromProto(result_proto, buffer_allocations));
    results.push_back(std::move(result));
  }

  ASSIGN_OR_RETURN(ffi::AttributesMap attributes,
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
