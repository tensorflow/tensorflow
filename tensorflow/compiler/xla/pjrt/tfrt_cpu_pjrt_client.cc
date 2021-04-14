/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"

#include <memory>

#define EIGEN_USE_THREADS

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace xla {

static const char kCpuPlatformName[] = "cpu";
static constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

static tfrt::AsyncValueRef<CpuEvent> GetOrCreateReadyEvent(
    tfrt::HostContext* host_context) {
  static const auto* ready_event = new tfrt::AsyncValueRef<CpuEvent>(
      tfrt::MakeAvailableAsyncValueRef<CpuEvent>(host_context));
  return ready_event->CopyRef();
}

TfrtCpuDevice::TfrtCpuDevice(int id, bool asynchronous)
    : id_(id),
      max_inflight_computations_semaphore_(/*capacity=*/asynchronous ? 32 : 1) {
}

absl::string_view TfrtCpuDevice::device_kind() const {
  return kCpuPlatformName;
}

std::string TfrtCpuDevice::DebugString() const {
  return absl::StrCat("TFRT_CPU_", id());
}

Status TfrtCpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  return TransferLiteralToInfeedOnCpu(local_hardware_id(), literal);
}

Status TfrtCpuDevice::TransferFromOutfeed(MutableBorrowingLiteral literal) {
  return TransferLiteralFromOutfeedOnCpu(local_hardware_id(), literal);
}

static int CpuDeviceCount() {
  // By default we fix the number of devices to one.  However we do let the user
  // override this behavior to help run tests on the host that run models in
  // parallel across multiple devices, e.g. pmap.
  return GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
}

static StatusOr<std::vector<std::unique_ptr<TfrtCpuDevice>>> GetTfrtCpuDevices(
    bool asynchronous) {
  std::vector<std::unique_ptr<TfrtCpuDevice>> devices;
  for (int i = 0; i < CpuDeviceCount(); ++i) {
    auto device = std::make_unique<TfrtCpuDevice>(
        /*id=*/i, asynchronous);
    devices.push_back(std::move(device));
  }
  return std::move(devices);
}

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous) {
  // TODO(zhangqiaorjc): Allow users set the number of threads.
  // `num_blocking_threads=16` is picked arbitrarily for now.
  // Need at least CpuDeviceCount threads to launch one collective.
  int num_threads = std::max(DefaultThreadPoolSize(), CpuDeviceCount());
  auto host_context = std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(ERROR) << "Encountered runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/num_threads,
          /*num_blocking_threads=*/16));

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                      GetTfrtCpuDevices(asynchronous));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtCpuClient>(
      /*process_index=*/0, std::move(devices), std::move(host_context)));
}

TfrtCpuClient::TfrtCpuClient(
    int process_index, std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
    std::unique_ptr<tfrt::HostContext> host_ctx)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      host_ctx_(std::move(host_ctx)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      eigen_intraop_pool_(new tensorflow::thread::ThreadPool(
          tensorflow::Env::Default(), "XLAEigen", DefaultThreadPoolSize())),
      eigen_intraop_device_(
          new Eigen::ThreadPoolDevice(eigen_intraop_pool_->AsEigenThreadPool(),
                                      eigen_intraop_pool_->NumThreads())) {
  for (const std::unique_ptr<TfrtCpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(id_to_device_.insert({device->id(), device.get()}).second)
        << "Duplicate device id: " << device->id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }
  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    CHECK(addressable_devices_[idx] != nullptr) << idx;
  }
  LOG(INFO) << "TfrtCpuClient created.";
}

StatusOr<PjRtDevice*> TfrtCpuClient::LookupDevice(int device_id) const {
  auto it = id_to_device_.find(device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         device_id);
}

StatusOr<PjRtDevice*> TfrtCpuClient::LookupAddressableDevice(
    int local_hardware_id) const {
  for (auto* device : addressable_devices_) {
    if (local_hardware_id == device->local_hardware_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_hardware_id %d",
                         local_hardware_id);
}

StatusOr<DeviceAssignment> TfrtCpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

StatusOr<std::unique_ptr<HloCostAnalysis>> TfrtCpuClient::GetHloCostAnalysis() {
  return absl::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
}

StatusOr<absl::optional<std::string>> TfrtCpuClient::ExecutableFingerprint(
    const PjRtExecutable& executable) const {
  return absl::optional<std::string>();
}

static StatusOr<std::unique_ptr<xla::Executable>> JitCompile(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_layouts, &execution_options,
                         execution_options.num_replicas(),
                         /*num_threads=*/absl::nullopt,
                         /*aot_options=*/nullptr));

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));
  VLOG(1) << "Unoptimized HLO module: " << hlo_module->ToString();

  // Run Hlo Passes
  cpu::CpuCompiler compiler;
  xla::Compiler::CompileOptions dummy;
  TF_ASSIGN_OR_RETURN(hlo_module,
                      compiler.RunHloPasses(std::move(hlo_module),
                                            /*stream_exec=*/nullptr, dummy));

  // Run backend.
  return compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr,
                             dummy);
}

// Find the root instruction of the entry computation.
static const InstructionValueSet& GetRootValueSet(
    const BufferAssignment& assignment, const HloModule& module) {
  return assignment.dataflow_analysis().GetInstructionValueSet(
      module.entry_computation()->root_instruction());
}

// Buffer table is indexed by buffer allocation indices. The output buffer is
// made up of a subset of those buffer allocations (for tuple, it includes tuple
// index table). This helper finds the buffer allocation indices in buffer
// assignment that make up for the output buffer. It is used by
// CreateResultShapedBuffer to reconstruct the output buffer from the buffer
// table allocated by MemoryForAllocation.
static StatusOr<absl::InlinedVector<BufferAllocation::Index, 4>>
FindResultBufferAllocationIndex(const BufferAssignment& assignment,
                                const HloModule& module) {
  absl::InlinedVector<BufferAllocation::Index, 4> buffer_indices;
  const InstructionValueSet& root_value_set =
      GetRootValueSet(assignment, module);
  const Shape& result_shape = module.result_shape();
  if (!result_shape.IsTuple()) {
    // Find the buffer allocation that corresponds to the output buffer.
    const HloValueSet& sources = root_value_set.element({});
    // The points to set is unambiguous so the set should be a singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment.GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();
    buffer_indices.push_back(buffer_index);
    return {std::move(buffer_indices)};
  }
  buffer_indices.reserve(result_shape.tuple_shapes_size());
  for (int i = 0; i < result_shape.tuple_shapes_size(); ++i) {
    // Find the buffer allocations that corresponds to the output tuple,
    // including the tuple index table.
    const HloValueSet& sources = root_value_set.element({i});
    // The points to set is unambiguous so the set should be a singleton.
    CHECK_EQ(1, sources.values().size());
    const HloValue* value_source = sources.values()[0];
    HloInstruction* src = value_source->instruction();
    TF_ASSIGN_OR_RETURN(const BufferAllocation::Slice slice,
                        assignment.GetUniqueSlice(src, value_source->index()));
    const BufferAllocation::Index buffer_index = slice.index();
    buffer_indices.push_back(buffer_index);
  }
  return {std::move(buffer_indices)};
}

StatusOr<std::unique_ptr<PjRtExecutable>> TfrtCpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuClient::Compile");
  ExecutableBuildOptions& build_options = options.executable_build_options;

  int num_replicas;
  int num_partitions;
  std::shared_ptr<DeviceAssignment> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options.compile_portable_executable, &options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  std::vector<const Shape*> argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation, &LayoutUtil::GetWithDefaultLayout, options.argument_layouts,
      &options.executable_build_options, &argument_layout_pointers));

  std::vector<PjRtExecutable::LogicalDeviceIds> addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int device_id = (*device_assignment)(replica, partition);
        TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
        if (device->process_index() != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        PjRtExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (addressable_devices.empty()) {
      return InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }

    if (build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id());
    }
  }

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> cpu_executable,
                      JitCompile(computation, argument_layout_pointers,
                                 build_options, execution_options));
  auto cpu_executable_ptr =
      down_cast<cpu::CpuExecutable*>(cpu_executable.get());

  // `buffer_table[result_slice.index()]` points to result buffer:
  // If output is a tuple, it points to the buffer index table.
  // If output is a non-tuple, it points to the buffer itself.
  TF_ASSIGN_OR_RETURN(
      const BufferAllocation::Slice result_slice,
      cpu_executable_ptr->buffer_assignment().GetUniqueTopLevelOutputSlice());

  // `result_buffer_indices` has the buffer allocation indices that make up the
  // output buffer (could be tuple).
  TF_ASSIGN_OR_RETURN(
      auto result_buffer_indices,
      FindResultBufferAllocationIndex(cpu_executable_ptr->buffer_assignment(),
                                      cpu_executable->module()));

  auto executable = std::make_unique<TfrtCpuExecutable>(
      num_replicas, num_partitions, std::move(device_assignment),
      options.parameter_is_tupled_arguments, std::move(cpu_executable),
      result_slice.index(), std::move(result_buffer_indices),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);
  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));

  return std::unique_ptr<PjRtExecutable>(std::move(executable));
}

StatusOr<std::unique_ptr<TfrtCpuBuffer>> AllocateDestinationBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events,
    TfrtCpuDevice* device, TfrtCpuClient* client) {
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers;
  if (!on_device_shape.IsTuple()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
    auto device_buffer = MaybeOwningCpuMemory::AllocateShared(byte_size);
    buffers.push_back(std::move(device_buffer));
    return std::make_unique<TfrtCpuBuffer>(
        on_device_shape,
        std::make_shared<TrackedTfrtCpuDeviceBuffer>(
            /*is_tuple=*/false, std::move(buffers),
            std::move(definition_events)),
        client, device);
  }
  // Tuple case.
  buffers.reserve(on_device_shape.tuple_shapes().size());
  for (const auto& leaf_shape : on_device_shape.tuple_shapes()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(leaf_shape);
    auto device_buffer = MaybeOwningCpuMemory::AllocateShared(byte_size);
    buffers.push_back(std::move(device_buffer));
  }
  return std::make_unique<TfrtCpuBuffer>(
      on_device_shape,
      std::make_shared<TrackedTfrtCpuDeviceBuffer>(
          /*is_tuple=*/true, std::move(buffers), std::move(definition_events)),
      client, device);
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtDevice* device,
    std::function<void()> on_delete_callback) {
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers;
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  auto non_owning_buffer =
      std::make_shared<MaybeOwningCpuMemory>(device_ptr, byte_size);
  buffers.push_back(std::move(non_owning_buffer));
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> empty_definition_events;
  auto tracked_device_buffer = std::make_shared<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/false, std::move(buffers),
      std::move(empty_definition_events), std::move(on_delete_callback));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<TfrtCpuBuffer>(shape, std::move(tracked_device_buffer),
                                      this, down_cast<TfrtCpuDevice*>(device)));
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme(
      "TfrtCpuClient::CreateUninitializedBuffer");
  VLOG(1) << "TfrtCpuClient::CreateUninitializedBuffer: shape: "
          << shape.DebugString() << " device: " << device->DebugString();
  return AllocateDestinationBuffer(shape, /*definition_events=*/{},
                                   down_cast<TfrtCpuDevice*>(device), this);
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::BufferFromHostBuffer(
    const void* data, const Shape& shape,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostBuffer");
  VLOG(2) << "TfrtCpuClient::BufferFromHostBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString();
  if (shape.IsTuple()) {
    return InvalidArgument("Use BufferFromHostLiteral to transfer a tuple");
  }
  // If the input buffer is sufficiently aligned, we can simply point to the
  // input array's data without any further copies. At the time of writing we
  // require a 16-byte alignment because XLA may generate code which requires
  // it.
  bool can_use_zero_copy =
      host_buffer_semantics == HostBufferSemantics::kZeroCopy &&
      ((absl::bit_cast<std::uintptr_t>(data) &
        (cpu_function_runtime::kMinAlign - 1)) == 0);
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
  std::function<void()> on_delete_callback;
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  if (can_use_zero_copy) {
    auto device_buffer = std::make_shared<MaybeOwningCpuMemory>(
        const_cast<void*>(data), byte_size);
    buffers.push_back(std::move(device_buffer));
    on_delete_callback = std::move(on_done_with_host_buffer);
  } else {
    auto device_buffer = MaybeOwningCpuMemory::AllocateShared(byte_size);
    auto dst_data_ptr = device_buffer->data();
    buffers.push_back(std::move(device_buffer));
    bool should_sync_copy =
        host_buffer_semantics ==
            HostBufferSemantics::kImmutableUntilTransferCompletes ||
        (byte_size < kSmallDataTransferByteSize);
    if (should_sync_copy) {
      std::memcpy(dst_data_ptr, data, byte_size);
      if (on_done_with_host_buffer) {
        on_done_with_host_buffer();
        on_done_with_host_buffer = nullptr;
      }
    } else {
      tfrt::AsyncValueRef<CpuEvent> copy_event =
          tfrt::MakeConstructedAsyncValueRef<CpuEvent>(host_ctx_.get());
      definition_events.push_back(copy_event.CopyRef());
      tfrt::EnqueueWork(
          host_ctx_.get(),
          [dst_data_ptr, data, byte_size, copy_event = std::move(copy_event),
           on_done_with_host_buffer =
               std::move(on_done_with_host_buffer)]() mutable {
            tensorflow::profiler::TraceMe traceme("H2D Dispatch");
            std::memcpy(dst_data_ptr, data, byte_size);
            if (on_done_with_host_buffer) {
              on_done_with_host_buffer();
              on_done_with_host_buffer = nullptr;
            }
            // Signal copy is complete.
            copy_event.SetStateConcrete();
          });
    }
  }
  auto tracked_device_buffer = std::make_shared<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/false, std::move(buffers), std::move(definition_events),
      std::move(on_delete_callback));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<TfrtCpuBuffer>(shape, std::move(tracked_device_buffer),
                                      this, down_cast<TfrtCpuDevice*>(device)));
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostLiteral");
  VLOG(1) << "TfrtCpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().DebugString()
          << " device: " << device->DebugString();
  const Shape& shape = literal.shape();

  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer. They are set only after h2d dispatch.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
  absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4> avs;
  int num_leaf_buffers = shape.IsTuple() ? shape.tuple_shapes_size() : 1;
  for (int i = 0; i < num_leaf_buffers; ++i) {
    tfrt::AsyncValueRef<CpuEvent> definition_event =
        tfrt::MakeConstructedAsyncValueRef<CpuEvent>(GetHostContext());
    definition_events.push_back(definition_event.CopyRef());
    avs.push_back(std::move(definition_event));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfrtCpuBuffer> output_buffer,
      AllocateDestinationBuffer(shape, std::move(definition_events),
                                down_cast<TfrtCpuDevice*>(device), this));

  if (!shape.IsTuple()) {
    TfrtCpuBuffer::ScopedHold device_buffer(
        output_buffer->GetBufferWithUsageHold());
    CHECK(device_buffer.ok());
    // It is OK to capture `buffer` pointer because the `output_buffer` can't be
    // deleted until all the usage holds have gone away.
    tfrt::EnqueueWork(
        GetHostContext(),
        [literal, av = avs[0].CopyRef(),
         movable_device_buffer{device_buffer.ToClosure()}, shape]() mutable {
          tensorflow::profiler::TraceMe traceme("H2D Dispatch");
          TfrtCpuBuffer::ScopedHold device_buffer(movable_device_buffer);
          const std::shared_ptr<MaybeOwningCpuMemory>& b =
              device_buffer->Buffers()[0];
          CHECK_EQ(literal.size_bytes(), b->size());
          std::memcpy(b->data(), literal.untyped_data(), b->size());
          // Signal copy is complete.
          av->SetStateConcrete();
        });
  } else {
    // For tuple, transfer leaf literal individually in parallel.
    for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
      TfrtCpuBuffer::ScopedHold device_buffer(
          output_buffer->GetBufferWithUsageHold());
      CHECK(device_buffer.ok());
      // It is OK to capture `buffer` pointer because the `output_buffer` can't
      // be deleted until all the usage holds have gone away.
      tfrt::EnqueueWork(
          GetHostContext(),
          [i, literal, av = avs[i].CopyRef(), shape,
           movable_device_buffer{device_buffer.ToClosure()}]() mutable {
            tensorflow::profiler::TraceMe traceme("H2D Dispatch");
            TfrtCpuBuffer::ScopedHold device_buffer(movable_device_buffer);
            auto slice = LiteralSlice(literal, {i});
            const std::shared_ptr<MaybeOwningCpuMemory>& b =
                device_buffer->Buffers()[i];
            CHECK_EQ(slice.size_bytes(), b->size());
            std::memcpy(b->data(), slice.untyped_data(), slice.size_bytes());
            // Signal copy is complete.
            av->SetStateConcrete();
          });
    }
  }
  return std::unique_ptr<PjRtBuffer>(std::move(output_buffer));
}

TfrtCpuBuffer::ScopedHold::~ScopedHold() {
  if (ok()) {
    parent_->DropHold(type_, buffer().get());
  }
}

TfrtCpuBuffer::ScopedHold::ScopedHold(ScopedHold&& other)
    : parent_(other.parent_),
      type_(other.type_),
      state_(other.state_),
      status_(std::move(other.status_)),
      buffer_(std::move(other.buffer_)) {
  // Preserve the invariant that status is invalid if buffer == nullptr.
  other.SetState(kMoved);
}

void TfrtCpuBuffer::ScopedHold::Acquire(
    StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>&& buffer_or) {
  CHECK(!ok());
  if (buffer_or.ok()) {
    buffer_ = buffer_or.ValueOrDie();
    SetState(kValid);
  } else {
    status_ = buffer_or.status();
    buffer_ = nullptr;
    SetState(kError);
  }
  // Check the invariant holds.
  CHECK(!ok() || buffer_ != nullptr);
}

TfrtCpuBuffer::ScopedHold::ForClosure TfrtCpuBuffer::ScopedHold::ToClosure() {
  CHECK(ok());
  ForClosure for_closure(parent_, type_, state_, std::move(status_),
                         std::move(buffer_));
  SetState(kReleased);
  return for_closure;
}

void TfrtCpuBuffer::ScopedHold::ConvertUsageHold(
    absl::Span<tfrt::AsyncValueRef<CpuEvent>> events) {
  CHECK(ok());
  CHECK_EQ(type_, kUsage);
  parent_->ConvertUsageHold(buffer().get(), events);
  SetState(kConverted);
}

void TfrtCpuBuffer::ScopedHold::ConfirmDonation() {
  CHECK(ok());
  CHECK_EQ(type_, kDonation);
  parent_->ConfirmDonation(buffer().get());
  SetState(kDonated);
}

TfrtCpuBuffer::TfrtCpuBuffer(
    Shape on_device_shape,
    std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
    TfrtCpuClient* client, TfrtCpuDevice* device)
    : client_(client),
      on_device_shape_(std::move(on_device_shape)),
      device_(device),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {
  for (int i = 0; i < ScopedHold::Type::kMaxValue; ++i) {
    holds_[i] = 0;
  }
}

TfrtCpuBuffer::~TfrtCpuBuffer() {
  Delete();
  for (int i = 0; i < ScopedHold::Type::kMaxValue; ++i) {
    CHECK_EQ(holds_[i], 0);
  }
}

int64 TfrtCpuBuffer::OnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

namespace {

// Implements PjRtBuffer::ExternalReference as a wrapped
// ScopedHold::kExternalReference.
class ScopedHoldAsExternalReference : public PjRtBuffer::ExternalReference {
 public:
  explicit ScopedHoldAsExternalReference(TfrtCpuBuffer::ScopedHold hold)
      : external_reference_(std::move(hold)) {
    CHECK(external_reference_.type() ==
          TfrtCpuBuffer::ScopedHold::kExternalReference);
    data_ptr_ = external_reference_->Buffers()[0]->data();
  }

  ~ScopedHoldAsExternalReference() override = default;

 private:
  TfrtCpuBuffer::ScopedHold external_reference_;
};

}  // namespace

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtCpuBuffer::AcquireExternalReference() {
  ScopedHold hold = GetBufferWithExternalReference();
  Status hold_status = hold.status();
  if (!hold_status.ok()) return hold_status;
  return std::unique_ptr<ExternalReference>(
      std::make_unique<ScopedHoldAsExternalReference>(std::move(hold)));
}

class TrackedCpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedCpuDeviceBufferExternalReference(
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->Buffers()[0]->data();
  }

  ~TrackedCpuDeviceBufferExternalReference() override = default;

 private:
  std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_;
};

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtCpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedCpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

void TfrtCpuBuffer::Delete() {
  // When wait_for_reads_to_complete is false, Release should never fail.
  TF_CHECK_OK(Release(/*wait_for_operations_to_complete=*/false).status());
}

bool TfrtCpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

void TfrtCpuBuffer::WaitForOutstandingUsageHolds() {
  auto not_in_usage_hold = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return holds_[ScopedHold::kUsage] == 0;
  };
  mu_.Await(absl::Condition(&not_in_usage_hold));
}

void TfrtCpuBuffer::WaitForOutstandingDonationHold() {
  auto not_in_donation_hold = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return holds_[ScopedHold::kDonation] == 0;
  };
  mu_.Await(absl::Condition(&not_in_donation_hold));
}

StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>> TfrtCpuBuffer::Release(
    bool wait_for_operations_to_complete) {
  std::shared_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> events;
  {
    absl::MutexLock lock(&mu_);
    // We first wait for a donation hold to complete if there is one in
    // progress. If the donation succeeds via ConfirmDonation() then it will
    // set device_buffer_ to nullptr before returning to this thread.
    WaitForOutstandingDonationHold();
    if (tracked_device_buffer_ == nullptr) {
      // Buffer has been deleted.
      return std::shared_ptr<TrackedTfrtCpuDeviceBuffer>();
    }
    // Set device_buffer_ to null now so that no other thread can add a hold
    // while we are in WaitForOutstandingUsageHolds() below.
    std::swap(tracked_device_buffer_, device_buffer);
    WaitForOutstandingUsageHolds();
    // Now that all holds have completed and no more can be added, we can get
    // the final set of usage events.
    events = device_buffer->LockUseAndTransferUsageEvents();
  }
  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    Status first_error;
    for (const auto& av : events) {
      client_->GetHostContext()->Await(av.CopyRCRef());
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(InternalError("Error Execute: %s", error->message));
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }
  return std::move(device_buffer);
}

StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>
TfrtCpuBuffer::GetBufferForHoldLocked(ScopedHold::Type type) {
  // All callers should have called WaitForOutstandingDonationHold().
  CHECK_EQ(holds_[ScopedHold::kDonation], 0);
  if (type == ScopedHold::kDonation) {
    if (tracked_device_buffer_ == nullptr) {
      return InvalidArgument("Donation requested for invalid buffer");
    }
    if (holds_[ScopedHold::kExternalReference] > 0) {
      return InvalidArgument(
          "Donation requested for buffer with external reference");
    }
    // First add the donation hold.
    ++holds_[type];
    // Then wait for any usage holds to be dropped or converted. No new usage
    // holds can be added until we drop the donation hold so this wait will
    // complete eventually.
    WaitForOutstandingUsageHolds();
    // Because we added a donation hold, nobody could release the buffer while
    // we were waiting.
    CHECK(tracked_device_buffer_ != nullptr);
  } else {
    if (tracked_device_buffer_ == nullptr) {
      return InvalidArgument("Hold requested on deleted or donated buffer");
    } else {
      ++holds_[type];
    }
  }
  return tracked_device_buffer_;
}

void TfrtCpuBuffer::AcquireHoldLocked(ScopedHold* hold) {
  hold->Acquire(GetBufferForHoldLocked(hold->type()));
}

TfrtCpuBuffer::ScopedHold TfrtCpuBuffer::GetBufferWithHold(
    ScopedHold::Type type) {
  absl::MutexLock lock(&mu_);
  // Ensure that at most one donation hold can be in progress at a time.
  WaitForOutstandingDonationHold();
  ScopedHold hold(this, type);
  AcquireHoldLocked(&hold);
  return hold;
}

void TfrtCpuBuffer::ConvertUsageHold(
    TrackedTfrtCpuDeviceBuffer* buffer,
    absl::Span<tfrt::AsyncValueRef<CpuEvent>> events) {
  absl::MutexLock lock(&mu_);
  CHECK(tracked_device_buffer_.get() == buffer ||
        tracked_device_buffer_ == nullptr);
  buffer->AddUsageEvents(events);
  CHECK_GT(holds_[ScopedHold::kUsage], 0);
  --holds_[ScopedHold::kUsage];
}

void TfrtCpuBuffer::ConfirmDonation(TrackedTfrtCpuDeviceBuffer* device_buffer) {
  {
    absl::MutexLock lock(&mu_);
    CHECK_EQ(holds_[ScopedHold::kUsage], 0);
    CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
    CHECK_EQ(holds_[ScopedHold::kDonation], 1);
    holds_[ScopedHold::kDonation] = 0;
    CHECK(tracked_device_buffer_.get() == device_buffer);
    // As a sanity check ensure no more usage events can be added to the buffer.
    device_buffer->LockUseAndTransferUsageEvents();
    // Give up ownership of the device memory so we don't free it when the last
    // reference to device_buffer_ goes away.
    device_buffer->ReleaseDeviceMemory();
    // Make *this invalid so it can't be used again. Any threads blocking in
    // Release or GetBufferWithHold will see an invalid buffer and return.
    tracked_device_buffer_.reset();
  }
}

void TfrtCpuBuffer::DropHold(ScopedHold::Type type,
                             TrackedTfrtCpuDeviceBuffer* buffer) {
  absl::MutexLock lock(&mu_);
  CHECK(tracked_device_buffer_.get() == buffer ||
        tracked_device_buffer_ == nullptr);
  CHECK_GT(holds_[type], 0);
  --holds_[type];
  if (type == ScopedHold::kDonation) {
    CHECK_EQ(holds_[ScopedHold::kDonation], 0);
    CHECK_EQ(holds_[ScopedHold::kUsage], 0);
    CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
  }
}

static ShapedBuffer AsShapedBuffer(
    int device_ordinal, const Shape& on_device_shape,
    absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> buffers) {
  ShapedBuffer shaped_buffer(on_device_shape, device_ordinal);
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  for (const auto& buf : buffers) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = se::DeviceMemoryBase(buf->data(), buf->size());
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

StatusOr<Shape> TfrtCpuBuffer::logical_on_device_shape() {
  ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    // We can't perform any other action while a donation hold is in progress.
    WaitForOutstandingDonationHold();
    if (tracked_device_buffer_ == nullptr) {
      return InvalidArgument(
          "logical_on_device_shape() called on deleted or donated buffer");
    }
    AcquireHoldLocked(&device_buffer);
  }

  // Wait for definition events.
  for (const auto& av : device_buffer->DefinitionEvents()) {
    client_->GetHostContext()->Await(av.CopyRCRef());
    if (auto* error = av.GetErrorIfPresent()) {
      return InternalError("Error Execute: %s", error->message);
    }
  }

  ShapedBuffer shaped_buffer = AsShapedBuffer(
      device_->local_hardware_id(), on_device_shape_, device_buffer->Buffers());
  Shape ret_shape = on_device_shape_;
  TF_RETURN_IF_ERROR(ReadDynamicShapesOnCpu(
      &shaped_buffer, &ret_shape, cpu::CpuExecutable::ShapeSizeBytes));
  return ret_shape;
}

static std::vector<tfrt::RCReference<tfrt::AsyncValue>> GetAsyncValues(
    absl::Span<const tfrt::AsyncValueRef<CpuEvent>> events) {
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev.CopyRCRef());
  }
  return avs;
}

// Enqueue to TFRT non-blocking work queue when all `values` are ready.
static void EnqueueWorkWhenReady(
    tfrt::HostContext* host_ctx,
    tfrt::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> values,
    llvm::unique_function<void()> callee) {
  tfrt::RunWhenReady(values, [host_ctx, callee = std::move(callee)]() mutable {
    tfrt::EnqueueWork(host_ctx, std::move(callee));
  });
}

void TfrtCpuBuffer::ToLiteral(MutableLiteralBase* literal,
                              std::function<void(Status)> on_ready) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuBuffer::ToLiteral");
  if (IsEmptyTuple()) {
    on_ready(InvalidArgument("ToLiteral called on empty tuple"));
    return;
  }
  TfrtCpuBuffer::ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    // We can't perform any other action while a donation hold is in progress.
    WaitForOutstandingDonationHold();
    if (tracked_device_buffer_ == nullptr) {
      on_ready(InvalidArgument(
          "CopyToHostAsync() called on deleted or donated buffer"));
      return;
    }
    AcquireHoldLocked(&device_buffer);
  }
  auto host_ctx = client_->GetHostContext();

  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs =
      GetAsyncValues(device_buffer.buffer()->DefinitionEvents());

  bool should_sync_copy = device_buffer_wait_avs.empty() &&
                          literal->size_bytes() < kSmallDataTransferByteSize;
  if (should_sync_copy) {
    if (!on_device_shape().IsTuple()) {
      const std::shared_ptr<MaybeOwningCpuMemory>& b =
          device_buffer.buffer()->Buffers()[0];
      std::memcpy(literal->untyped_data(), b->data(), b->size());
    } else {
      // Tuple case.
      int num_leaves = literal->shape().tuple_shapes().size();
      for (int i = 0; i < num_leaves; ++i) {
        const std::shared_ptr<MaybeOwningCpuMemory>& b =
            device_buffer.buffer()->Buffers()[i];
        std::memcpy(literal->untyped_data({i}), b->data(), b->size());
      }
    }
    // Unblock ToLiteral caller.
    on_ready(Status::OK());
  } else {
    // Wait for buffer definition events to finish before d2h dispatch.
    // D2H dispatch should be in parallel, e.g. one Execute event finish may
    // trigger multiple outputs' D2H, they should happen in different threads in
    // parallel.
    EnqueueWorkWhenReady(
        host_ctx, device_buffer_wait_avs,
        [this, movable_device_buffer{device_buffer.ToClosure()},
         device_buffer_wait_avs = std::move(device_buffer_wait_avs), literal,
         on_ready{std::move(on_ready)}] {
          tensorflow::profiler::TraceMe traceme("D2H Dispatch");
          TfrtCpuBuffer::ScopedHold device_buffer(movable_device_buffer);
          // Errors in src buffer are surfaced to user.
          for (const auto& av : device_buffer_wait_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              on_ready(
                  Internal("Error converting to literal: %s", error->message));
              return;
            }
          }

          if (!on_device_shape().IsTuple()) {
            const std::shared_ptr<MaybeOwningCpuMemory>& b =
                device_buffer.buffer()->Buffers()[0];
            std::memcpy(literal->untyped_data(), b->data(), b->size());
          } else {
            // Tuple case.
            int num_leaves = literal->shape().tuple_shapes().size();
            for (int i = 0; i < num_leaves; ++i) {
              const std::shared_ptr<MaybeOwningCpuMemory>& b =
                  device_buffer.buffer()->Buffers()[i];
              std::memcpy(literal->untyped_data({i}), b->data(), b->size());
            }
          }

          // Unblock ToLiteral caller.
          on_ready(Status::OK());
        });
  }
}

// TODO(zhangqiaorjc): Consider disallowing multiple CPU devices and assign
// multiple pmap replicas to the same CPU device for multi-CPU pmap testing.
StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuBuffer::CopyToDevice");

  // Copy each leaf buffer to a destination buffer.
  TfrtCpuBuffer::ScopedHold src_device_buffer(
      this, TfrtCpuBuffer::ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    WaitForOutstandingDonationHold();
    if (tracked_device_buffer_ == nullptr) {
      return InvalidArgument(
          "CopyToDevice called on deleted or donated buffer");
    }
    AcquireHoldLocked(&src_device_buffer);
  }

  int num_leaf_buffers = src_device_buffer->Buffers().size();
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> src_buffers;
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> dst_buffers;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
  absl::InlinedVector<tfrt::RCReference<tfrt::IndirectAsyncValue>, 4>
      indirect_avs;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> src_usage_events;
  src_buffers.reserve(num_leaf_buffers);
  dst_buffers.reserve(num_leaf_buffers);
  definition_events.reserve(num_leaf_buffers);
  indirect_avs.reserve(num_leaf_buffers);
  src_usage_events.reserve(num_leaf_buffers);

  for (int i = 0; i < num_leaf_buffers; ++i) {
    auto src_buffer = src_device_buffer->Buffers()[i];
    auto dst_buffer = MaybeOwningCpuMemory::AllocateShared(src_buffer->size());
    src_buffers.push_back(std::move(src_buffer));
    dst_buffers.push_back(std::move(dst_buffer));
    tfrt::RCReference<tfrt::IndirectAsyncValue> definition_event =
        tfrt::MakeIndirectAsyncValue(client_->GetHostContext());
    definition_events.push_back(
        tfrt::AsyncValueRef<CpuEvent>(definition_event.CopyRef()));
    indirect_avs.push_back(definition_event.CopyRef());
    src_usage_events.push_back(
        tfrt::AsyncValueRef<CpuEvent>(std::move(definition_event)));
  }

  // Wait for src buffer definition events to finish before d2d dispatch.
  // Errors are propagated asynchronously in dst buffer's definition events.
  std::vector<tfrt::RCReference<tfrt::AsyncValue>>
      src_device_buffer_definition_events_avs =
          GetAsyncValues(src_device_buffer.buffer()->DefinitionEvents());

  // Add d2d as usage event on src_buffer.
  src_device_buffer.ConvertUsageHold(absl::MakeSpan(src_usage_events));

  EnqueueWorkWhenReady(
      client()->GetHostContext(), src_device_buffer_definition_events_avs,
      [client = client_, num_leaf_buffers, src_buffers = std::move(src_buffers),
       dst_buffers_copies = dst_buffers, indirect_avs = std::move(indirect_avs),
       src_device_buffer_definition_events_avs =
           std::move(src_device_buffer_definition_events_avs)]() mutable {
        tensorflow::profiler::TraceMe traceme("D2D Dispatch");
        for (const auto& av : src_device_buffer_definition_events_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            for (int i = 0; i < num_leaf_buffers; ++i) {
              // Any error discovered in src buffer are propagated to dst buffer
              // definition events, which will surface to users in
              // dst_buffer->ToLiteral().
              indirect_avs[i]->ForwardTo(av.CopyRef());
            }
            return;
          }
        }
        auto copy_ready = GetOrCreateReadyEvent(client->GetHostContext());
        for (int i = 0; i < num_leaf_buffers; ++i) {
          std::memcpy(dst_buffers_copies[i]->data(), src_buffers[i]->data(),
                      src_buffers[i]->size());
          indirect_avs[i]->ForwardTo(copy_ready.CopyRCRef());
        }
      });

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      on_device_shape_,
      std::make_shared<TrackedTfrtCpuDeviceBuffer>(
          on_device_shape_.IsTuple(), std::move(dst_buffers),
          std::move(definition_events)),
      client(), down_cast<TfrtCpuDevice*>(dst_device)));
}

Status TfrtCpuBuffer::BlockHostUntilReady() {
  tensorflow::profiler::TraceMe traceme("TfrtCpuBuffer::BlockHostUntilReady");
  std::shared_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(&mu_);
    if (tracked_device_buffer_ == nullptr) {
      return InvalidArgument(
          "BlockHostUntilReady() called on deleted or donated buffer");
    }
    device_buffer = tracked_device_buffer_;
  }

  // Wait for all definition events to complete.
  Status status;
  for (const auto& ev : device_buffer->DefinitionEvents()) {
    client_->GetHostContext()->Await(ev.CopyRCRef());
    if (auto* error = ev.GetErrorIfPresent()) {
      status.Update(FailedPrecondition(
          "Error in BlockHostUntilReady waiting for definition events: %s",
          error->message));
    }
  }
  return status;
}

TfrtCpuExecutable::TfrtCpuExecutable(
    int num_replicas, int num_partitions,
    std::shared_ptr<DeviceAssignment> device_assignment,
    bool parameter_is_tupled_arguments,
    std::unique_ptr<Executable> cpu_executable,
    BufferAllocation::Index result_buffer_index,
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtCpuClient* client)
    : client_(client),
      num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      device_assignment_(std::move(device_assignment)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      cpu_executable_(std::move(cpu_executable)),
      result_buffer_index_(result_buffer_index),
      result_buffer_indices_(std::move(result_buffer_indices)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {}

void TfrtCpuExecutable::Delete() {}

StatusOr<absl::optional<std::string>> TfrtCpuExecutable::Fingerprint() const {
  return absl::optional<std::string>();
}

Status TfrtCpuExecutable::SetUpDonation(bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(parameters_that_must_be_donated_,
                      GetParametersThatMustBeDonated(
                          *cpu_executable_->shared_module(), tuple_inputs));
  return Status::OK();
}

bool TfrtCpuExecutable::MustDonateParameter(int parameter) const {
  return parameters_that_must_be_donated_.contains(parameter);
}

}  // namespace xla
