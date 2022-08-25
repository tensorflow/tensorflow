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

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/util.h"

#define EIGEN_USE_THREADS

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace xla {
namespace {

// A RAII helper class used to set an AsyncValueRef<CpuEvent> to a ready state
// upon destruction. In many cases in PjRt implementation, there will be
// multiple return statements in the function, all of which require setting some
// AsyncValueRef<CpuEvent> to be ready. This class could make such code more
// robust by using setting the AsyncValue in the destructor.
class MarkEventReadyOnExit {
 public:
  explicit MarkEventReadyOnExit(tfrt::AsyncValueRef<CpuEvent> event)
      : event_(std::move(event)) {}

  MarkEventReadyOnExit(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit& operator=(const MarkEventReadyOnExit&) = delete;
  MarkEventReadyOnExit(MarkEventReadyOnExit&&) = default;
  MarkEventReadyOnExit& operator=(MarkEventReadyOnExit&&) = default;

  ~MarkEventReadyOnExit() {
    if (event_) event_.SetStateConcrete();
  }

  tfrt::AsyncValueRef<CpuEvent> Release() && { return std::move(event_); }

 private:
  tfrt::AsyncValueRef<CpuEvent> event_;
};

}  // namespace

static const char kCpuPlatformName[] = "cpu";
static constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

static tfrt::AsyncValueRef<CpuEvent> GetOrCreateReadyEvent(
    tfrt::HostContext* host_context) {
  static const auto* ready_event = new tfrt::AsyncValueRef<CpuEvent>(
      tfrt::MakeAvailableAsyncValueRef<CpuEvent>());
  return ready_event->CopyRef();
}

TfrtCpuDevice::TfrtCpuDevice(int id, bool asynchronous)
    : id_(id),
      max_inflight_computations_semaphore_(/*capacity=*/asynchronous ? 32 : 1) {
  debug_string_ = absl::StrCat("TFRT_CPU_", id);
  to_string_ = absl::StrCat("CpuDevice(id=", id, ")");
}

absl::string_view TfrtCpuDevice::device_kind() const {
  return kCpuPlatformName;
}

absl::string_view TfrtCpuDevice::DebugString() const { return debug_string_; }

absl::string_view TfrtCpuDevice::ToString() const { return to_string_; }

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
    bool asynchronous, int cpu_device_count) {
  std::vector<std::unique_ptr<TfrtCpuDevice>> devices;
  for (int i = 0; i < cpu_device_count; ++i) {
    auto device = std::make_unique<TfrtCpuDevice>(
        /*id=*/i, asynchronous);
    devices.push_back(std::move(device));
  }
  return std::move(devices);
}

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous,
                                                       int cpu_device_count) {
  // TODO(zhangqiaorjc): Allow users set the number of threads.
  // `num_blocking_threads=16` is picked arbitrarily for now.
  // Need at least CpuDeviceCount threads to launch one collective.
  int num_threads = std::max(DefaultThreadPoolSize(), cpu_device_count);
  auto host_context = std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(ERROR) << "Encountered runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/num_threads,
          /*num_blocking_threads=*/16));

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                      GetTfrtCpuDevices(asynchronous, cpu_device_count));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtCpuClient>(
      /*process_index=*/0, std::move(devices), std::move(host_context)));
}

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous) {
  return GetTfrtCpuClient(asynchronous, CpuDeviceCount());
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
                                      eigen_intraop_pool_->NumThreads())),
      last_collective_launch_event_(
          tfrt::MakeAvailableAsyncValueRef<CpuEvent>()),
      transpose_cache_(1024) {
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

TfrtCpuClient::~TfrtCpuClient() { LOG(INFO) << "TfrtCpuClient destroyed."; }

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
  return std::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
}

StatusOr<std::optional<std::string>> TfrtCpuClient::ExecutableFingerprint(
    const PjRtLoadedExecutable& executable) const {
  return std::optional<std::string>();
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
                         /*num_threads=*/std::nullopt,
                         /*aot_options=*/nullptr));

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));
  VLOG(3) << "Unoptimized HLO module: " << hlo_module->ToString();
  static constexpr char kBeforeOptimizationsDumpName[] = "before_optimizations";
  DumpHloModuleIfEnabled(*hlo_module, kBeforeOptimizationsDumpName);

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

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtCpuClient::Compile(
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

  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
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
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
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
      tensorflow::down_cast<cpu::CpuExecutable*>(cpu_executable.get());

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

  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtCpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false));
  return Compile(xla_computation, options);
}

StatusOr<std::unique_ptr<TfrtCpuBuffer>> AllocateDestinationBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events,
    TfrtCpuDevice* device, TfrtCpuClient* client) {
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers;
  if (!on_device_shape.IsTuple()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
    TF_ASSIGN_OR_RETURN(auto device_buffer,
                        MaybeOwningCpuMemory::AllocateShared(byte_size));
    buffers.push_back(std::move(device_buffer));
    return std::make_unique<TfrtCpuBuffer>(
        on_device_shape,
        std::make_unique<TrackedTfrtCpuDeviceBuffer>(
            /*is_tuple=*/false, std::move(buffers),
            std::move(definition_events)),
        client, device);
  }
  // Tuple case.
  buffers.reserve(on_device_shape.tuple_shapes().size());
  for (const auto& leaf_shape : on_device_shape.tuple_shapes()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(leaf_shape);
    TF_ASSIGN_OR_RETURN(auto device_buffer,
                        MaybeOwningCpuMemory::AllocateShared(byte_size));
    buffers.push_back(std::move(device_buffer));
  }
  return std::make_unique<TfrtCpuBuffer>(
      on_device_shape,
      std::make_unique<TrackedTfrtCpuDeviceBuffer>(
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
  auto tracked_device_buffer = std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/false, std::move(buffers),
      /*definition_event=*/tfrt::MakeAvailableAsyncValueRef<CpuEvent>(),
      std::move(on_delete_callback));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tensorflow::down_cast<TfrtCpuDevice*>(device)));
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme(
      "TfrtCpuClient::CreateUninitializedBuffer");
  VLOG(1) << "TfrtCpuClient::CreateUninitializedBuffer: shape: "
          << shape.DebugString() << " device: " << device->DebugString();
  return AllocateDestinationBuffer(
      shape, /*definition_events=*/{},
      tensorflow::down_cast<TfrtCpuDevice*>(device), this);
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostBuffer");
  Shape shape = ShapeUtil::MakeShape(type, dims);
  VLOG(2) << "TfrtCpuClient::BufferFromHostBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString();
  bool has_default_layout =
      !byte_strides || HasMajorToMinorLayout(type, dims, *byte_strides);
  // If the input buffer has a default layout and is sufficiently aligned, we
  // can simply point to the input array's data without any further copies. At
  // the time of writing we require a 16-byte alignment because XLA may generate
  // code which requires it.
  bool can_use_zero_copy =
      has_default_layout &&
      host_buffer_semantics == HostBufferSemantics::kZeroCopy &&
      ((absl::bit_cast<std::uintptr_t>(data) &
        (cpu_function_runtime::MinAlign() - 1)) == 0);
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
    TF_ASSIGN_OR_RETURN(auto device_buffer,
                        MaybeOwningCpuMemory::AllocateShared(byte_size));
    auto dst_data_ptr = device_buffer->data();
    buffers.push_back(device_buffer);
    if (!has_default_layout) {
      // If the input array does not have a major-to-minor layout, transpose it
      // into major-to-minor layout. Currently we choose to always do this
      // synchronously.
      // TODO(phawkins): consider performing the transpose asynchronously.
      // TODO(phawkins): parallelize the transpose.
      std::shared_ptr<TransposePlan> transpose;
      {
        absl::InlinedVector<int64_t, 4> permutation(dims.size());
        absl::c_iota(permutation, 0);
        absl::MutexLock lock(&transpose_mu_);
        TF_ASSIGN_OR_RETURN(
            transpose, transpose_cache_.GetOrCreate(
                           primitive_util::ByteWidth(type), dims, permutation,
                           TransposePlan::Striding{*byte_strides}));
      }
      transpose->Execute(data, dst_data_ptr);
      if (on_done_with_host_buffer) {
        on_done_with_host_buffer();
        on_done_with_host_buffer = nullptr;
      }
    } else {
      bool should_sync_copy =
          host_buffer_semantics ==
              HostBufferSemantics::kImmutableOnlyDuringCall ||
          (byte_size < kSmallDataTransferByteSize);
      if (should_sync_copy) {
        std::memcpy(dst_data_ptr, data, byte_size);
        if (on_done_with_host_buffer) {
          on_done_with_host_buffer();
          on_done_with_host_buffer = nullptr;
        }
      } else {
        tfrt::AsyncValueRef<CpuEvent> copy_event =
            tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
        definition_events.push_back(copy_event.CopyRef());
        tfrt::EnqueueWork(
            host_ctx_.get(),
            [device_buffer = std::move(device_buffer), dst_data_ptr, data,
             byte_size, copy_event = std::move(copy_event),
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
  }
  auto tracked_device_buffer = std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/false, std::move(buffers), std::move(definition_events),
      std::move(on_delete_callback));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tensorflow::down_cast<TfrtCpuDevice*>(device)));
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
        tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
    definition_events.push_back(definition_event.CopyRef());
    avs.push_back(std::move(definition_event));
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TfrtCpuBuffer> output_buffer,
                      AllocateDestinationBuffer(
                          shape, std::move(definition_events),
                          tensorflow::down_cast<TfrtCpuDevice*>(device), this));

  auto usage_event = tfrt::MakeAvailableAsyncValueRef<CpuEvent>();
  auto* device_buffer = output_buffer->AcquireUsage(std::move(usage_event));
  CHECK(device_buffer);
  if (!shape.IsTuple()) {
    // It is OK to capture `buffer` pointer because the `output_buffer` can't be
    // deleted until all the usage holds have gone away.
    tfrt::EnqueueWork(GetHostContext(), [literal, av = avs[0].CopyRef(),
                                         device_buffer, shape]() mutable {
      tensorflow::profiler::TraceMe traceme("H2D Dispatch");
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
      // It is OK to capture `buffer` pointer because the `output_buffer` can't
      // be deleted until all the usage holds have gone away.
      tfrt::EnqueueWork(GetHostContext(), [i, literal, av = avs[i].CopyRef(),
                                           shape, device_buffer]() mutable {
        tensorflow::profiler::TraceMe traceme("H2D Dispatch");
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

TfrtCpuBuffer::TfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
    TfrtCpuClient* client, TfrtCpuDevice* device)
    : client_(client),
      on_device_shape_(std::move(on_device_shape)),
      device_(device),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {}

TfrtCpuBuffer::~TfrtCpuBuffer() {
  Delete();
  CHECK_EQ(external_reference_counter_, 0);
}

StatusOr<size_t> TfrtCpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtCpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(TfrtCpuBuffer* buffer,
                                     std::shared_ptr<MaybeOwningCpuMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->data();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    TfrtCpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    std::shared_ptr<MaybeOwningCpuMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  ++external_reference_counter_;

  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->Buffers()[0])};
}

class TrackedCpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedCpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->Buffers()[0]->data();
  }

  ~TrackedCpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_;
};

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtCpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedCpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

void TfrtCpuBuffer::CommitDonation() {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
}

void TfrtCpuBuffer::AbortDonation(
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer) {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
  tracked_device_buffer_ = std::move(device_buffer);
}

void TfrtCpuBuffer::Delete() {
  auto device_buffer = ReleaseBufferLocked();
  if (device_buffer == nullptr) return;

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> usage_events =
      device_buffer->LockUseAndTransferUsageEvents();

  std::vector<tfrt::AsyncValue*> event_avs;
  event_avs.reserve(usage_events.size() + 1);
  for (auto& event : usage_events) {
    event_avs.push_back(event.GetAsyncValue());
  }

  // We should also wait for the definition event.
  event_avs.push_back(device_buffer->definition_event().GetAsyncValue());

  tfrt::RunWhenReady(event_avs,
                     [device_buffer = std::move(device_buffer)]() mutable {
                       device_buffer.reset();
                     });
}

bool TfrtCpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

std::unique_ptr<TrackedTfrtCpuDeviceBuffer>
TfrtCpuBuffer::ReleaseBufferLocked() {
  absl::MutexLock lock(&mu_);
  auto condition = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return !pending_donation_;
  };
  mu_.Await(absl::Condition(&condition));
  return std::move(tracked_device_buffer_);
}

StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>> TfrtCpuBuffer::Release(
    bool wait_for_operations_to_complete) {
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer =
      ReleaseBufferLocked();
  if (device_buffer == nullptr) return {nullptr};

  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> events;
  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  events = device_buffer->LockUseAndTransferUsageEvents();

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

  return device_buffer;
}

TrackedTfrtCpuDeviceBuffer* TfrtCpuBuffer::AcquireUsage(
    tfrt::AsyncValueRef<CpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return nullptr;
  }

  tracked_device_buffer_->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return tracked_device_buffer_.get();
}

StatusOr<TfrtCpuBuffer::DonationTransaction> TfrtCpuBuffer::AcquireDonation() {
  absl::MutexLock lock(&mu_);

  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Donation requested for invalid buffer");
  }

  if (external_reference_counter_ > 0) {
    return InvalidArgument(
        "Donation requested for buffer with external reference");
  }

  CHECK(!pending_donation_);
  pending_donation_ = true;

  // Swap out `tracked_device_buffer_` so that no one can acquire a usage event
  // after this point.
  return DonationTransaction(this, std::move(tracked_device_buffer_));
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
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }

  auto usage_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return InvalidArgument(
        "logical_on_device_shape() called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  // Wait for the definition event.
  const auto& av = device_buffer->definition_event();
  client_->GetHostContext()->Await(av.CopyRCRef());
  if (auto* error = av.GetErrorIfPresent()) {
    return InternalError("Error Execute: %s", error->message);
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

static std::vector<tfrt::RCReference<tfrt::AsyncValue>> CopyAsyncValues(
    absl::Span<const tfrt::RCReference<tfrt::AsyncValue>> events) {
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev.CopyRef());
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

PjRtFuture<Status> TfrtCpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuBuffer::ToLiteral");
  if (IsEmptyTuple()) {
    return PjRtFuture<Status>(
        InvalidArgument("ToLiteral called on empty tuple"));
  }
  auto usage_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return PjRtFuture<Status>(InvalidArgument(
        "CopyToHostAsync() called on deleted or donated buffer"));
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  auto host_ctx = client_->GetHostContext();

  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs = {
      device_buffer->definition_event().CopyRCRef()};
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs_copy =
      CopyAsyncValues(device_buffer_wait_avs);

  bool should_sync_copy = device_buffer_wait_avs.empty() &&
                          literal->size_bytes() < kSmallDataTransferByteSize;
  if (should_sync_copy) {
    if (!on_device_shape().IsTuple()) {
      const std::shared_ptr<MaybeOwningCpuMemory>& b =
          device_buffer->Buffers()[0];
      std::memcpy(literal->untyped_data(), b->data(), b->size());
    } else {
      // Tuple case.
      int num_leaves = literal->shape().tuple_shapes().size();
      for (int i = 0; i < num_leaves; ++i) {
        const std::shared_ptr<MaybeOwningCpuMemory>& b =
            device_buffer->Buffers()[i];
        std::memcpy(literal->untyped_data({i}), b->data(), b->size());
      }
    }
    // Unblock ToLiteral caller.
    return PjRtFuture<Status>(OkStatus());
  } else {
    auto ready_event = tfrt::MakeUnconstructedAsyncValueRef<Status>();
    // Wait for buffer definition events to finish before d2h dispatch. D2H
    // dispatch should be in parallel, e.g. one Execute event finish may trigger
    // multiple outputs' D2H, they should happen in different threads in
    // parallel.
    EnqueueWorkWhenReady(
        host_ctx, device_buffer_wait_avs,
        [this, device_buffer_wait_avs = std::move(device_buffer_wait_avs_copy),
         literal, ready_event = ready_event.CopyRef(), device_buffer,
         ready_on_exit = std::move(ready_on_exit)]() mutable {
          tensorflow::profiler::TraceMe traceme("D2H Dispatch");
          // Errors in src buffer are surfaced to user.
          for (const auto& av : device_buffer_wait_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              ready_event.emplace(
                  Internal("Error converting to literal: %s", error->message));
              return;
            }
          }

          if (!on_device_shape().IsTuple()) {
            const std::shared_ptr<MaybeOwningCpuMemory>& b =
                device_buffer->Buffers()[0];
            std::memcpy(literal->untyped_data(), b->data(), b->size());
          } else {
            // Tuple case.
            int num_leaves = literal->shape().tuple_shapes().size();
            for (int i = 0; i < num_leaves; ++i) {
              const std::shared_ptr<MaybeOwningCpuMemory>& b =
                  device_buffer->Buffers()[i];
              std::memcpy(literal->untyped_data({i}), b->data(), b->size());
            }
          }

          // Unblock ToLiteral event.
          ready_event.emplace(OkStatus());
        });
    return PjRtFuture<Status>(
        std::move(ready_event),
        /*on_block_start=*/
        []() {
          tensorflow::profiler::TraceMeProducer traceme(
              "TfrtCpuBuffer::ToLiteral");
          VLOG(1) << "TfrtCpuBuffer::ToLiteral";
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id =*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [](PjRtFutureHelpers::ProfilingKeys keys) {
          tensorflow::profiler::TraceMeConsumer traceme(
              "TfrtCpuBuffer::ToLiteral", keys.traceme_context_id);
        });
  }
}

// TODO(zhangqiaorjc): Consider disallowing multiple CPU devices and assign
// multiple pmap replicas to the same CPU device for multi-CPU pmap testing.
StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuBuffer::CopyToDevice");
  // TODO(zhangqiaorjc): Remove this restriction after removing the test that
  // explicitly asserts this.
  if (dst_device == device_) {
    return InvalidArgument(
        "CopyToDevice cannot accept the same source and destination devices");
  }

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions_size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    return dst_device->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        TfrtCpuClient::HostBufferSemantics::kZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ }, dst_device);
  }

  // Copy each leaf buffer to a destination buffer.
  auto usage_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument("CopyToDevice called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  int num_leaf_buffers = src_device_buffer->Buffers().size();
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> src_buffers;
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> dst_buffers;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> dst_definition_events;
  src_buffers.reserve(num_leaf_buffers);
  dst_buffers.reserve(num_leaf_buffers);
  dst_definition_events.reserve(num_leaf_buffers);

  for (int i = 0; i < num_leaf_buffers; ++i) {
    auto src_buffer = src_device_buffer->Buffers()[i];
    TF_ASSIGN_OR_RETURN(auto dst_buffer, MaybeOwningCpuMemory::AllocateShared(
                                             src_buffer->size()));
    src_buffers.push_back(std::move(src_buffer));
    dst_buffers.push_back(std::move(dst_buffer));
    dst_definition_events.push_back(
        tfrt::MakeConstructedAsyncValueRef<CpuEvent>());
  }

  // Wait for src buffer definition events to finish before d2d dispatch.
  // Errors are propagated asynchronously in dst buffer's definition events.
  const auto& src_definition_event = src_device_buffer->definition_event();

  auto copy_task = [num_leaf_buffers, src_buffers = std::move(src_buffers),
                    dst_buffers_copies = dst_buffers, dst_definition_events,
                    src_definition_event,
                    ready_on_exit = std::move(ready_on_exit)]() mutable {
    tensorflow::profiler::TraceMe traceme("D2D Dispatch");
    if (auto* error = src_definition_event.GetErrorIfPresent()) {
      for (int i = 0; i < num_leaf_buffers; ++i) {
        // Any error discovered in src buffer are propagated to dst buffer
        // definition events, which will surface to users in
        // dst_buffer->ToLiteral().
        dst_definition_events[i].SetError(*error);
      }
      return;
    }

    for (int i = 0; i < num_leaf_buffers; ++i) {
      std::memcpy(dst_buffers_copies[i]->data(), src_buffers[i]->data(),
                  src_buffers[i]->size());
      dst_definition_events[i].SetStateConcrete();
    }
  };

  src_definition_event.AndThen([host_ctx = client()->GetHostContext(),
                                copy_task = std::move(copy_task)]() mutable {
    tfrt::EnqueueWork(host_ctx, std::move(copy_task));
  });

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      on_device_shape_,
      std::make_unique<TrackedTfrtCpuDeviceBuffer>(
          on_device_shape_.IsTuple(), std::move(dst_buffers),
          std::move(dst_definition_events)),
      client(), tensorflow::down_cast<TfrtCpuDevice*>(dst_device)));
}

PjRtFuture<Status> TfrtCpuBuffer::GetReadyFuture() {
  tfrt::AsyncValueRef<CpuEvent> definition_event;
  {
    absl::MutexLock lock(&mu_);
    if (!tracked_device_buffer_) {
      return PjRtFuture<Status>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    definition_event = tracked_device_buffer_->definition_event();
  }
  DCHECK(definition_event);

  if (definition_event.IsAvailable()) {
    if (definition_event.IsError()) {
      return PjRtFuture<Status>(FailedPrecondition(
          "Buffer Definition Event: %s", definition_event.GetError().message));
    }
    return PjRtFuture<Status>(OkStatus());
  } else {
    tfrt::AsyncValueRef<Status> status_event =
        tfrt::MakeUnconstructedAsyncValueRef<Status>();

    definition_event.AndThen(
        [definition_event = definition_event.AsPtr(), status_event]() {
          if (definition_event.IsError()) {
            status_event.emplace(
                FailedPrecondition("Buffer Definition Event: %s",
                                   definition_event.GetError().message));
          } else {
            status_event.emplace(OkStatus());
          }
        });

    return PjRtFuture<Status>(
        std::move(status_event),
        /*on_block_start=*/
        []() {
          tensorflow::profiler::TraceMeProducer traceme("TfrtCpuBuffer::Await");
          VLOG(1) << "TfrtCpuBuffer::Await";
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id=*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [](PjRtFutureHelpers::ProfilingKeys keys) {
          tensorflow::profiler::TraceMeConsumer traceme(
              "TfrtCpuBuffer::Await", keys.traceme_context_id);
        });
  }
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
      addressable_devices_(std::move(addressable_devices)) {
  auto hlo_cost_analysis =
      std::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
  // Cache to avoid std::map lookup in flop_count() on critical path.
  // The magic constant 1000 is determined by correlating computation with flop
  // estimate. It is a crude heuristic to find computation less than the thread
  // context switch time (~5us).
  cheap_computation_ = hlo_cost_analysis->flop_count() < 1000;

  const auto& computation_layout =
      cpu_executable_->module().entry_computation_layout();
  if (computation_layout.parameter_count() == 0) {
    return;
  }
  // Assume compiled program expects either many non-tupled arguments or a
  // singled tupled argument. Nested tuple is not yet supported.
  if (computation_layout.parameter_count() > 1 ||
      !computation_layout.parameter_shape(0).IsTuple()) {
    input_buffer_sizes_in_bytes_.reserve(computation_layout.parameter_count());
    for (int i = 0; i < computation_layout.parameter_count(); ++i) {
      input_buffer_sizes_in_bytes_.push_back(
          ShapeUtil::ByteSizeOf(computation_layout.parameter_shape(i)));
    }
  } else {
    input_buffer_sizes_in_bytes_.reserve(
        computation_layout.parameter_shape(0).tuple_shapes_size());
    for (int i = 0;
         i < computation_layout.parameter_shape(0).tuple_shapes_size(); ++i) {
      input_buffer_sizes_in_bytes_.push_back(ShapeUtil::ByteSizeOf(
          computation_layout.parameter_shape(0).tuple_shapes(i)));
    }
  }
}

void TfrtCpuExecutable::Delete() {}

bool TfrtCpuExecutable::IsDeleted() { return false; }

StatusOr<std::optional<std::string>> TfrtCpuExecutable::Fingerprint() const {
  return std::optional<std::string>();
}

Status TfrtCpuExecutable::SetUpDonation(bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(parameters_that_must_be_donated_,
                      ComputeParametersThatMustBeDonated(
                          *cpu_executable_->shared_module(), tuple_inputs));
  return OkStatus();
}

// The following few helpers are adapted from XLA:CPU to create a buffer table
// and assemble the buffer pointers in order to call into CpuExecutable.
static StatusOr<std::shared_ptr<MaybeOwningCpuMemory>> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<TrackedTfrtCpuDeviceBuffer* const> arguments) {
  if (allocation.is_entry_computation_parameter()) {
    TrackedTfrtCpuDeviceBuffer* arg = arguments[allocation.parameter_number()];
    std::shared_ptr<MaybeOwningCpuMemory> out =
        arg->Buffer(allocation.param_shape_index());
    CHECK_EQ(allocation.size(), out->size())
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();
    return out;
  } else if (allocation.is_constant()) {
    return std::make_shared<MaybeOwningCpuMemory>();
  } else if (allocation.is_thread_local()) {
    return std::make_shared<MaybeOwningCpuMemory>();
  }

  // Output and temporary buffer.
  int64_t buffer_size = allocation.size();
  TF_ASSIGN_OR_RETURN(auto out,
                      MaybeOwningCpuMemory::AllocateShared(buffer_size));

  // Since the output buffer and all the temporary buffers were written into
  // by the JITed code, msan has no way of knowing their memory was
  // initialized. Mark them initialized so that msan doesn't flag loads from
  // these buffers.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->data(), buffer_size);
  return out;
}

static StatusOr<std::vector<std::shared_ptr<MaybeOwningCpuMemory>>>
CreateBufferTable(const BufferAssignment& assignment,
                  absl::Span<TrackedTfrtCpuDeviceBuffer* const> arguments) {
  std::vector<std::shared_ptr<MaybeOwningCpuMemory>> buffers(
      assignment.Allocations().size());
  for (BufferAllocation::Index i = 0; i < assignment.Allocations().size();
       ++i) {
    const BufferAllocation& allocation = assignment.GetAllocation(i);
    TF_ASSIGN_OR_RETURN(buffers[i], MemoryForAllocation(allocation, arguments));
  }
  return std::move(buffers);
}

static absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4>
CreateResultShapedBuffer(
    absl::Span<const BufferAllocation::Index> buffer_indices,
    absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> buffer_table,
    absl::Span<TrackedTfrtCpuDeviceBuffer* const> arguments) {
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> output_buffers;
  output_buffers.reserve(buffer_indices.size());
  for (int i = 0; i < buffer_indices.size(); ++i) {
    output_buffers.push_back(buffer_table[buffer_indices[i]]);
  }
  return output_buffers;
}

Status TfrtCpuExecutable::CheckBufferCompatibilities(
    absl::Span<TrackedTfrtCpuDeviceBuffer* const> input_buffers) const {
  if (input_buffers.size() != input_buffer_sizes_in_bytes_.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes_.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i];
    if (input_buffer_sizes_in_bytes_[i] != buffer->Buffers()[0]->size()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with "
          "incompatible size %lld",
          i, input_buffer_sizes_in_bytes_[i], buffer->Buffers()[0]->size());
    }
  }
  return OkStatus();
}

StatusOr<PjRtLoadedExecutable::Result> TfrtCpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event,
    bool fill_future, TfrtCpuDevice* device) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuExecutable::ExecuteHelper");
  auto* host_context = client_->GetHostContext();

  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(device_id));
    device = tensorflow::down_cast<TfrtCpuDevice*>(pjrt_device);
    device_assignment = device_assignment_;
  } else {
    CHECK(device_assignment_ == nullptr);
    CHECK_EQ(replica, 0);
    CHECK_EQ(partition, 0);
    CHECK(addressable_devices_.empty());
    device_assignment = std::make_shared<DeviceAssignment>(1, 1);
    (*device_assignment)(0, 0) = device->id();
  }
  CHECK_EQ(device->process_index(), client_->process_index());

  // Handle inputs.
  if (options.arguments_are_tupled) {
    if (!parameter_is_tupled_arguments_) {
      return InvalidArgument(
          "Arguments may only be supplied as a tuple when the executable was "
          "compiled with a single tupled parameter");
    }
    if (argument_handles.size() != 1) {
      return InvalidArgument(
          "Option arguments_are_tupled was true but %d buffers were passed to "
          "execution",
          argument_handles.size());
    }
  }

  // `execute_event` indicates whether cpu computation is complete and whether
  // there was an error.
  auto execute_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
  MarkEventReadyOnExit ready_on_exit(execute_event);

  absl::InlinedVector<TfrtCpuBuffer::DonationTransaction, 4>
      donation_transactions;
  absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> tracked_buffers;
  tracked_buffers.reserve(argument_handles.size());
  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as the
  // single definition event.
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> input_deps;
  input_deps.reserve(argument_handles.size());

  auto donate_it = parameters_that_must_be_donated_.begin();

  for (int i = 0; i < argument_handles.size(); ++i) {
    PjRtBuffer* handle = argument_handles[i];
    auto* tfrt_buffer = tensorflow::down_cast<TfrtCpuBuffer*>(handle);
    if (tfrt_buffer->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, tfrt_buffer->device()->DebugString(),
          device->DebugString());
    }

    bool must_donate =
        donate_it != parameters_that_must_be_donated_.end() && *donate_it == i;
    TrackedTfrtCpuDeviceBuffer* tracked_buffer = nullptr;
    if (must_donate) {
      ++donate_it;
      TF_ASSIGN_OR_RETURN(auto donation_transaction,
                          tfrt_buffer->AcquireDonation());

      // After acquiring the buffer for donation, we retrieve the dependent
      // usage events. Note that we don't need any locking here as
      // AcquireDonation() is supposed to synchronize with other usages.
      for (const auto& ev :
           donation_transaction.device_buffer()->UsageEvents()) {
        if (!ev.IsAvailable()) {
          input_deps.push_back(ev.CopyRCRef());
        }
      }
      tracked_buffer = donation_transaction.device_buffer();
      tracked_buffers.push_back(tracked_buffer);
      donation_transactions.push_back(std::move(donation_transaction));

    } else {
      tracked_buffer = tfrt_buffer->AcquireUsage(execute_event);
      if (!tracked_buffer)
        return InvalidArgument(
            "Invalid buffer passed: buffer has been deleted or donated.");
      tracked_buffers.push_back(tracked_buffer);
    }

    // Definition events are never modified after buffer construction.
    const auto& definition_event = tracked_buffer->definition_event();
    if (!definition_event.IsAvailable()) {
      input_deps.push_back(definition_event.CopyRCRef());
    }
  }

  TF_RETURN_IF_ERROR(CheckBufferCompatibilities(tracked_buffers));

  // Tuplize the inputs if compiler expects a single tuple argument but runtime
  // gets many inputs that are not yet tupled.
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tuplized_arg;
  if (parameter_is_tupled_arguments_ && !options.arguments_are_tupled) {
    absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> leaf_buffers;
    leaf_buffers.reserve(tracked_buffers.size());
    for (const auto& tracked_buffer : tracked_buffers) {
      auto span = tracked_buffer->Buffers();
      leaf_buffers.insert(leaf_buffers.end(), span.begin(), span.end());
    }

    // Tuplize into a single input.
    tracked_buffers.clear();
    tuplized_arg = std::make_unique<TrackedTfrtCpuDeviceBuffer>(
        /*is_tuple=*/true, std::move(leaf_buffers),
        /*definition_event=*/tfrt::MakeAvailableAsyncValueRef<CpuEvent>());
    tracked_buffers.push_back(tuplized_arg.get());
  }

  auto* cpu_executable =
      tensorflow::down_cast<cpu::CpuExecutable*>(cpu_executable_.get());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<MaybeOwningCpuMemory>> buffer_table,
      CreateBufferTable(cpu_executable->buffer_assignment(), tracked_buffers));
  auto result_buffers = CreateResultShapedBuffer(result_buffer_indices_,
                                                 buffer_table, tracked_buffers);

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  auto compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
      device->max_inflight_computations_semaphore().ScopedAcquire(1));

  // Call the computation function following the calling convention.
  std::vector<void*> buffer_pointers;
  buffer_pointers.reserve(buffer_table.size());
  for (const auto& buffer : buffer_table) {
    buffer_pointers.push_back(buffer->data());
  }
  void* result_buffer = buffer_pointers[result_buffer_index_];

  ExecutableRunOptions run_options;
  run_options.set_run_id(run_id);
  run_options.set_device_ordinal(device->local_hardware_id());
  // Need to keep device_assignment alive until execution completes.
  run_options.set_device_assignment(device_assignment.get());
  run_options.set_intra_op_thread_pool(client_->eigen_intraop_device());

  // Schedule only one collective at a time.
  bool is_a_collective_launch = !!last_collective_launch_event;
  if (is_a_collective_launch) {
    input_deps.push_back(std::move(last_collective_launch_event));
  }

  bool execute_inline = cheap_computation_;

  // Overwrite `execute_inline` if it is specified in the ExecuteOptions.
  if (options.execution_mode == ExecuteOptions::ExecutionMode::kAsynchronous) {
    execute_inline = false;
  } else if (options.execution_mode ==
             ExecuteOptions::ExecutionMode::kSynchronous) {
    execute_inline = true;
  }

  if (input_deps.empty() && execute_inline) {
    // Synchronously call generated function.

    // Set denormal and rounding behavior to match the default TF
    // ThreadPool behavior.
    tensorflow::port::ScopedFlushDenormal flush;
    tensorflow::port::ScopedSetRound round(FE_TONEAREST);

    XlaCustomCallStatus status;

    // Call generated function.
    cpu_executable->compute_function()(result_buffer, &run_options, nullptr,
                                       buffer_pointers.data(), &status,
                                       nullptr);

    for (auto& donation_transaction : donation_transactions) {
      std::move(donation_transaction).Commit();
    }

    std::optional<absl::string_view> error_message =
        xla::CustomCallStatusGetMessage(&status);
    if (error_message) {
      return InternalError("Generated function failed: %s", *error_message);
    }

  } else {
    // TODO(zhangqiaorjc): Only async launch expensive computations. Need
    // heuristics to decide what computation is expensive.
    // Asynchronously call generated function.

    // We only created enough threads for one collective to complete.
    // The next collective launch will not be scheduled onto threadpool until
    // this one completes.
    if (is_a_collective_launch) {
      client_->SetLastCollectiveLaunchEvent(execute_event.CopyRef());
    }
    std::vector<tfrt::RCReference<tfrt::AsyncValue>> input_deps_avs_copy =
        CopyAsyncValues(input_deps);
    EnqueueWorkWhenReady(
        host_context, input_deps,
        [cpu_executable, result_buffer,
         buffer_pointers = std::move(buffer_pointers),
         buffer_table = std::move(buffer_table),
         run_options = std::move(run_options),
         cpu_executable_copy = cpu_executable_,
         device_assignment = std::move(device_assignment),
         compute_reservation = std::move(compute_reservation),
         tuplized_arg = std::move(tuplized_arg),
         donation_transactions = std::move(donation_transactions),
         execute_event = std::move(ready_on_exit).Release(),
         input_deps_avs = std::move(input_deps_avs_copy)]() mutable {
          for (const auto& av : input_deps_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              execute_event.SetError(absl::StrCat(
                  "Error dispatching computation: %s", error->message));
              return;
            }
          }

          // Set denormal and rounding behavior to match the default TF
          // ThreadPool behavior.
          tensorflow::port::ScopedFlushDenormal flush;
          tensorflow::port::ScopedSetRound round(FE_TONEAREST);

          XlaCustomCallStatus status;

          // Call generated function.
          cpu_executable->compute_function()(result_buffer, &run_options,
                                             nullptr, buffer_pointers.data(),
                                             &status, nullptr);

          std::optional<absl::string_view> error_message =
              xla::CustomCallStatusGetMessage(&status);

          for (auto& donation_transaction : donation_transactions) {
            std::move(donation_transaction).Commit();
          }

          if (error_message) {
            // CPU computation fails with an error.
            execute_event.SetError(absl::StrFormat(
                "Generated function failed: %s", *error_message));
            return;
          }

          // CPU computation completes.
          execute_event.SetStateConcrete();
        });
  }

  // Create output TFRT buffers.
  const Shape& result_shape = cpu_executable_->result_shape();
  std::vector<std::unique_ptr<PjRtBuffer>> res;
  if (options.untuple_result && result_shape.IsTuple()) {
    res.reserve(result_buffers.size());
    for (int i = 0; i < result_buffers.size(); ++i) {
      absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> sub_buffer;
      sub_buffer.push_back(std::move(result_buffers[i]));
      // Program execution writes to output buffers so it's a definition event.
      absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
      definition_events.push_back(execute_event.CopyRef());
      auto leaf_tracked_device_buffer =
          std::make_unique<TrackedTfrtCpuDeviceBuffer>(
              /*is_tuple=*/false, std::move(sub_buffer),
              std::move(definition_events));
      auto leaf_buffer = std::make_unique<TfrtCpuBuffer>(
          result_shape.tuple_shapes(i), std::move(leaf_tracked_device_buffer),
          client_, device);
      res.push_back(std::move(leaf_buffer));
    }
  } else {
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedTfrtCpuDeviceBuffer>(
        /*is_tuple=*/result_shape.IsTuple(), std::move(result_buffers),
        /*definition_event=*/execute_event);
    auto tfrt_output_buffer = std::make_unique<TfrtCpuBuffer>(
        result_shape, std::move(tracked_device_buffer), client_, device);
    res.push_back(std::move(tfrt_output_buffer));
  }
  std::optional<PjRtFuture<Status>> future;
  if (fill_future) {
    auto done_event = tfrt::MakeUnconstructedAsyncValueRef<Status>();
    execute_event.AndThen(
        [done_event = done_event.CopyRef(), event = execute_event.CopyRef()]() {
          Status s;
          if (auto* error = event.GetErrorIfPresent()) {
            s = InternalError("Compute error: %s", error->message);
          }
          done_event.emplace(std::move(s));
        });
    future = PjRtFuture<Status>(std::move(done_event));
  }
  return Result({/*future=*/std::move(future), /*buffers=*/std::move(res)});
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtCpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuExecutable::Execute");
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }

  RunId run_id;
  tensorflow::profiler::TraceMeProducer activity(
      "TfrtCpuExecutable::Execute", tensorflow::profiler::ContextType::kPjRt,
      run_id.ToInt());

  const int num_addressable_devices = addressable_devices_.size();

  if (argument_handles.size() != num_addressable_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_addressable_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation " << name()
          << "; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions()
          << " num_addressable_devices=" << num_addressable_devices;

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(
      num_addressable_devices);
  if (returned_futures.has_value()) {
    returned_futures->resize(num_addressable_devices);
  }
  if (num_addressable_devices == 1) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = addressable_device_logical_ids_[0].replica;
    const int partition = addressable_device_logical_ids_[0].partition;

    auto statusor = ExecuteHelper(
        argument_handles[0], replica, partition, run_id, options,
        /*last_collective_launch_event=*/tfrt::AsyncValueRef<CpuEvent>(),
        returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

  } else {
    // Gang schedule collectives to ensure that collectives with the same RunId
    // are run at the same time. We conservatively run only one collective at a
    // time, because we may not have enough threads to run arbitrary number of
    // collectives concurrently.
    tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event =
        client_->GetLastCollectiveLaunchEvent();

    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;
      tfrt::EnqueueWork(client_->GetHostContext(), [&, replica, partition, i] {
        auto statusor =
            ExecuteHelper(argument_handles[i], replica, partition, run_id,
                          options, last_collective_launch_event.CopyRef(),
                          returned_futures.has_value());
        if (statusor.ok()) {
          wrapped_results[i] = std::move(statusor->buffers);
          if (returned_futures.has_value()) {
            (*returned_futures)[i] = std::move(*statusor->future);
          }
        }

        absl::MutexLock lock(&mu);
        --running;
        if (!statusor.ok()) {
          if (failed == 0) {
            first_failure_status = AppendStatus(
                std::move(statusor).status(),
                absl::StrFormat(
                    "while running replica %d and partition %d of a "
                    "replicated computation (other "
                    "replicas may have failed as well).",
                    replica, partition));
          }
          ++failed;
        }
      });
    }

    {
      auto done_running = [&]() {
        mu.AssertHeld();
        return running == 0;
      };
      absl::MutexLock lock(&mu);
      mu.Await(absl::Condition(&done_running));
    }

    if (!first_failure_status.ok()) return first_failure_status;
  }
  VLOG(1) << "Replicated execution complete.";

  return wrapped_results;
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtCpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuExecutable::ExecuteSharded");
  if (device_assignment_ == nullptr) {
    return InvalidArgument("ExecuteShard expects a non-null device_assignment");
  }
  for (int i = 0; i < addressable_devices_.size(); ++i) {
    if (addressable_devices_[i] == device) {
      VLOG(1) << "ExecuteShard executes computation " << name()
              << " on assigned replica/partition on device "
              << device->DebugString();
      TF_ASSIGN_OR_RETURN(
          auto result,
          ExecuteHelper(
              argument_handles, addressable_device_logical_ids_[i].replica,
              addressable_device_logical_ids_[i].partition, RunId(), options,
              /*last_collective_launch_event=*/
              tfrt::AsyncValueRef<CpuEvent>(), fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->id());
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtCpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  tensorflow::profiler::TraceMe traceme("TfrtCpuExecutable::ExecutePortable");
  if (device_assignment_ != nullptr) {
    return InvalidArgument("ExecutePortable gets a non-portable executable");
  }
  if (num_replicas() != 1 || num_partitions() != 1) {
    return InvalidArgument(
        "ExecutePortable expects a single-core executable but gets "
        "one with %d replica %d partition",
        num_replicas(), num_partitions());
  }
  if (device == nullptr) {
    return InvalidArgument("ExecutePortable expects a device to be specified");
  }
  VLOG(1) << "ExecutePortable executes single-core portable executable "
          << name();
  TF_ASSIGN_OR_RETURN(
      auto result,
      ExecuteHelper(
          argument_handles,
          /*replica=*/0,
          /*partition=*/0, RunId(), options,
          /*last_collective_launch_event=*/tfrt::AsyncValueRef<CpuEvent>(),
          fill_future, tensorflow::down_cast<TfrtCpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}
}  // namespace xla
