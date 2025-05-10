/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/cpu_client.h"

#include <tuple>

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cfenv>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/array.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/client/executable_build_options.h"
#include "xla/cpu_function_runtime.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/cpu/abstract_cpu_buffer.h"
#include "xla/pjrt/cpu/cpu_async_execution_tracker.h"
#include "xla/pjrt/cpu/cpu_device.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/host_to_device_transfer_manager.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_client_utils.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_execute_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_status_internal.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_value.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/denormal.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/setround.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace {

// Converts the shape used to represent the host buffer to the shape used to
// represent the on-device buffer.
Shape HostShapeToOnDeviceShape(const Shape& shape) {
  // AbstractCpuBuffer packs sub-byte non-pred types. The on-device shape
  // should reflect this so that our memory allocation and overflow checks are
  // correct.
  if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
    Shape on_device_shape = shape;
    on_device_shape.mutable_layout()->set_element_size_in_bits(
        primitive_util::BitWidth(shape.element_type()));
    return on_device_shape;
  }

  // Otherwise, return the shape as-is.
  return shape;
}

absl::StatusOr<std::unique_ptr<TfrtCpuBuffer>> AllocateDestinationBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events,
    TfrtCpuDevice* device, TfrtCpuClient* client) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer,
      AbstractCpuBuffer::AllocateTrackedDeviceBuffer(
          on_device_shape, std::move(definition_events)));
  return std::make_unique<TfrtCpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer), client, device,
      *device->default_memory_space());
}

absl::StatusOr<std::unique_ptr<TfrtCpuBuffer>> AllocateDestinationBufferAndAvs(
    const Shape& on_device_shape,
    absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
    TfrtCpuDevice* device, TfrtCpuClient* client) {
  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events;
  AbstractCpuBuffer::AllocateAvsAndEvents(on_device_shape, avs,
                                          &definition_events);
  return AllocateDestinationBuffer(
      on_device_shape, std::move(definition_events),
      tensorflow::down_cast<TfrtCpuDevice*>(device), client);
}

void EnqueueWork(tsl::thread::ThreadPool* pool,
                 absl::AnyInvocable<void() &&> callee) {
  // TSL TheadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool->Schedule(
      [ptr = new absl::AnyInvocable<void() &&>(std::move(callee))]() {
        std::move (*ptr)();
        delete ptr;
      });
}

// Enqueue to PjRtClient pool when all `values` are ready.
void EnqueueWorkWhenReady(
    tsl::thread::ThreadPool* pool,
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
    absl::AnyInvocable<void() &&> callee) {
  RunWhenReady(values, [pool, callee = std::move(callee)]() mutable {
    EnqueueWork(pool, std::move(callee));
  });
}

class ThreadPoolAsyncWorkRunner : public AsyncWorkRunner {
 public:
  explicit ThreadPoolAsyncWorkRunner(tsl::thread::ThreadPool* pool)
      : pool_(pool) {}

  void Schedule(absl::AnyInvocable<void() &&> work) override {
    EnqueueWork(pool_, std::move(work));
  }

  void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void() &&> work) override {
    EnqueueWorkWhenReady(pool_, values, std::move(work));
  }

 private:
  tsl::thread::ThreadPool* pool_;
};

}  // namespace

static int CpuDeviceCount() {
  // By default we fix the number of devices to one.  However we do let the user
  // override this behavior to help run tests on the host that run models in
  // parallel across multiple devices, e.g. pmap.
  return GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(
    CpuClientOptions options) {
  // Need at least CpuDeviceCount threads to launch one collective.
  int cpu_device_count = options.cpu_device_count.value_or(CpuDeviceCount());
  size_t num_threads = std::max(DefaultThreadPoolSize(), cpu_device_count);

  std::vector<std::unique_ptr<TfrtCpuDevice>> devices;
  for (int i = 0; i < cpu_device_count; ++i) {
    auto device = std::make_unique<TfrtCpuDevice>(
        options.process_id, /*local_device_id=*/i,
        options.max_inflight_computations_per_device);
    devices.push_back(std::move(device));
  }

  return std::unique_ptr<PjRtClient>(new TfrtCpuClient(
      options.process_id, std::move(devices), std::move(options.collectives),
      num_threads, options.asynchronous, options.legacy_memory_space_behavior,
      std::move(options.customize_hlo_module_config)));
}

// An upper bound on the number of threads to use for intra-op parallelism. It
// is nearly impossible to utilize efficiently more than 256 threads for compute
// intensive operations that are supposed to run inside the intra-op threadpool.
static const size_t kMaxIntraOpThreads = 256;

static tsl::ThreadOptions GetThreadOptions() {
  tsl::ThreadOptions thread_options;
  // On Mac OS the default stack size is 512KiB, which is too small for some
  // BLAS and LAPACK functions (https://github.com/google/jax/issues/20428).
  // On Linux we also observed that 2MB wasn't enough to run some OpenBLAS
  // functions.
  thread_options.stack_size = 8 * 1024 * 1024;
  return thread_options;
}

// Returns the CPU devices from the given TfrtCpuDevices.
// Precondition: `devices` doesn't contain nullptr.
static std::vector<CpuTopology::CpuDevice> GetCpuDevices(
    absl::Span<const std::unique_ptr<TfrtCpuDevice>> devices) {
  std::vector<CpuTopology::CpuDevice> cpu_devices;
  cpu_devices.reserve(devices.size());
  for (const auto& device : devices) {
    cpu_devices.push_back(CpuTopology::CpuDevice{
        device->process_index(), device->local_hardware_id().value()});
  }
  return cpu_devices;
}

TfrtCpuClient::TfrtCpuClient(
    int process_index, std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
    std::shared_ptr<cpu::CpuCollectives> collectives, size_t num_threads,
    bool asynchronous, bool legacy_memory_space_behavior,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      eigen_intraop_pool_(new tsl::thread::ThreadPool(
          tsl::Env::Default(), GetThreadOptions(), "XLAEigen",
          std::min(num_threads, kMaxIntraOpThreads))),
      eigen_intraop_device_(
          new Eigen::ThreadPoolDevice(eigen_intraop_pool_->AsEigenThreadPool(),
                                      eigen_intraop_pool_->NumThreads())),
      pjrt_client_thread_pool_(
          new tsl::thread::ThreadPool(tsl::Env::Default(), GetThreadOptions(),
                                      "XLATfrtCpuClient", num_threads)),
      async_work_runner_(std::make_unique<ThreadPoolAsyncWorkRunner>(
          pjrt_client_thread_pool_.get())),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<CpuEvent>()),
      transpose_cache_(1024),
      collectives_(std::move(collectives)),
      topology_(platform_id(), platform_name(), platform_version(),
                GetCpuDevices(owned_devices_), cpu::DetectMachineAttributes()),
      asynchronous_(asynchronous),
      customize_hlo_module_config_(std::move(customize_hlo_module_config)) {
  for (const std::unique_ptr<TfrtCpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(
        id_to_device_.insert({device->global_device_id(), device.get()}).second)
        << "Duplicate device id: " << device->global_device_id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id().value();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }
  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    auto* const device = addressable_devices_[idx];
    CHECK(device != nullptr) << idx;
    auto* cpu_device = tensorflow::down_cast<TfrtCpuDevice*>(device);

    // Use the device id to construct a globally unique memory space id.
    const int id = device->id();

    if (legacy_memory_space_behavior) {
      auto memory_space = std::make_unique<UnpinnedHostMemorySpace>(id, device);
      cpu_device->AttachMemorySpace(memory_space.get());
      memory_spaces_.push_back(memory_space.get());
      owned_memory_spaces_.push_back(std::move(memory_space));
    } else {
      // The first attached memory space is returned as the default by
      // TfrtCpuDevice, so attach the device memory space first.
      auto cpu_device_memory_space =
          std::make_unique<CpuDeviceMemorySpace>(id * 3 + 0, device);
      cpu_device->AttachMemorySpace(cpu_device_memory_space.get());
      memory_spaces_.push_back(cpu_device_memory_space.get());
      owned_memory_spaces_.push_back(std::move(cpu_device_memory_space));

      auto unpinned_memory_space =
          std::make_unique<UnpinnedHostMemorySpace>(id * 3 + 1, device);
      cpu_device->AttachMemorySpace(unpinned_memory_space.get());
      memory_spaces_.push_back(unpinned_memory_space.get());
      owned_memory_spaces_.push_back(std::move(unpinned_memory_space));

      auto pinned_memory_space =
          std::make_unique<PinnedHostMemorySpace>(id * 3 + 2, device);
      cpu_device->AttachMemorySpace(pinned_memory_space.get());
      memory_spaces_.push_back(pinned_memory_space.get());
      owned_memory_spaces_.push_back(std::move(pinned_memory_space));
    }
  }
  VLOG(1) << "TfrtCpuClient created.";
}

TfrtCpuClient::~TfrtCpuClient() { VLOG(1) << "TfrtCpuClient destroyed."; }

absl::StatusOr<PjRtDevice*> TfrtCpuClient::LookupDevice(
    xla::PjRtGlobalDeviceId global_device_id) const {
  auto it = id_to_device_.find(global_device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         global_device_id.value());
}

absl::StatusOr<PjRtDevice*> TfrtCpuClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_device_id %d",
                         local_device_id.value());
}

absl::Span<PjRtMemorySpace* const> TfrtCpuClient::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<DeviceAssignment> TfrtCpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  if (num_partitions * num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, num_partitions);
    for (int i = 0; i < num_replicas; ++i) {
      for (int j = 0; j < num_partitions; ++j) {
        assignment(i, j) =
            addressable_devices().at(i * num_partitions + j)->id();
      }
    }
    return assignment;
  }
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

absl::StatusOr<Layout> TfrtCpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  return LayoutUtil::GetWithDefaultLayout(shape).layout();
}

absl::StatusOr<std::unique_ptr<HloCostAnalysis>>
TfrtCpuClient::GetHloCostAnalysis() const {
  return std::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
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
static absl::StatusOr<absl::InlinedVector<BufferAllocation::Index, 4>>
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
  buffer_indices.reserve(result_shape.tuple_shapes().size());
  for (int i = 0; i < result_shape.tuple_shapes().size(); ++i) {
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

absl::StatusOr<std::string> TfrtCpuExecutable::SerializeExecutable() const {
  cpu::CpuCompiler compiler;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler.Export(cpu_executable_.get()));

  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  if (serialized.empty()) {
    return Internal(
        "TfrtCpuClient::SerializeExecutable proto serialization failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtCpuClient::LoadSerializedExecutable(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  ExecutableAndOptionsProto proto;
  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal(
        "TfrtCpuClient::DeserializeExecutable proto too large (>2GB)");
  }
  if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
    return Internal(
        "TfrtCpuClient::DeserializeExecutable proto deserialization failed");
  }
  CompileOptions compile_options;
  if (options.has_value()) {
    compile_options = *std::move(options);
  } else {
    TF_ASSIGN_OR_RETURN(compile_options,
                        CompileOptions::FromProto(proto.compile_options()));
  }
  auto input_options = compile_options;
  // Load a CpuExecutable
  cpu::CpuCompiler compiler;
  std::string str = std::move(*proto.mutable_serialized_executable());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler.LoadAotCompilationResult(str));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result).LoadExecutable(&compiler, /*executor=*/nullptr));

  // Set up other arguments for TfrtCpuExecutable
  // TODO(b/232263665): Remove duplicated code in DeserializeExecutable and
  // Compile.
  int num_replicas;
  int num_partitions;
  std::shared_ptr<DeviceAssignment> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      compile_options.compile_portable_executable,
      &compile_options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  auto cpu_executable_ptr =
      tensorflow::down_cast<cpu::CpuExecutable*>(executable.get());

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
                                      executable->module()));

  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  ExecutableBuildOptions& build_options =
      compile_options.executable_build_options;
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        PjRtGlobalDeviceId device_id((*device_assignment)(replica, partition));
        if (UnpackCpuProcessIndex(device_id) != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }

    if (!addressable_devices.empty() && build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id().value());
    }
  }

  auto tfrt_cpu_executable = std::make_unique<TfrtCpuExecutable>(
      num_replicas, num_partitions, std::move(device_assignment),
      compile_options.parameter_is_tupled_arguments, std::move(input_options),
      std::move(executable), result_slice.index(),
      std::move(result_buffer_indices),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);
  TF_RETURN_IF_ERROR(tfrt_cpu_executable->SetUpDonation(
      compile_options.parameter_is_tupled_arguments));

  return std::unique_ptr<PjRtLoadedExecutable>(std::move(tfrt_cpu_executable));
}

static absl::StatusOr<std::unique_ptr<xla::Executable>> JitCompile(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options,
    const xla::Compiler::CompileOptions& compile_options, int num_threads,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_layouts, &execution_options,
                         execution_options.num_replicas(), num_threads,
                         /*aot_options=*/nullptr));

  // Apply the user-provided callback to customize the HloModuleConfig.
  if (customize_hlo_module_config) {
    customize_hlo_module_config(*hlo_module_config);
  }

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));
  VLOG(3) << "Unoptimized HLO module: " << hlo_module->ToString();
  static constexpr char kBeforeOptimizationsDumpName[] = "before_optimizations";
  DumpHloModuleIfEnabled(*hlo_module, kBeforeOptimizationsDumpName);

  // RunHloPasses and RunBackend both look at the LLVM command line options.
  auto llvm_options = llvm_ir::ExtractXlaBackendExtraOptions(
      hlo_module->config().debug_options().xla_backend_extra_options());
  llvm_ir::LLVMCommandLineOptionsLock llvm_lock(llvm_options);

  // Run Hlo Passes
  cpu::CpuCompiler compiler;
  if (!build_options.run_backend_only()) {
    TF_ASSIGN_OR_RETURN(
        hlo_module,
        compiler.RunHloPasses(std::move(hlo_module),
                              /*stream_exec=*/nullptr, compile_options));
  }

  // Run backend.
  return compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr,
                             compile_options);
}

static absl::StatusOr<std::unique_ptr<xla::Executable>> CompileAheadOfTime(
    const XlaComputation& computation,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options,
    const xla::AotCompilationOptions& compile_options, int num_threads,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config) {
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_layouts, &execution_options,
                         execution_options.num_replicas(), num_threads,
                         /*aot_options=*/&compile_options));

  // Apply the user-provided callback to customize the HloModuleConfig.
  if (customize_hlo_module_config) {
    customize_hlo_module_config(*hlo_module_config);
  }

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));

  cpu::CpuCompiler compiler;
  // TODO (basioli): honor build_options.run_backend_only() for AOT.

  auto hlo_module_group =
      std::make_unique<HloModuleGroup>(std::move(hlo_module));

  // Compile AOT.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler.CompileAheadOfTime(std::move(hlo_module_group),
                                  compile_options));

  if (aot_results.size() != 1) {
    return Internal("Expected 1 AOT compilation result, got %d.",
                    aot_results.size());
  }

  // Technically not needed, but it makes sense so that we know serialization
  // and deserialization works.
  TF_ASSIGN_OR_RETURN(std::string serialized_aot_result,
                      aot_results[0]->SerializeAsString());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler.LoadAotCompilationResult(serialized_aot_result));

  return std::move(*aot_result).LoadExecutable(&compiler, /*executor=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtCpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  XlaComputation xla_computation;
  const ExecutableBuildOptions& exec_build_options =
      options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, exec_build_options.use_shardy_partitioner()));

  if (options.argument_layouts) {
    return CompileAndLoad(xla_computation, options);
  }

  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                      GetOutputLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                      GetArgMemoryKinds(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                      GetOutputMemoryKinds(module));

  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [&arg_layout_modes, &out_layout_modes,
                          &arg_memory_spaces,
                          &out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces, &LayoutUtil::GetWithDefaultLayout);
  };

  // This call will update result_layout in options.executable_build_options.
  TF_ASSIGN_OR_RETURN(
      auto arg_layouts_and_pointers,
      LayoutModesToXla(xla_computation, arg_layout_modes, out_layout_modes,
                       arg_memory_spaces, out_memory_spaces,
                       &LayoutUtil::GetWithDefaultLayout,
                       options.executable_build_options));

  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtCpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return LayoutUtil::GetWithDefaultLayout(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers,
                         /*layout_canonicalization_callback=*/nullptr, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtCpuClient::CompileAheadOfTimeAndLoad(
    const XlaComputation& computation, CompileOptions options,
    const AotCompilationOptions& aot_options) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return LayoutUtil::GetWithDefaultLayout(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers,
                         /*layout_canonicalization_callback=*/nullptr, options,
                         &aot_options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtCpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options,
    const AotCompilationOptions* absl_nullable aot_options) {
  tsl::profiler::TraceMe traceme("TfrtCpuClient::Compile");
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

  if (layout_canonicalization_callback) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

  int num_replicas;
  int num_partitions;
  std::shared_ptr<DeviceAssignment> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options.compile_portable_executable, &options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  if (collectives_ == nullptr && device_assignment) {
    for (int replica = 0; replica < device_assignment->replica_count();
         ++replica) {
      for (int computation = 0;
           computation < device_assignment->computation_count();
           ++computation) {
        PjRtGlobalDeviceId id((*device_assignment)(replica, computation));
        if (UnpackCpuProcessIndex(id) != process_index()) {
          // TODO(phawkins): improve this error message when we're ready to
          // publicize that multiprocess collectives exist.
          return InvalidArgument(
              "Multiprocess computations aren't implemented on the CPU "
              "backend.");
        }
      }
    }
  }

  ExecutableBuildOptions& build_options = options.executable_build_options;
  std::vector<PjRtLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        PjRtGlobalDeviceId device_id((*device_assignment)(replica, partition));
        if (UnpackCpuProcessIndex(device_id) != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (!addressable_devices.empty() && build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id().value());
    }
  }

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());

  std::unique_ptr<Executable> cpu_executable;
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  if (aot_options) {
    TF_ASSIGN_OR_RETURN(
        cpu_executable,
        CompileAheadOfTime(computation, argument_layout_pointers, build_options,
                           execution_options, *aot_options,
                           eigen_intraop_device()->getPool()->NumThreads(),
                           customize_hlo_module_config_));
  } else {
    xla::Compiler::CompileOptions compile_options{
        build_options.device_allocator(), build_options.compile_thread_pool(),
        build_options.layout_canonicalization_callback()};
    if (!compile_options.thread_pool) {
      compile_options.thread_pool = pjrt_client_thread_pool();
    }
    TF_ASSIGN_OR_RETURN(
        cpu_executable,
        JitCompile(computation, argument_layout_pointers, build_options,
                   execution_options, compile_options,
                   eigen_intraop_device()->getPool()->NumThreads(),
                   customize_hlo_module_config_));
  }

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
      options.parameter_is_tupled_arguments, std::move(input_options),
      std::move(cpu_executable), result_slice.index(),
      std::move(result_buffer_indices),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);
  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));

  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

static bool IsAlignedData(void* ptr) {
  return (absl::bit_cast<std::uintptr_t>(ptr) &
          (cpu_function_runtime::MinAlign() - 1)) == 0;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtCpuClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  if (stream) {
    return Unimplemented(
        "TfrtCpuClient::CreateViewOfDeviceBuffer does not support `stream` "
        "argument.");
  }
  if (!IsAlignedData(device_ptr)) {
    return InvalidArgument(
        "Can't create a view of buffer with unaligned data, ptr: %#x is not "
        "aligned to %d bytes. ",
        reinterpret_cast<std::uintptr_t>(device_ptr),
        cpu_function_runtime::MinAlign());
  }
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  auto non_owning_buffer = CpuDeviceMemory::CreateForeignMemory(
      device_ptr, byte_size, std::move(on_delete_callback));
  auto tracked_device_buffer = std::make_unique<TrackedCpuDeviceBuffer>(
      /*owns_buffers=*/false, std::move(non_owning_buffer),
      /*definition_event=*/tsl::MakeAvailableAsyncValueRef<CpuEvent>());
  CHECK_EQ(memory_space->devices().size(), 1);
  auto* device = memory_space->devices().front();
  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tensorflow::down_cast<TfrtCpuDevice*>(device), memory_space));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices().front();
  if (device->client() != this) {
    return absl::InvalidArgumentError("Device is not attached to this client");
  }
  // Create a dummy buffer because the rest of the code expects a buffer
  // regardless of whether the definition event is an error.
  TF_ASSIGN_OR_RETURN(auto buffer,
                      CpuDeviceMemory::Allocate(ShapeUtil::ByteSizeOf(shape)));
  return std::make_unique<TfrtCpuBuffer>(
      shape,
      std::make_unique<TrackedCpuDeviceBuffer>(
          /*owns_buffers=*/true, std::move(buffer),
          absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>{
              tsl::AsyncValueRef<CpuEvent>(
                  tsl::MakeErrorAsyncValueRef(std::move(error)))}),
      this, tensorflow::down_cast<TfrtCpuDevice*>(device),
      *device->default_memory_space());
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtCpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  return xla::CreateAsyncHostToDeviceTransferManager(
      shape_specs, device_layouts, memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices().front();
  tsl::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostBuffer");
  Shape shape = ShapeUtil::MakeShape(type, dims);
  VLOG(2) << "TfrtCpuClient::BufferFromHostBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString();

  if (!device->IsAddressable()) {
    return InvalidArgument("Cannot copy array to non-addressable device %s",
                           device->DebugString());
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer,
      AbstractCpuBuffer::BufferFromHostBufferHelper(
          data, type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), shape, async_work_runner(),
          &transpose_mu_, &transpose_cache_));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tensorflow::down_cast<TfrtCpuDevice*>(device), memory_space));
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> TfrtCpuClient::LinearizeInto(
    const LiteralSlice& literal, const xla::Layout& layout,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
  return tensorflow::down_cast<CpuRawBuffer*>(raw_buffer.get())
      ->CopyFromLiteral(literal, layout, async_work_runner());
}

absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                         tsl::RCReference<PjRtDeviceEvent>>>
TfrtCpuClient::CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                                        absl::string_view debug_info) {
  auto definition_event_promise = tsl::MakeIndirectAsyncValue();
  auto definition_event = tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::AsyncValueRef<CpuEvent>(definition_event_promise));
  return std::make_pair(tsl::MakeRef<CpuTrackedDeviceEventPromise>(
                            std::move(definition_event_promise)),
                        std::move(definition_event));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::DefineBuffer(
    const Shape& on_device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
        definition_device_events) {
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events;
  for (auto& ev : definition_device_events) {
    definition_events.push_back(
        tensorflow::down_cast<CpuTrackedDeviceEvent*>(ev.get())->event());
  }
  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      on_device_shape,
      std::make_unique<TrackedCpuDeviceBuffer>(
          /*owns_buffers=*/true,
          tensorflow::down_cast<CpuRawBuffer*>(raw_buffer.get())->buffer(),
          std::move(definition_events)),
      this,
      tensorflow::down_cast<TfrtCpuDevice*>(
          raw_buffer->memory_space()->devices()[0]),
      raw_buffer->memory_space()));
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
TfrtCpuClient::AllocateRawBuffer(PjRtMemorySpace* memory_space,
                                 size_t on_device_bytes_count,
                                 tsl::AsyncValueRef<bool> allocate_after) {
  CHECK(allocate_after == nullptr) << "allocate_after is not supported for "
                                      "TfrtCpuClient.";
  return xla::CpuRawBuffer::Allocate(memory_space, on_device_bytes_count);
}

absl::StatusOr<int64_t> TfrtCpuClient::GetOnDeviceBytesCount(
    PjRtMemorySpace* memory_space, const xla::Shape& shape) const {
  return xla::ShapeUtil::ByteSizeOf(shape);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtCpuClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                     PjRtMemorySpace* memory_space,
                                     const Layout* device_layout) {
  if (device_layout) {
    return absl::UnimplementedError(absl::StrCat(
        "BufferFromHostLiteral with device_layout is not implemented on "
        "platform: ",
        platform_name()));
  }
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices().front();

  tsl::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostLiteral");
  VLOG(1) << "TfrtCpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().DebugString()
          << " device: " << device->DebugString();
  const Shape& shape = literal.shape();

  absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4> avs;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TfrtCpuBuffer> output_buffer,
                      AllocateDestinationBufferAndAvs(
                          HostShapeToOnDeviceShape(shape), &avs,
                          tensorflow::down_cast<TfrtCpuDevice*>(device), this));

  output_buffer->CopyFromLiteral(literal, shape, &avs, async_work_runner());

  return std::unique_ptr<PjRtBuffer>(std::move(output_buffer));
}

TfrtCpuBuffer::TfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer,
    TfrtCpuClient* client, TfrtCpuDevice* device, PjRtMemorySpace* memory_space)
    : AbstractCpuBuffer(std::move(on_device_shape),
                        std::move(tracked_device_buffer)),
      client_(client),
      device_(device),
      memory_space_(memory_space) {}

static std::vector<tsl::RCReference<tsl::AsyncValue>> CopyAsyncValues(
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events) {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev.CopyRef());
  }
  return avs;
}

PjRtFuture<> TfrtCpuBuffer::CopyRawToHost(void* dst, int64_t offset,
                                          int64_t transfer_size) {
  return CopyRawToHostHelper(dst, offset, transfer_size,
                             client()->async_work_runner());
}

PjRtFuture<> TfrtCpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  return ToLiteralHelper(literal, client()->async_work_runner());
}

PjRtFuture<> TfrtCpuBuffer::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  auto buffer = std::move(generator)();
  if (!buffer.ok()) {
    return PjRtFuture<>(buffer.status());
  }
  return ToLiteralHelper(buffer.value(), client()->async_work_runner());
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  CHECK_EQ(dst_memory_space->devices().size(), 1);
  PjRtDevice* dst_device = dst_memory_space->devices().front();
  tsl::profiler::TraceMe traceme("TfrtCpuBuffer::CopyToDevice");

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    return CopyToDeviceAcrossClients(dst_device);
  }

  if (!dst_device->IsAddressable()) {
    return InvalidArgument("Cannot copy array to non-addressable device %s",
                           dst_device->DebugString());
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer,
      CopyToDeviceHelper(client()->async_work_runner()));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      on_device_shape_, std::move(tracked_device_buffer), client(),
      tensorflow::down_cast<TfrtCpuDevice*>(dst_device),
      *dst_device->default_memory_space()));
}

TfrtCpuExecutable::TfrtCpuExecutable(
    int num_replicas, int num_partitions,
    std::shared_ptr<DeviceAssignment> device_assignment,
    bool parameter_is_tupled_arguments, CompileOptions compile_options,
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
      compile_options_(std::move(compile_options)),
      cpu_executable_(std::move(cpu_executable)),
      result_buffer_index_(result_buffer_index),
      result_buffer_indices_(std::move(result_buffer_indices)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  auto hlo_cost_analysis =
      std::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
  CHECK_OK(cpu_executable_->module().entry_computation()->Accept(
      hlo_cost_analysis.get()));
  // Cache to avoid std::map lookup in flop_count() on critical path. The magic
  // constant 1000 is determined by correlating computation with flop estimate.
  // It is a crude heuristic to find computation less than the thread context
  // switch time (~5us).
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
        computation_layout.parameter_shape(0).tuple_shapes().size());
    for (int i = 0;
         i < computation_layout.parameter_shape(0).tuple_shapes().size(); ++i) {
      input_buffer_sizes_in_bytes_.push_back(ShapeUtil::ByteSizeOf(
          computation_layout.parameter_shape(0).tuple_shapes(i)));
    }
  }

  // Compute fingerprint of the executable from the HloModule.
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
  fingerprint = tsl::FingerprintCat128(
      tsl::Fingerprint128(fingerprint_),
      tsl::Fingerprint128(cpu_executable_->module().ToString()));
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);
}

void TfrtCpuExecutable::Delete() {}

bool TfrtCpuExecutable::IsDeleted() { return false; }

absl::Status TfrtCpuExecutable::SetUpDonation(bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(parameters_that_must_be_donated_,
                      ComputeParametersThatMustBeDonated(
                          *cpu_executable_->shared_module(), tuple_inputs));
  return absl::OkStatus();
}

namespace {

// Some helper structs to support delayed memory allocation.

struct BufferInfo {
  tsl::AsyncValueRef<CpuDeviceMemory> buffer;
  bool owns_buffer;
  size_t buffer_size;
};

struct BufferAlloc {
  // All data members should have the same size.
  absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> buffers;
  absl::InlinedVector<size_t, 4> allocation_sizes;

  void Allocate() {
    for (int i = 0; i < buffers.size(); ++i) {
      auto status = CpuDeviceMemory::AllocateInto(allocation_sizes[i],
                                                  buffers[i].AsPtr());
      if (!status.ok()) {
        buffers[i].SetError(status);
        return;
      }
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(buffers[i]->untyped_data(),
                                          allocation_sizes[i]);
    }
  }
};

struct BufferAllocAndCopy {
  // All data members should have the same size.
  absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> src_buffers;
  absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> dst_buffers;
  absl::InlinedVector<size_t, 4> allocation_sizes;

  void AllocateAndCopy() {
    for (int i = 0; i < src_buffers.size(); ++i) {
      auto status = CpuDeviceMemory::AllocateInto(allocation_sizes[i],
                                                  dst_buffers[i].AsPtr());
      if (!status.ok()) {
        dst_buffers[i].SetError(status);
        return;
      }
      CHECK(src_buffers[i].IsConcrete());
      std::memcpy(dst_buffers[i]->untyped_data(),
                  src_buffers[i]->untyped_data(), allocation_sizes[i]);
    }
  }
};

}  // namespace

// The following few helpers are adapted from XLA:CPU to create a buffer table
// and assemble the buffer pointers in order to call into CpuExecutable.
static absl::StatusOr<BufferInfo> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<const cpu::ConstantAllocation> constants,
    absl::Span<std::pair<bool, TrackedCpuDeviceBuffer*> const> arguments,
    BufferAlloc& buffer_alloc, BufferAllocAndCopy& buffer_alloc_and_copy,
    const tsl::AsyncValueRef<CpuDeviceMemory>& tuple_index_table) {
  BufferInfo buffer_info;
  if (allocation.is_entry_computation_parameter()) {
    bool can_donate = false;
    TrackedCpuDeviceBuffer* arg = nullptr;
    size_t buffer_size;
    tsl::AsyncValuePtr<CpuDeviceMemory> out;
    if (tuple_index_table) {
      if (allocation.param_shape_index().empty()) {
        out = tuple_index_table.AsPtr();
        buffer_size = arguments.size() * sizeof(void*);
      } else if (allocation.param_shape_index().size() == 1) {
        std::tie(can_donate, arg) =
            arguments[allocation.param_shape_index()[0]];
        out = arg->buffer().AsPtr();
        buffer_size = arg->BufferSize();
      } else {
        return absl::InvalidArgumentError(absl::StrCat(
            "Nested tuples are not supported for argument: ",
            allocation.parameter_number(),
            " at shape index:", allocation.param_shape_index().ToString()));
      }
    } else if (!allocation.param_shape_index().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Nested tuples are not supported for argument: ",
          allocation.parameter_number(),
          " at shape index:", allocation.param_shape_index().ToString()));
    } else {
      std::tie(can_donate, arg) = arguments[allocation.parameter_number()];
      out = arg->buffer().AsPtr();
      buffer_size = arg->BufferSize();
    }
    CHECK_EQ(allocation.size(), buffer_size)
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();

    // If we don't own the buffer, we can't overwrite it or donate it. For
    // example we might be pointing to a buffer owned by the client whose
    // lifetime will not extend past the lifetime of the donated input buffer.
    if ((!can_donate || (arg && !arg->owns_buffers())) &&
        !allocation.is_readonly()) {
      auto copy = CpuDeviceMemory::CreateDelayedMemory();

      buffer_alloc_and_copy.src_buffers.push_back(out.CopyRef());
      buffer_alloc_and_copy.dst_buffers.push_back(copy);
      buffer_alloc_and_copy.allocation_sizes.push_back(allocation.size());

      buffer_info.buffer = std::move(copy);
      buffer_info.owns_buffer = true;
      buffer_info.buffer_size = allocation.size();
      return buffer_info;
    }

    buffer_info.buffer = out.CopyRef();
    buffer_info.owns_buffer = !arg || arg->owns_buffers();
    buffer_info.buffer_size = buffer_size;
    return buffer_info;

  } else if (allocation.is_constant() &&
             allocation.index() < constants.size()) {
    se::DeviceMemoryBase constant =
        constants[allocation.index()].AsDeviceMemoryBase();
    buffer_info.buffer = CpuDeviceMemory::CreateConstantMemory(
        constant.opaque(), constant.size());
    buffer_info.owns_buffer = false;
    buffer_info.buffer_size = constant.size();
    return buffer_info;

  } else if (allocation.is_constant() || allocation.is_thread_local()) {
    buffer_info.buffer = CpuDeviceMemory::CreateConstantMemory(nullptr, 0);
    buffer_info.owns_buffer = true;
    buffer_info.buffer_size = 0;
    return buffer_info;
  }

  // Output and temporary buffer.
  auto out = CpuDeviceMemory::CreateDelayedMemory();

  buffer_alloc.buffers.push_back(out);
  buffer_alloc.allocation_sizes.push_back(allocation.size());

  buffer_info.buffer = std::move(out);
  buffer_info.owns_buffer = true;
  buffer_info.buffer_size = allocation.size();
  return buffer_info;
}

static absl::StatusOr<std::vector<BufferInfo>> CreateBufferTable(
    const BufferAssignment& assignment,
    absl::Span<const cpu::ConstantAllocation> constants,
    absl::Span<std::pair<bool, TrackedCpuDeviceBuffer*> const> arguments,
    BufferAlloc& buffer_alloc, BufferAllocAndCopy& buffer_alloc_and_copy,
    const tsl::AsyncValueRef<CpuDeviceMemory>& tuple_index_table) {
  std::vector<BufferInfo> buffer_table(assignment.Allocations().size());
  for (BufferAllocation::Index i = 0; i < buffer_table.size(); ++i) {
    const BufferAllocation& allocation = assignment.GetAllocation(i);
    TF_ASSIGN_OR_RETURN(
        buffer_table[i],
        MemoryForAllocation(allocation, constants, arguments, buffer_alloc,
                            buffer_alloc_and_copy, tuple_index_table));
  }
  return std::move(buffer_table);
}

static absl::InlinedVector<BufferInfo, 4> CreateResultBufferInfo(
    absl::Span<const BufferAllocation::Index> buffer_indices,
    absl::Span<const BufferInfo> buffer_table) {
  absl::InlinedVector<BufferInfo, 4> output_buffer_info;
  output_buffer_info.reserve(buffer_indices.size());
  for (int i = 0; i < buffer_indices.size(); ++i) {
    output_buffer_info.push_back(buffer_table[buffer_indices[i]]);
  }
  return output_buffer_info;
}

absl::Status TfrtCpuExecutable::CheckBufferCompatibilities(
    absl::Span<std::pair<bool, TrackedCpuDeviceBuffer*> const> input_buffers)
    const {
  if (input_buffers.size() != input_buffer_sizes_in_bytes_.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes_.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i].second;
    if (input_buffer_sizes_in_bytes_[i] != buffer->BufferSize()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with "
          "incompatible size %lld",
          i, input_buffer_sizes_in_bytes_[i], buffer->BufferSize());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutable::Result> TfrtCpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    tsl::AsyncValueRef<CpuEvent> last_collective_launch_event, bool fill_future,
    TfrtCpuDevice* device) {
  tsl::profiler::TraceMe traceme("TfrtCpuExecutable::ExecuteHelper");

  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int64_t device_id = (*device_assignment_)(replica, partition);
    PjRtGlobalDeviceId global_device_id(device_id);
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(global_device_id));
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
  auto execute_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  MarkEventReadyOnExit ready_on_exit(execute_event);

  absl::InlinedVector<TfrtCpuBuffer::ScopedHold, 4> donation_transactions;

  absl::InlinedVector<std::pair<bool, TrackedCpuDeviceBuffer*>, 4>
      tracked_buffers;
  tracked_buffers.reserve(argument_handles.size());
  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as the
  // single definition event.
  std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
  input_deps.reserve(argument_handles.size());

  auto donate_it = parameters_that_must_be_donated_.begin();

  // State for `TestBufferDonationClashes`.
  absl::flat_hash_map<const void*, std::pair<bool, int>> donation_clashes;
  donation_clashes.reserve(argument_handles.size());
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

    TrackedCpuDeviceBuffer* tracked_buffer;
    auto get_buffer = [&](int i) -> absl::Status {
      bool must_donate = donate_it != parameters_that_must_be_donated_.end() &&
                         *donate_it == i;
      TF_RETURN_IF_ERROR(TestBufferDonationClashes(
          tfrt_buffer, donation_clashes, must_donate, i, replica, partition));
      if (must_donate) {
        ++donate_it;
        TfrtCpuBuffer::ScopedHold donation_transaction =
            tfrt_buffer->AcquireDonation();
        // On CPU, we allow donation to succeed by introducing a copy. This was
        // added when enabling buffer donation on CPU since it turned out that a
        // number of users were holding external references to buffers that were
        // supposed to be donated. We may wish to tighten those semantics in the
        // future.
        if (donation_transaction.ok()) {
          // After acquiring the buffer for donation, we retrieve the dependent
          // usage events. Note that we don't need any locking here as
          // AcquireDonation() is supposed to synchronize with other usages.
          for (const auto& ev : donation_transaction->UsageEvents()) {
            if (!ev.IsAvailable()) {
              input_deps.push_back(ev.CopyRCRef());
            }
          }
          tracked_buffer = donation_transaction.buffer();
          tracked_buffers.emplace_back(/*can_donate=*/true, tracked_buffer);
          donation_transactions.push_back(std::move(donation_transaction));
          return absl::OkStatus();
        }
      }
      tracked_buffer = tfrt_buffer->AcquireUsage(execute_event);
      if (!tracked_buffer)
        return InvalidArgument(
            "Invalid buffer passed: buffer has been deleted or donated.");
      tracked_buffers.emplace_back(/*can_donate=*/false, tracked_buffer);
      return absl::OkStatus();
    };
    TF_RETURN_IF_ERROR(get_buffer(i));

    // Definition events are never modified after buffer construction. If they
    // are available and have no error, they can be skipped in input deps.
    // In contrast, already known errors in the input are taken as deps so that
    // they can poison output buffers.
    const auto& definition_event = tracked_buffer->definition_event();
    if (!definition_event.IsAvailable() || definition_event.IsError()) {
      input_deps.push_back(definition_event.CopyRCRef());
    }
  }

  TF_RETURN_IF_ERROR(CheckBufferCompatibilities(tracked_buffers));

  // Tuplize the inputs if compiler expects a single tuple argument but runtime
  // gets many inputs that are not yet tupled.
  tsl::AsyncValueRef<CpuDeviceMemory> tuple_index_table;
  if (parameter_is_tupled_arguments_ && !options.arguments_are_tupled) {
    absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> leaf_buffers;
    leaf_buffers.reserve(tracked_buffers.size());
    for (const auto& tracked_buffer : tracked_buffers) {
      leaf_buffers.push_back(tracked_buffer.second->buffer());
    }
    tuple_index_table = CpuDeviceMemory::CreateDelayedMemory();
    tsl::RunWhenReady(
        absl::MakeConstSpan(leaf_buffers),
        [buffers = leaf_buffers,
         tuple_index_table = tuple_index_table]() mutable {
          size_t index_table_byte_size = buffers.size() * sizeof(void*);
          // We assume tuple table allocations will not fail.
          CHECK_OK(CpuDeviceMemory::AllocateInto(index_table_byte_size,
                                                 tuple_index_table.AsPtr()));
          uintptr_t* index_table =
              reinterpret_cast<uintptr_t*>(tuple_index_table->untyped_data());
          for (int i = 0; i < buffers.size(); ++i) {
            index_table[i] =
                absl::bit_cast<uintptr_t>(buffers[i]->untyped_data());
          }
        });
  }

  auto* cpu_executable =
      tensorflow::down_cast<cpu::CpuExecutable*>(cpu_executable_.get());
  // `buffer_alloc` and `buffer_alloc_and_copy` are used to do real memory
  // allocation and copy work.
  BufferAlloc buffer_alloc;
  BufferAllocAndCopy buffer_alloc_and_copy;
  TF_ASSIGN_OR_RETURN(
      std::vector<BufferInfo> buffer_table,
      CreateBufferTable(cpu_executable->buffer_assignment(),
                        cpu_executable->constants(), tracked_buffers,
                        buffer_alloc, buffer_alloc_and_copy,
                        tuple_index_table));
  auto result_buffers_info =
      CreateResultBufferInfo(result_buffer_indices_, buffer_table);

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  auto compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
      device->max_inflight_computations_semaphore().ScopedAcquire(1));

  ExecutableRunOptions run_options;
  run_options.set_run_id(run_id);
  // Need to keep device_assignment alive until execution completes.
  run_options.set_device_assignment(device_assignment.get());
  run_options.set_intra_op_thread_pool(client_->eigen_intraop_device());

  auto cpu_run_options = std::make_shared<cpu::CpuExecutableRunOptions>();
  run_options.set_cpu_executable_run_options(cpu_run_options.get());

  const CpuExecuteContext* cpu_execute_context =
      options.context == nullptr
          ? nullptr
          : dynamic_cast<const CpuExecuteContext*>(options.context);
  if (cpu_execute_context != nullptr &&
      cpu_execute_context->process_index().has_value()) {
    run_options.set_device_ordinal(
        PackCpuDeviceId(*cpu_execute_context->process_index(),
                        UnpackCpuLocalDeviceId(device->global_device_id()))
            .value());
  } else {
    run_options.set_device_ordinal(device->global_device_id().value());
  }
  if (cpu_execute_context != nullptr &&
      cpu_execute_context->collectives() != nullptr) {
    cpu_run_options->set_collectives(cpu_execute_context->collectives());
  } else {
    cpu_run_options->set_collectives(client_->collectives_.get());
  }

  // Schedule only one collective at a time.
  bool is_a_collective_launch = !!last_collective_launch_event;
  // Add additional dependency conditioned on whether this is a collective
  // launch or not.
  if (is_a_collective_launch) {
    input_deps.push_back(std::move(last_collective_launch_event));
  } else {
    // This is a non-parallel computation. Add the last enqueue event as a
    // dependency with any error cleared.
    auto last_enqueue_event = client_->GetLastEnqueueEvent();
    if (!last_enqueue_event.IsAvailable()) {
      auto last_enqueue_done_event =
          tsl::MakeUnconstructedAsyncValueRef<CpuEvent>();
      last_enqueue_event.AndThen(
          [last_enqueue_done_event = last_enqueue_done_event.CopyRef()]() {
            last_enqueue_done_event.emplace();
          });
      input_deps.push_back(std::move(last_enqueue_done_event));
    }
  }
  if (options.context != nullptr) {
    run_options.set_ffi_execution_context(&options.context->ffi_context());
  }

  bool execute_inline = cheap_computation_ || !client_->asynchronous_ ||
                        ThisThreadIsInsideHostCallback();

  // Overwrite `execute_inline` if it is specified in the ExecuteOptions.
  if (options.execution_mode == ExecuteOptions::ExecutionMode::kAsynchronous) {
    execute_inline = false;
  } else if (options.execution_mode ==
             ExecuteOptions::ExecutionMode::kSynchronous) {
    execute_inline = true;
  }

  if (input_deps.empty() && execute_inline) {
    // Synchronously call generated function or thunk sequence.

    // Set denormal and rounding behavior to match the default TF
    // ThreadPool behavior.
    tsl::port::ScopedFlushDenormal flush;
    tsl::port::ScopedSetRound round(FE_TONEAREST);

    // Execution status for XLA:CPU "classic" runtime or thunks.
    XlaCustomCallStatus compute_function_status;
    tsl::AsyncValueRef<cpu::Thunk::ExecuteEvent> thunks_execute_event;

    // Immediately allocate memory and prepare for computation.
    buffer_alloc.Allocate();
    buffer_alloc_and_copy.AllocateAndCopy();
    std::vector<void*> buffer_pointers;
    buffer_pointers.reserve(buffer_table.size());
    for (const auto& buffer_info : buffer_table) {
      CHECK(buffer_info.buffer.IsAvailable());
      if (buffer_info.buffer.IsError()) {
        return buffer_info.buffer.GetError();
      }
      buffer_pointers.push_back(buffer_info.buffer->untyped_data());
    }
    void* result_buffer = buffer_pointers[result_buffer_index_];

    if (cpu_executable->has_compute_function()) {
      // Call jit-compiled function implementing XLA executable.
      cpu_executable->compute_function()(result_buffer, &run_options, nullptr,
                                         buffer_pointers.data(),
                                         &compute_function_status, nullptr);

    } else if (cpu_executable->has_thunks()) {
      // Call interpreted thunk sequence implementing XLA executable.
      absl::InlinedVector<MaybeOwningDeviceMemory, 8> buffer_device_mem;
      buffer_device_mem.reserve(buffer_table.size());
      for (const auto& buffer_info : buffer_table) {
        buffer_device_mem.emplace_back(
            se::DeviceMemoryBase(buffer_info.buffer->untyped_data(),
                                 buffer_info.buffer->size_bytes()));
      }

      cpu::BufferAllocations allocations(buffer_device_mem);

      TF_ASSIGN_OR_RETURN(
          cpu::Thunk::CollectiveExecuteParams collective_params,
          cpu::Thunk::CollectiveExecuteParams::Create(&run_options));

      TF_ASSIGN_OR_RETURN(
          cpu::Thunk::CustomCallExecuteParams custom_call_execute_params,
          cpu::Thunk::CustomCallExecuteParams::Create(&run_options));

      cpu::ThreadPoolTaskRunner task_runner(
          run_options.intra_op_thread_pool()->getPool());

      cpu::Thunk::ExecuteParams execute_params = {
          cpu_executable->function_library(),
          &allocations,
          cpu::runtime::GetXfeedManager(run_options.device_ordinal()),
          run_options.intra_op_thread_pool(),
          &task_runner,
          &collective_params,
          &custom_call_execute_params};

      thunks_execute_event = cpu_executable->thunks().Execute(execute_params);

      tsl::profiler::TraceMe trace(
          "ThunkExecutor::Execute (wait for completion)");
      tsl::BlockUntilReady(thunks_execute_event);

    } else {
      return Internal("CpuExecutable has no compute function or thunks.");
    }

    for (auto& donation_transaction : donation_transactions) {
      std::move(donation_transaction).ConfirmDonation();
    }

    // Forward errors (if any) after executing compute function or thunks.
    if (cpu_executable->has_compute_function()) {
      if (auto error_message =
              xla::CustomCallStatusGetMessage(&compute_function_status)) {
        return Internal("Generated function failed: %s", *error_message);
      }
    } else if (thunks_execute_event.IsError()) {
      return thunks_execute_event.GetError();
    }

  } else {
    // Asynchronously call generated function.

    // We only created enough threads for one collective to complete.
    // The next collective launch will not be scheduled onto threadpool until
    // this one completes.
    if (is_a_collective_launch) {
      client_->SetLastCollectiveLaunchEvent(execute_event.CopyRef());
    } else {
      // This is a non-parallel computation. Set the execute event as the new
      // last enqueue event.
      client_->SetLastEnqueueEvent(execute_event.CopyRef());
    }
    std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps_avs_copy =
        CopyAsyncValues(input_deps);
    CpuScopedAsyncExecution scoped_async_execution =
        device->async_execution_tracker()->NewAsyncExecution(
            run_id.ToInt(), std::move(ready_on_exit).Release());
    EnqueueWorkWhenReady(
        client()->pjrt_client_thread_pool(), input_deps,
        [cpu_executable, buffer_alloc = std::move(buffer_alloc),
         buffer_alloc_and_copy = std::move(buffer_alloc_and_copy),
         result_buffer_index = result_buffer_index_,
         buffer_table = std::move(buffer_table),
         run_options = std::move(run_options),
         cpu_executable_copy = cpu_executable_,
         device_assignment = std::move(device_assignment),
         cpu_run_options = std::move(cpu_run_options),
         compute_reservation = std::move(compute_reservation),
         tuple_index_table = std::move(tuple_index_table),
         donation_transactions = std::move(donation_transactions),
         scoped_async_execution = std::move(scoped_async_execution),
         input_deps_avs = std::move(input_deps_avs_copy),
         eigen_device = client()->eigen_intraop_device()]() mutable {
          // Because `input_deps` contains the definition events of all inputs,
          // when it is ready, all input buffers must have been allocated. So,
          // we are safe to allocate and copy memory here. Since `execute_event`
          // may error out, we need to do it early.
          buffer_alloc.Allocate();
          buffer_alloc_and_copy.AllocateAndCopy();

          for (const auto& av : input_deps_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              scoped_async_execution.SetError(Internal(
                  "Error dispatching computation: %s", error->message()));
              return;
            }
          }

          // Set denormal and rounding behavior to match the default TF
          // ThreadPool behavior.
          tsl::port::ScopedFlushDenormal flush;
          tsl::port::ScopedSetRound round(FE_TONEAREST);

          // Prepare for computation.
          std::vector<void*> buffer_pointers;
          buffer_pointers.reserve(buffer_table.size());
          for (const auto& buffer_info : buffer_table) {
            CHECK(buffer_info.buffer.IsAvailable());
            if (buffer_info.buffer.IsError()) {
              scoped_async_execution.SetError(
                  Internal("Error preparing computation: %s",
                           buffer_info.buffer.GetError().message()));
              return;
            }
            buffer_pointers.push_back(buffer_info.buffer->untyped_data());
          }
          void* result_buffer = buffer_pointers[result_buffer_index];

          absl::Status status;

          if (cpu_executable->has_compute_function()) {
            // Call jit-compiled function implementing XLA executable.
            XlaCustomCallStatus compute_function_status;

            cpu_executable->compute_function()(
                result_buffer, &run_options, nullptr, buffer_pointers.data(),
                &compute_function_status, nullptr);
            if (auto error_message =
                    xla::CustomCallStatusGetMessage(&compute_function_status)) {
              status =
                  Internal("Generated function failed: %s", *error_message);
            }

          } else if (cpu_executable->has_thunks()) {
            // Call interpreted thunk sequence implementing XLA executable.
            absl::InlinedVector<MaybeOwningDeviceMemory, 8> buffer_device_mem;
            buffer_device_mem.reserve(buffer_table.size());
            for (const auto& buffer_info : buffer_table) {
              buffer_device_mem.emplace_back(
                  se::DeviceMemoryBase(buffer_info.buffer->untyped_data(),
                                       buffer_info.buffer->size_bytes()));
            }

            cpu::BufferAllocations allocations(buffer_device_mem);

            absl::StatusOr<cpu::Thunk::CollectiveExecuteParams>
                collective_params =
                    cpu::Thunk::CollectiveExecuteParams::Create(&run_options);

            absl::StatusOr<cpu::Thunk::CustomCallExecuteParams>
                custom_call_params =
                    cpu::Thunk::CustomCallExecuteParams::Create(&run_options);

            cpu::ThreadPoolTaskRunner task_runner(
                run_options.intra_op_thread_pool()->getPool());

            if (collective_params.ok()) {
              cpu::Thunk::ExecuteParams execute_params = {
                  cpu_executable->function_library(),
                  &allocations,
                  cpu::runtime::GetXfeedManager(run_options.device_ordinal()),
                  run_options.intra_op_thread_pool(),
                  &task_runner,
                  &*collective_params,
                  &*custom_call_params};

              auto thunks_execute_event =
                  cpu_executable->thunks().Execute(execute_params);

              tsl::profiler::TraceMe trace(
                  "ThunkExecutor::Execute (wait for completion)");
              tsl::BlockUntilReady(thunks_execute_event);
              status = thunks_execute_event.IsError()
                           ? thunks_execute_event.GetError()
                           : absl::OkStatus();
            } else {
              status = collective_params.status();
            }

          } else {
            status =
                Internal("CpuExecutable has no compute function or thunks.");
          }

          for (auto& donation_transaction : donation_transactions) {
            std::move(donation_transaction).ConfirmDonation();
          }

          if (!status.ok()) {
            // CPU computation fails with an error.
            scoped_async_execution.SetError(std::move(status));
          }

          // CPU computation completes.
          scoped_async_execution.SetStateConcrete();
        });
  }

  // Create output TFRT buffers.
  const Shape& result_shape = cpu_executable_->result_shape();
  std::vector<std::unique_ptr<PjRtBuffer>> res;
  if (result_shape.IsTuple()) {
    res.reserve(result_buffers_info.size());
    for (int i = 0; i < result_buffers_info.size(); ++i) {
      // Program execution writes to output buffers so it's a definition event.
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events;
      definition_events.push_back(execute_event.CopyRef());
      auto leaf_tracked_device_buffer =
          std::make_unique<TrackedCpuDeviceBuffer>(
              result_buffers_info[i].owns_buffer,
              std::move(result_buffers_info[i].buffer),
              result_buffers_info[i].buffer_size, std::move(definition_events));
      auto leaf_buffer = std::make_unique<TfrtCpuBuffer>(
          result_shape.tuple_shapes(i), std::move(leaf_tracked_device_buffer),
          client_, device, *device->default_memory_space());
      res.push_back(std::move(leaf_buffer));
    }
  } else {
    CHECK_EQ(result_buffers_info.size(), 1);
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedCpuDeviceBuffer>(
        result_buffers_info[0].owns_buffer,
        std::move(result_buffers_info[0].buffer),
        result_buffers_info[0].buffer_size,
        /*definition_event=*/execute_event);
    auto tfrt_output_buffer = std::make_unique<TfrtCpuBuffer>(
        result_shape, std::move(tracked_device_buffer), client_, device,
        *device->default_memory_space());
    res.push_back(std::move(tfrt_output_buffer));
  }

  std::optional<PjRtFuture<>> future;
  if (fill_future) {
    PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
    execute_event.AndThen([promise, event = execute_event.CopyRef()]() mutable {
      if (auto* error = event.GetErrorIfPresent()) {
        promise.Set(Internal("Compute error: %s", error->message()));
      } else {
        promise.Set();
      }
    });
    future = PjRtFuture<>(std::move(promise));
  }
  return Result({/*future=*/std::move(future), /*buffers=*/std::move(res)});
}

static void MaybeDumpHloSnapshot(
    const HloModule& module, RunId run_id,
    const std::vector<PjRtBuffer*>& arguments,
    const std::vector<std::unique_ptr<PjRtBuffer>>& results) {
  if (!DumpingEnabledForHloModule(module)) {
    return;
  }
  if (!module.config().debug_options().xla_dump_hlo_snapshots()) {
    return;
  }
  xla::HloSnapshot hlo_snapshot;
  *hlo_snapshot.mutable_hlo()->mutable_hlo_module() = module.ToProto();

  for (auto* argument : arguments) {
    *hlo_snapshot.add_arguments() = (*argument->ToLiteralSync())->ToProto();
  }

  // If there are multiple results, wrap them in a tuple.
  if (results.size() == 1) {
    *hlo_snapshot.mutable_result() = (*results[0]->ToLiteralSync())->ToProto();
  } else {
    std::vector<Literal> result_literals;
    result_literals.reserve(results.size());
    for (auto& result : results) {
      result_literals.push_back(std::move(**result->ToLiteralSync()));
    }
    *hlo_snapshot.mutable_result() =
        LiteralUtil::MakeTupleOwned(std::move(result_literals)).ToProto();
  }

  DumpToFileInDir(module, "", absl::StrCat("snapshot.", run_id.ToInt(), ".pb"),
                  hlo_snapshot.SerializeAsString());
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtCpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtCpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
  if (!options.untuple_result && cpu_executable_->module()
                                     .config()
                                     .entry_computation_layout()
                                     .result_shape()
                                     .IsTuple()) {
    return InvalidArgument(
        "Tuple results must be untupled using ExecuteOptions::untuple_result.");
  }
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }
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

    // Dump once before running, in case there's a crash.
    MaybeDumpHloSnapshot(cpu_executable_->module(), run_id, argument_handles[0],
                         {});
    auto statusor = ExecuteHelper(
        argument_handles[0], replica, partition, run_id, options,
        /*last_collective_launch_event=*/tsl::AsyncValueRef<CpuEvent>(),
        returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    MaybeDumpHloSnapshot(cpu_executable_->module(), run_id, argument_handles[0],
                         wrapped_results[0]);
  } else {
    // Gang schedule collectives to ensure that collectives with the same RunId
    // are run at the same time. We conservatively run only one collective at a
    // time, because we may not have enough threads to run arbitrary number of
    // collectives concurrently.
    tsl::AsyncValueRef<CpuEvent> last_collective_launch_event =
        client_->GetLastCollectiveLaunchEvent();

    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;

      auto* thread_pool = client()->pjrt_client_thread_pool();
      EnqueueWork(thread_pool, [&, replica, partition, i] {
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

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtCpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtCpuExecutable::ExecuteSharded",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
  if (device_assignment_ == nullptr) {
    return InvalidArgument("ExecuteShard expects a non-null device_assignment");
  }
  if (!options.untuple_result && cpu_executable_->module()
                                     .config()
                                     .entry_computation_layout()
                                     .result_shape()
                                     .IsTuple()) {
    return InvalidArgument(
        "Tuple results must be untupled using ExecuteOptions::untuple_result.");
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
              addressable_device_logical_ids_[i].partition, run_id, options,
              /*last_collective_launch_event=*/
              tsl::AsyncValueRef<CpuEvent>(), fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->id());
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtCpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtCpuExecutable::ExecutePortable",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
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
          /*partition=*/0, run_id, options,
          /*last_collective_launch_event=*/tsl::AsyncValueRef<CpuEvent>(),
          fill_future, tensorflow::down_cast<TfrtCpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}
}  // namespace xla
