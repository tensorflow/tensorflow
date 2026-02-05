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

#include <algorithm>
#include <cfenv>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/array.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/constant_allocation.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thread_pool_task_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/backends/cpu/runtime/xfeed_manager.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/client/executable_build_options.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/cpu/abstract_cpu_buffer.h"
#include "xla/pjrt/cpu/cpu_async_execution_tracker.h"
#include "xla/pjrt/cpu/cpu_device.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/dump/dump.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/host_to_device_transfer_manager.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_execute_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_memory.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/thread_pool_async_work_runner.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_value.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
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
#include "tsl/profiler/lib/traceme.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla {

static int CpuDeviceCount() {
  // By default we fix the number of devices to one.  However we do let the user
  // override this behavior to help run tests on the host that run models in
  // parallel across multiple devices, e.g. pmap.
  return GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
}

namespace {

// A custom memory allocator function passed via the CPU client options.
using CustomAllocatorFn =
    std::function<absl::StatusOr<std::unique_ptr<CpuMemory>>(size_t size_bytes,
                                                             size_t alignment)>;

// A custom raw memory that wraps a CpuMemory allocated by the user.
class CustomMemory final : public CpuDeviceMemory::RawMemory {
 public:
  explicit CustomMemory(std::unique_ptr<CpuMemory> mem)
      : mem_(std::move(mem)) {}

  void* base() const final { return mem_->base(); }
  size_t size_bytes() const final { return mem_->size_bytes(); }

 private:
  std::unique_ptr<CpuMemory> mem_;
};

// A custom raw memory allocator that wraps an allocation function passed via
// the client options.
class CustomAllocator final : public CpuDeviceMemory::Allocator {
 public:
  explicit CustomAllocator(CustomAllocatorFn allocator_fn)
      : allocator_fn_(std::move(allocator_fn)) {}

  absl::StatusOr<std::unique_ptr<CpuDeviceMemory::RawMemory>> Allocate(
      size_t size_bytes, size_t alignment) const final {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<CpuMemory> mem,
                        allocator_fn_(size_bytes, alignment));
    return std::make_unique<CustomMemory>(std::move(mem));
  }

 private:
  CustomAllocatorFn allocator_fn_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtClient>> GetPjRtCpuClient(
    CpuClientOptions options) {
  // Need at least CpuDeviceCount threads to launch one collective.
  int cpu_device_count = options.cpu_device_count.value_or(CpuDeviceCount());
  size_t num_threads = std::max(DefaultThreadPoolSize(), cpu_device_count);

  std::unique_ptr<CpuDeviceMemory::Allocator> allocator =
      options.allocator ? std::make_unique<CustomAllocator>(options.allocator)
                        : CpuDeviceMemory::MakeDefaultAllocator();

  std::unique_ptr<CpuTopologyDescription> topology = nullptr;
  if (!options.topology) {
    std::vector<CpuTopology::CpuDevice> cpu_topology_devices;
    cpu_topology_devices.reserve(cpu_device_count);
    for (int i = 0; i < cpu_device_count; ++i) {
      cpu_topology_devices.push_back(
          CpuTopology::CpuDevice{options.process_id, i});
    }

    topology = std::make_unique<CpuTopologyDescription>(
        xla::CpuPlatformId(), xla::CpuPlatformName(), xla::CpuPlatformVersion(),
        CpuTopology(cpu_topology_devices,
                    cpu::TargetMachineOptions(GetDebugOptionsFromFlags())));
  } else {
    topology = std::make_unique<CpuTopologyDescription>(*options.topology);
    if (topology->cpu_topology().number_of_devices() != cpu_device_count) {
      return InvalidArgument(
          "Number of devices in topology (%d) does not match "
          "cpu_device_count (%d)",
          topology->cpu_topology().number_of_devices(), cpu_device_count);
    }
  }

  std::vector<std::unique_ptr<PjRtCpuDevice>> devices;
  devices.reserve(topology->cpu_topology().number_of_devices());
  for (const auto& topology_device : topology->cpu_topology().devices()) {
    auto device = std::make_unique<PjRtCpuDevice>(
        topology_device.process_id, topology_device.local_device_id,
        options.max_inflight_computations_per_device);
    devices.push_back(std::move(device));
  }

  return std::unique_ptr<PjRtClient>(new PjRtCpuClient(
      options.process_id, std::move(devices), std::move(allocator),
      std::move(options.collectives), num_threads, options.asynchronous,
      std::move(options.customize_hlo_module_config),
      options.max_transpose_threads, std::move(topology)));
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

PjRtCpuClient::PjRtCpuClient(
    int process_index, std::vector<std::unique_ptr<PjRtCpuDevice>> devices,
    std::shared_ptr<CpuDeviceMemory::Allocator> allocator,
    std::shared_ptr<cpu::CpuCollectives> collectives, size_t num_threads,
    bool asynchronous,
    std::function<void(HloModuleConfig&)> customize_hlo_module_config,
    int max_transpose_threads, std::unique_ptr<CpuTopologyDescription> topology)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      allocator_(std::move(allocator)),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<CpuEvent>()),
      transpose_cache_(1024),
      collectives_(std::move(collectives)),
      topology_(std::move(topology)),
      asynchronous_(asynchronous),
      customize_hlo_module_config_(std::move(customize_hlo_module_config)),
      eigen_intraop_pool_(new tsl::thread::ThreadPool(
          tsl::Env::Default(), GetThreadOptions(), "XLAEigen",
          std::min(num_threads, kMaxIntraOpThreads))),
      eigen_intraop_device_(
          new Eigen::ThreadPoolDevice(eigen_intraop_pool_->AsEigenThreadPool(),
                                      eigen_intraop_pool_->NumThreads())),
      pjrt_client_thread_pool_(
          new tsl::thread::ThreadPool(tsl::Env::Default(), GetThreadOptions(),
                                      "XLAPjRtCpuClient", num_threads)),
      async_work_runner_(
          MakeThreadPoolAsyncWorkRunner(pjrt_client_thread_pool_.get())),
      max_transpose_threads_(max_transpose_threads) {
  for (const std::unique_ptr<PjRtCpuDevice>& device : owned_devices_) {
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
    auto* cpu_device = tsl::down_cast<PjRtCpuDevice*>(device);

    // Use the device id to construct a globally unique memory space id.
    const int id = device->id();

    // The first attached memory space is returned as the default by
    // PjRtCpuDevice, so attach the device memory space first.
    auto cpu_device_memory_space =
        std::make_unique<CpuDeviceMemorySpace>(id * 3 + 0, device);
    cpu_device->AttachMemorySpace(cpu_device_memory_space.get());
    memory_spaces_.push_back(cpu_device_memory_space.get());
    owned_memory_spaces_.push_back(std::move(cpu_device_memory_space));

    auto pinned_memory_space =
        std::make_unique<PinnedHostMemorySpace>(id * 3 + 1, device);
    cpu_device->AttachMemorySpace(pinned_memory_space.get());
    memory_spaces_.push_back(pinned_memory_space.get());
    owned_memory_spaces_.push_back(std::move(pinned_memory_space));

    auto unpinned_memory_space =
        std::make_unique<UnpinnedHostMemorySpace>(id * 3 + 2, device);
    cpu_device->AttachMemorySpace(unpinned_memory_space.get());
    memory_spaces_.push_back(unpinned_memory_space.get());
    owned_memory_spaces_.push_back(std::move(unpinned_memory_space));
  }
  VLOG(1) << "PjRtCpuClient created.";
}

PjRtCpuClient::~PjRtCpuClient() { VLOG(1) << "PjRtCpuClient destroyed."; }

absl::StatusOr<PjRtDevice*> PjRtCpuClient::LookupDevice(
    xla::PjRtGlobalDeviceId global_device_id) const {
  auto it = id_to_device_.find(global_device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         global_device_id.value());
}

absl::StatusOr<PjRtDevice*> PjRtCpuClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_device_id %d",
                         local_device_id.value());
}

absl::Span<PjRtMemorySpace* const> PjRtCpuClient::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<DeviceAssignment> PjRtCpuClient::GetDefaultDeviceAssignment(
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

absl::StatusOr<Layout> PjRtCpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  return LayoutUtil::GetWithDefaultLayout(shape).layout();
}

absl::StatusOr<std::unique_ptr<HloCostAnalysis>>
PjRtCpuClient::GetHloCostAnalysis() const {
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

absl::StatusOr<std::string> PjRtCpuExecutable::SerializeExecutable() const {
  cpu::CpuCompiler compiler;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler.Export(cpu_executable_.get()));

  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  if (serialized.empty()) {
    return Internal(
        "PjRtCpuClient::SerializeExecutable proto serialization failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::LoadSerializedExecutable(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  ExecutableAndOptionsProto proto;
  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal(
        "PjRtCpuClient::DeserializeExecutable proto too large (>2GB)");
  }
  if (!proto.ParseFromString(serialized)) {
    return Internal(
        "PjRtCpuClient::DeserializeExecutable proto deserialization failed");
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
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      std::move(*aot_result).LoadExecutable());

  // Set up other arguments for PjRtCpuLoadedExecutable
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
      tsl::down_cast<cpu::CpuExecutable*>(executable.get());

  // `result_buffer_indices` has the buffer allocation indices that make up the
  // output buffer (could be tuple).
  TF_ASSIGN_OR_RETURN(
      auto result_buffer_indices,
      FindResultBufferAllocationIndex(cpu_executable_ptr->buffer_assignment(),
                                      executable->module()));

  // Propagate env_option_overrides-> debug_options
  TF_RETURN_IF_ERROR(compile_options.ApplyAllOptionOverrides());
  // Override the debug_options() embedded in the module with those
  // explicitly passed in when deserializing. This allows options such as
  // --xla_dump_to to be changed.
  if (executable->has_module()) {
    ExecutableBuildOptions& build_options =
        compile_options.executable_build_options;
    DumpHloModuleIfEnabled(executable->module(), kAfterOptimizationsDumpName,
                           build_options.has_debug_options()
                               ? &build_options.debug_options()
                               : nullptr);
  }

  auto cpu_executable = std::make_shared<PjRtCpuExecutable>(
      num_replicas, num_partitions,
      compile_options.parameter_is_tupled_arguments, std::move(input_options),
      std::move(executable), std::move(result_buffer_indices), nullptr);
  TF_RETURN_IF_ERROR(cpu_executable->SetUpDonation(
      compile_options.parameter_is_tupled_arguments));
  return LoadInternal(std::move(cpu_executable), std::move(device_assignment));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::LoadInternal(
    std::shared_ptr<PjRtCpuExecutable> cpu_executable,
    std::shared_ptr<DeviceAssignment> device_assignment) {
  int num_replicas = cpu_executable->num_replicas();
  int num_partitions = cpu_executable->num_partitions();
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
  }
  const auto& result_shape = cpu_executable->cpu_executable()->result_shape();
  if (result_shape.IsTuple()) {
    for (auto& leaf_shape : result_shape.tuple_shapes()) {
      if (leaf_shape.IsTuple()) {
        return absl::InternalError(absl::StrCat(
            "Nested tuples are not supported with PjRtCpuClient. got: ",
            result_shape.ToString()));
      }
    }
  }
  return std::make_unique<PjRtCpuLoadedExecutable>(
      std::move(cpu_executable), std::move(device_assignment),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);
}

static absl::StatusOr<std::unique_ptr<xla::Executable>> JitCompile(
    std::unique_ptr<HloModule> hlo_module,
    const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options,
    const xla::Compiler::CompileOptions& compile_options) {
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
    std::unique_ptr<HloModule> hlo_module,
    const ExecutableBuildOptions& build_options,
    const ExecutionOptions& execution_options,
    const xla::AotCompilationOptions& compile_options) {
  cpu::CpuCompiler compiler;
  // TODO (basioli): honor build_options.run_backend_only() for AOT.
  // Compile AOT.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
      compiler.CompileAheadOfTime(std::move(hlo_module), compile_options));

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

  return std::move(*aot_result).LoadExecutable();
}

absl::StatusOr<std::pair<std::unique_ptr<PjRtCpuExecutable>,
                         std::shared_ptr<DeviceAssignment>>>
PjRtCpuClient::CompileAndAssignDevices(mlir::ModuleOp module,
                                       CompileOptions options) {
  TF_RETURN_IF_ERROR(pjrt::MaybeDumpCompileInputs(options, module, *topology_));

  XlaComputation xla_computation;
  ExecutableBuildOptions& exec_build_options = options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, &exec_build_options));

  if (options.argument_layouts) {
    return CompileAndAssignDevices(xla_computation, options);
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

absl::StatusOr<std::pair<std::unique_ptr<PjRtCpuExecutable>,
                         std::shared_ptr<DeviceAssignment>>>
PjRtCpuClient::CompileAndAssignDevices(const XlaComputation& computation,
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

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(auto results,
                      CompileAndAssignDevices(computation, std::move(options)));
  return std::move(results.first);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(auto results, CompileAndAssignDevices(
                                        std::move(module), std::move(options)));
  return std::move(results.first);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  TF_ASSIGN_OR_RETURN(auto results,
                      CompileAndAssignDevices(computation, std::move(options)));
  return LoadInternal(std::move(results.first), std::move(results.second));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(auto results, CompileAndAssignDevices(
                                        std::move(module), std::move(options)));
  return LoadInternal(std::move(results.first), std::move(results.second));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> PjRtCpuClient::Load(
    std::unique_ptr<PjRtExecutable> executable,
    const LoadOptions& load_options) {
  TF_RET_CHECK(tensorflow::down_cast<PjRtCpuExecutable*>(executable.get()));
  auto cpu_executable = absl::WrapUnique<PjRtCpuExecutable>(
      tensorflow::down_cast<PjRtCpuExecutable*>(executable.release()));
  CompileOptions options = cpu_executable->compile_options();
  int unused_num_replicas;
  int unused_num_partitions;
  std::shared_ptr<DeviceAssignment> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options.compile_portable_executable, &options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &unused_num_replicas, &unused_num_partitions, &device_assignment));
  return LoadInternal(std::move(cpu_executable), std::move(device_assignment));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtCpuClient::CompileAheadOfTimeAndLoad(
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
  TF_ASSIGN_OR_RETURN(
      auto results,
      CompileInternal(computation, argument_layout_pointers,
                      /*layout_canonicalization_callback=*/nullptr, options,
                      &aot_options));
  return LoadInternal(std::move(results.first), std::move(results.second));
}

absl::StatusOr<std::pair<std::unique_ptr<PjRtCpuExecutable>,
                         std::shared_ptr<DeviceAssignment>>>
PjRtCpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options,
    const AotCompilationOptions* absl_nullable aot_options) {
  tsl::profiler::TraceMe traceme("PjRtCpuClient::Compile");
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
  if (device_assignment != nullptr && build_options.device_ordinal() < 0) {
    TF_RETURN_IF_ERROR([&]() -> absl::Status {
      for (int replica = 0; replica < num_replicas; ++replica) {
        for (int partition = 0; partition < num_partitions; ++partition) {
          PjRtGlobalDeviceId device_id(
              (*device_assignment)(replica, partition));
          if (UnpackCpuProcessIndex(device_id) != process_index()) {
            VLOG(3) << "Non-local device: " << device_id;
            continue;
          }
          TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
          build_options.set_device_ordinal(device->local_hardware_id().value());
          return absl::OkStatus();
        }
      }
      return absl::OkStatus();
    }());
  }

  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());

  std::unique_ptr<Executable> cpu_executable;
  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  // Unoptimized HloModuleConfig.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModuleConfig> hlo_module_config,
      CreateModuleConfig(program_shape, argument_layout_pointers,
                         &execution_options, execution_options.num_replicas(),
                         eigen_intraop_device()->getPool()->NumThreads(),
                         aot_options));

  // Apply the user-provided callback to customize the HloModuleConfig.
  if (customize_hlo_module_config_) {
    customize_hlo_module_config_(*hlo_module_config);
  }

  // Unoptimized HloModule.
  const xla::HloModuleProto& hlo_module_proto = computation.proto();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> hlo_module,
      xla::HloModule::CreateFromProto(hlo_module_proto, *hlo_module_config));

  if (aot_options) {
    TF_ASSIGN_OR_RETURN(cpu_executable,
                        CompileAheadOfTime(std::move(hlo_module), build_options,
                                           execution_options, *aot_options));
  } else {
    xla::Compiler::CompileOptions compile_options{
        build_options.device_allocator(), build_options.compile_thread_pool(),
        build_options.layout_canonicalization_callback()};
    if (!compile_options.thread_pool) {
      compile_options.thread_pool = pjrt_client_thread_pool();
    }

    const cpu::TargetMachineOptions& target_machine_options =
        topology_->cpu_topology().target_machine_options();

    compile_options.cpu_target_config.emplace(target_machine_options);

    TF_ASSIGN_OR_RETURN(cpu_executable,
                        JitCompile(std::move(hlo_module), build_options,
                                   execution_options, compile_options));
  }

  auto cpu_executable_ptr =
      tsl::down_cast<cpu::CpuExecutable*>(cpu_executable.get());

  // `result_buffer_indices` has the buffer allocation indices that make up the
  // output buffer (could be tuple).
  TF_ASSIGN_OR_RETURN(
      auto result_buffer_indices,
      FindResultBufferAllocationIndex(cpu_executable_ptr->buffer_assignment(),
                                      cpu_executable->module()));

  std::unique_ptr<HloModule> unoptimized_hlo_module = nullptr;

  const bool xla_dump_hlo_unoptimized_snapshots =
      options.executable_build_options.has_debug_options() &&
      options.executable_build_options.debug_options()
          .xla_dump_hlo_unoptimized_snapshots();

  if (xla_dump_hlo_unoptimized_snapshots) {
    TF_ASSIGN_OR_RETURN(
        unoptimized_hlo_module,
        HloModule::CreateFromProto(computation.proto(),
                                   cpu_executable->module().config()));
  }

  auto executable = std::make_unique<PjRtCpuExecutable>(
      num_replicas, num_partitions, options.parameter_is_tupled_arguments,
      std::move(input_options), std::move(cpu_executable),
      std::move(result_buffer_indices), std::move(unoptimized_hlo_module));
  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));

  return std::make_pair(std::move(executable), std::move(device_assignment));
}

static bool IsAlignedData(void* ptr) {
  return (absl::bit_cast<std::uintptr_t>(ptr) & (cpu::MinAlign() - 1)) == 0;
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
PjRtCpuClient::ImportForeignMemory(
    void* device_ptr, absl::AnyInvocable<void() &&> on_delete_callback,
    size_t on_device_bytes_count, PjRtMemorySpace* memory_space,
    bool is_mutable) {
  return CpuRawBuffer::ImportForeignMemory(
      device_ptr, std::move(on_delete_callback), on_device_bytes_count,
      memory_space, is_mutable);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCpuClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices().front();
  if (device->client() != this) {
    return absl::InvalidArgumentError("Device is not attached to this client");
  }
  // Create a dummy buffer because the rest of the code expects a buffer
  // regardless of whether the definition event is an error.
  TF_ASSIGN_OR_RETURN(
      auto raw_buffer,
      CpuRawBuffer::Allocate(memory_space, ShapeUtil::ByteSizeOf(shape),
                             *allocator_));
  return std::make_unique<CommonPjRtBufferImpl>(
      shape,
      std::make_unique<TrackedCpuDeviceBuffer>(
          std::move(raw_buffer),
          tsl::AsyncValueRef<CpuEvent>(
              tsl::MakeErrorAsyncValueRef(std::move(error)))),
      memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
PjRtCpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  return xla::CreateAsyncHostToDeviceTransferManager(
      shape_specs, device_layouts, memory_space);
}

bool PjRtCpuClient::BufferFromHostBufferSupportsZeroCopy(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape,
    PjRtMemorySpace* memory_space, const Layout* device_layout) const {
  return AbstractCpuBuffer::BufferFromHostBufferSupportsZeroCopy(
      data, type, dims, byte_strides, shape);
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>>
PjRtCpuClient::LinearizeHostBufferInto(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    const xla::Shape& device_shape,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
  return tsl::down_cast<CpuRawBuffer*>(raw_buffer.get())
      ->CopyFromHostBuffer(
          data, type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), device_shape,
          async_work_runner(), &transpose_mu_, &transpose_cache_,
          eigen_intraop_pool(), max_transpose_threads_);
}

absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> PjRtCpuClient::LinearizeInto(
    const LiteralSlice& literal, const xla::Shape& device_shape,
    HostBufferSemantics host_buffer_semantics,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer) {
  if (host_buffer_semantics ==
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall) {
    return absl::UnimplementedError(
        "ImmutableOnlyDuringCall semantics is not supported on CPU.");
  }
  return tsl::down_cast<CpuRawBuffer*>(raw_buffer.get())
      ->CopyFromLiteral(literal, device_shape.layout(), async_work_runner());
}

absl::StatusOr<CompiledMemoryStats> PjRtCpuExecutable::GetCompiledMemoryStats()
    const {
  auto cpu_executable_ptr =
      tsl::down_cast<cpu::CpuExecutable*>(cpu_executable_.get());
  const auto& buffer_assignment = cpu_executable_ptr->buffer_assignment();
  auto proto = buffer_assignment.ToProto();

  CompiledMemoryStats memory_stats = CompiledMemoryStats();
  memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
  memory_stats.serialized_buffer_assignment = proto.SerializeAsString();
  memory_stats.PopulateBufferStatsFromAllocations(
      cpu_executable_->GetAllocations());
  TF_ASSIGN_OR_RETURN(memory_stats.peak_memory_in_bytes,
                      ComputePeakMemory(proto));
  return memory_stats;
}

absl::StatusOr<std::pair<tsl::RCReference<PjRtDeviceEventPromise>,
                         tsl::RCReference<PjRtDeviceEvent>>>
PjRtCpuClient::CreateLinkedEventPromise(PjRtMemorySpace* memory_space,
                                        absl::string_view debug_info) {
  auto definition_event_promise = tsl::MakeIndirectAsyncValue();
  auto definition_event = tsl::MakeRef<CpuTrackedDeviceEvent>(
      tsl::AsyncValueRef<CpuEvent>(definition_event_promise));
  return std::make_pair(tsl::MakeRef<CpuTrackedDeviceEventPromise>(
                            std::move(definition_event_promise)),
                        std::move(definition_event));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCpuClient::DefineBuffer(
    const Shape& on_device_shape, PjRtMemorySpace* memory_space,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    absl::InlinedVector<tsl::RCReference<PjRtDeviceEvent>, 4>
        definition_device_events) {
  if (raw_buffer && raw_buffer->memory_space() != memory_space) {
    return absl::InvalidArgumentError(
        absl::StrFormat("DefineBuffer: Mismatch in memory spaces: %s vs %s",
                        raw_buffer->memory_space()->DebugString(),
                        memory_space->DebugString()));
  }
  return std::unique_ptr<PjRtBuffer>(std::make_unique<CommonPjRtBufferImpl>(
      on_device_shape,
      std::make_unique<TrackedCpuDeviceBuffer>(
          std::move(raw_buffer),
          CpuTrackedDeviceEvent::AfterAll(definition_device_events)),
      memory_space));
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
PjRtCpuClient::AllocateRawBuffer(PjRtMemorySpace* memory_space,
                                 size_t on_device_bytes_count,
                                 bool retry_on_oom,
                                 tsl::AsyncValueRef<bool> allocate_after) {
  CHECK(allocate_after == nullptr) << "allocate_after is not supported for "
                                      "PjRtCpuClient.";
  return xla::CpuRawBuffer::Allocate(memory_space, on_device_bytes_count,
                                     *allocator_);
}

absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>>
PjRtCpuClient::AllocateRawBufferForExecute(PjRtMemorySpace* memory_space,
                                           size_t on_device_bytes_count,
                                           bool retry_on_oom) {
  return tsl::MakeRef<CpuRawBuffer>(memory_space,
                                    CpuDeviceMemory::CreateDelayedMemory(),
                                    on_device_bytes_count,
                                    /*owns_buffer=*/true);
}

absl::StatusOr<std::pair<tsl::RCReference<CommonPjRtRawBuffer>,
                         CommonPjRtClient::PjRtFulfillAliasRawBufferCallback>>
PjRtCpuClient::CreateRawBufferChannel(PjRtMemorySpace* memory_space,
                                      size_t on_device_bytes_count) {
  auto buffer_promise = tsl::MakeIndirectAsyncValue();
  auto raw_buffer = tsl::MakeRef<CpuRawBuffer>(
      memory_space, tsl::AsyncValueRef<CpuDeviceMemory>(buffer_promise),
      on_device_bytes_count, /*is_mutable=*/true);

  auto buffer_promise_cb =
      [buffer_promise = std::move(buffer_promise), memory_space](
          absl::StatusOr<tsl::RCReference<CommonPjRtRawBuffer>> raw_buffer)
      -> absl::Status {
    if (!raw_buffer.ok()) {
      buffer_promise->SetError(raw_buffer.status());
      return raw_buffer.status();
    }
    if (memory_space != (*raw_buffer)->memory_space()) {
      auto status = absl::InvalidArgumentError(absl::StrFormat(
          "Memory space mismatch when forarding raw buffers: %s vs %s",
          memory_space->DebugString(),
          (*raw_buffer)->memory_space()->DebugString()));
      buffer_promise->SetError(status);
      return status;
    }
    buffer_promise->ForwardTo(
        tensorflow::down_cast<xla::CpuRawBuffer*>(raw_buffer->get())
            ->buffer()
            .CopyRCRef());
    return absl::OkStatus();
  };

  return std::make_pair(std::move(raw_buffer), std::move(buffer_promise_cb));
}

absl::StatusOr<int64_t> PjRtCpuClient::GetOnDeviceBytesCount(
    PjRtMemorySpace* memory_space, const xla::Shape& shape) const {
  return xla::ShapeUtil::ByteSizeOf(shape);
}

absl::StatusOr<xla::Shape> PjRtCpuClient::MakeDefaultShapeForMemorySpace(
    PjRtMemorySpace* memory_space, xla::Shape shape,
    const xla::Layout* layout) const {
  return MakeDefaultCpuBufferShape(std::move(shape), layout);
}

static std::vector<tsl::RCReference<tsl::AsyncValue>> CopyAsyncValues(
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events) {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev.CopyRef());
  }
  return avs;
}

static std::vector<Shape> GetParameterShapes(const ComputationLayout& layout) {
  // For now, TPU programs compiled with multiple arguments cannot use tuples
  // for any of their arguments, so we can assume that a tuple can only arise
  // when there is a single argument.
  std::vector<Shape> shapes;
  if (layout.parameter_count() == 1 && layout.parameter_shape(0).IsTuple()) {
    shapes.reserve(layout.parameter_shape(0).tuple_shapes().size());
    absl::c_copy(layout.parameter_shape(0).tuple_shapes(),
                 std::back_inserter(shapes));
  } else {
    shapes.reserve(layout.parameter_count());
    for (const ShapeLayout& sl : layout.parameter_layouts()) {
      shapes.push_back(sl.shape());
    }
  }
  return shapes;
}

PjRtCpuExecutable::PjRtCpuExecutable(
    int num_replicas, int num_partitions, bool parameter_is_tupled_arguments,
    CompileOptions compile_options, std::unique_ptr<Executable> cpu_executable,
    absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
    std::unique_ptr<HloModule> unoptimized_hlo_module)
    : num_replicas_(num_replicas),
      num_partitions_(num_partitions),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      compile_options_(std::move(compile_options)),
      cpu_executable_(std::move(cpu_executable)),
      parameter_device_shapes_(GetParameterShapes(
          cpu_executable_->module().entry_computation_layout())),
      result_buffer_indices_(std::move(result_buffer_indices)),
      unoptimized_hlo_module_(std::move(unoptimized_hlo_module)) {
  auto hlo_cost_analysis =
      std::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
  CHECK_OK(cpu_executable_->module().entry_computation()->Accept(
      hlo_cost_analysis.get()));
  // Cache to avoid std::map lookup in flop_count() on critical path. The magic
  // constant 1000 is determined by correlating computation with flop estimate.
  // It is a crude heuristic to find computation less than the thread context
  // switch time (~5us).
  cheap_computation_ = hlo_cost_analysis->flop_count() < 1000;

  output_memory_space_kind_ids_.resize(result_buffer_indices_.size(),
                                       CpuDeviceMemorySpace::kKindId);
  output_indices_.resize(
      tsl::down_pointer_cast<cpu::CpuExecutable>(cpu_executable_)
          ->buffer_assignment()
          .Allocations()
          .size(),
      -1);
  for (int i = 0; i < result_buffer_indices_.size(); ++i) {
    CHECK_LT(result_buffer_indices_[i], output_indices_.size());
    CHECK_EQ(output_indices_[result_buffer_indices_[i]], -1)
        << "Unexpected duplicate.";
    output_indices_[result_buffer_indices_[i]] = i;
  }
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

PjRtCpuLoadedExecutable::PjRtCpuLoadedExecutable(
    std::shared_ptr<PjRtCpuExecutable> executable,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, PjRtCpuClient* client)
    : client_(client),
      executable_(std::move(executable)),
      device_assignment_(std::move(device_assignment)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {}

void PjRtCpuLoadedExecutable::Delete() {}

bool PjRtCpuLoadedExecutable::IsDeleted() const { return false; }

absl::Status PjRtCpuLoadedExecutable::SetUpDonation(bool tuple_inputs) {
  return executable_->SetUpDonation(tuple_inputs);
}

absl::Status PjRtCpuExecutable::SetUpDonation(bool tuple_inputs) {
  TF_ASSIGN_OR_RETURN(parameters_that_must_be_donated_,
                      ComputeParametersThatMustBeDonated(
                          *cpu_executable_->shared_module(), tuple_inputs));
  return absl::OkStatus();
}

namespace {

// Some helper structs to support delayed memory allocation.

struct BufferAlloc {
  // All data members should have the same size.
  absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> buffers;
  absl::InlinedVector<size_t, 4> allocation_sizes;

  void Allocate(const CpuDeviceMemory::Allocator& allocator) {
    for (int i = 0; i < buffers.size(); ++i) {
      auto status = CpuDeviceMemory::AllocateInto(
          allocation_sizes[i], buffers[i].AsPtr(), allocator);
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

  void AllocateAndCopy(const CpuDeviceMemory::Allocator& allocator) {
    for (int i = 0; i < src_buffers.size(); ++i) {
      auto status = CpuDeviceMemory::AllocateInto(
          allocation_sizes[i], dst_buffers[i].AsPtr(), allocator);
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
static absl::StatusOr<tsl::AsyncValueRef<CpuDeviceMemory>> MemoryForAllocation(
    const BufferAllocation& allocation,
    absl::Span<const cpu::ConstantAllocation> constants,
    absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> input_buffers,
    BufferAlloc& buffer_alloc, BufferAllocAndCopy& buffer_alloc_and_copy,
    const tsl::AsyncValueRef<CpuDeviceMemory>& tuple_index_table,
    const tsl::RCReference<CommonPjRtRawBuffer>& allocated_output) {
  auto buffer_or_default = [&]() -> tsl::AsyncValueRef<CpuDeviceMemory> {
    if (allocated_output) {
      return tensorflow::down_cast<CpuRawBuffer*>(allocated_output.get())
          ->buffer();
    }
    return CpuDeviceMemory::CreateDelayedMemory();
  };
  if (allocation.is_entry_computation_parameter()) {
    CpuRawBuffer* arg = nullptr;
    if (tuple_index_table) {
      if (allocation.param_shape_index().empty()) {
        return tuple_index_table;
      } else if (allocation.param_shape_index().size() == 1) {
        arg = tensorflow::down_cast<CpuRawBuffer*>(
            input_buffers[allocation.param_shape_index()[0]].get());
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
      arg = tensorflow::down_cast<CpuRawBuffer*>(
          input_buffers[allocation.parameter_number()].get());
    }
    if (allocated_output) {
      if (arg != allocated_output.get()) {
        auto copy = tensorflow::down_cast<CpuRawBuffer*>(allocated_output.get())
                        ->buffer();
        buffer_alloc_and_copy.src_buffers.push_back(arg->buffer());
        buffer_alloc_and_copy.dst_buffers.push_back(copy);
        buffer_alloc_and_copy.allocation_sizes.push_back(allocation.size());
        return copy;
      }
      return tensorflow::down_cast<CpuRawBuffer*>(allocated_output.get())
          ->buffer();
    }
    return arg->buffer();
  } else if (allocation.is_constant() &&
             allocation.index() < constants.size()) {
    se::DeviceAddressBase constant =
        constants[allocation.index()].AsDeviceAddress();
    return CpuDeviceMemory::CreateConstantMemory(constant.opaque(),
                                                 constant.size());
  } else if (allocation.is_constant() || allocation.is_thread_local()) {
    return CpuDeviceMemory::CreateConstantMemory(nullptr, 0);
  }

  // Output and temporary buffer.
  auto out = buffer_or_default();

  buffer_alloc.buffers.push_back(out);
  buffer_alloc.allocation_sizes.push_back(allocation.size());

  return out;
}

static absl::StatusOr<std::vector<tsl::AsyncValueRef<CpuDeviceMemory>>>
CreateBufferTable(
    const BufferAssignment& assignment,
    absl::Span<const cpu::ConstantAllocation> constants,
    absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>> input_buffers,
    BufferAlloc& buffer_alloc, BufferAllocAndCopy& buffer_alloc_and_copy,
    const tsl::AsyncValueRef<CpuDeviceMemory>& tuple_index_table,
    const absl::Span<const tsl::RCReference<CommonPjRtRawBuffer>>
        output_buffers,
    absl::Span<const int64_t> output_indices) {
  std::vector<tsl::AsyncValueRef<CpuDeviceMemory>> buffer_table(
      assignment.Allocations().size());
  tsl::RCReference<CommonPjRtRawBuffer> null_output;
  for (BufferAllocation::Index i = 0; i < buffer_table.size(); ++i) {
    const BufferAllocation& allocation = assignment.GetAllocation(i);
    int64_t out_index = output_indices[i];
    TF_ASSIGN_OR_RETURN(
        buffer_table[i],
        MemoryForAllocation(
            allocation, constants, input_buffers, buffer_alloc,
            buffer_alloc_and_copy, tuple_index_table,
            out_index != -1 ? output_buffers[out_index] : null_output));
  }

  return std::move(buffer_table);
}

absl::Status PjRtCpuLoadedExecutable::CheckBufferCompatibilities(
    absl::Span<const CommonPjRtBuffer::ScopedHold> input_buffers,
    absl::Span<PjRtBuffer* const> argument_handles) const {
  if (input_buffers.size() !=
      executable_->input_buffer_sizes_in_bytes_.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), executable_->input_buffer_sizes_in_bytes_.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    auto* buffer = tensorflow::down_cast<TrackedCpuDeviceBuffer*>(
        input_buffers[i].buffer());
    if (executable_->input_buffer_sizes_in_bytes_[i] != buffer->BufferSize()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with "
          "incompatible size %lld",
          i, executable_->input_buffer_sizes_in_bytes_[i],
          buffer->BufferSize());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CpuPjRtRawLoadedExecutable>>
PjRtCpuLoadedExecutable::StartRawExecutable(
    const ExecuteOptions& options,
    PjRtCpuClient::CollectiveLaunchEvent last_collective_launch_event,
    const RunId& run_id, int replica, int partition, PjRtDevice* device) const {
  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int64_t device_id = (*device_assignment_)(replica, partition);
    PjRtGlobalDeviceId global_device_id(device_id);
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(global_device_id));
    device = tsl::down_cast<PjRtCpuDevice*>(pjrt_device);
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
  auto result = std::make_unique<CpuPjRtRawLoadedExecutable>(run_id);
  result->last_collective_launch_event_ =
      std::move(last_collective_launch_event);
  result->executable_ = executable_.get();
  result->client_ = client_;
  result->device_assignment_ = device_assignment;
  result->device_ = tsl::down_cast<PjRtCpuDevice*>(device);
  return result;
}

absl::StatusOr<PjRtLoadedExecutable::Result>
PjRtCpuLoadedExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    PjRtCpuClient::CollectiveLaunchEvent last_collective_launch_event,
    bool fill_future, PjRtCpuDevice* device) const {
  tsl::profiler::TraceMe traceme([&]() {
    return tsl::profiler::TraceMeEncode(
        "PjRtCpuLoadedExecutable::ExecuteHelper",
        {
            {"run_id", run_id.ToInt()},
            {"replica", replica},
            {"partition", partition},
        });
  });

  TF_ASSIGN_OR_RETURN(
      auto executable,
      StartRawExecutable(options, std::move(last_collective_launch_event),
                         run_id, replica, partition, device));
  device = tsl::down_cast<PjRtCpuDevice*>(executable->device());

  bool is_error = false;
  absl::InlinedVector<CommonPjRtBuffer::ScopedHold, 4> device_buffers;
  absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4> input_buffers;
  CpuTrackedDeviceEventSet input_deps(argument_handles.size());
  TF_RETURN_IF_ERROR(client()->PrepareArguments(
      options, argument_handles, executable_->parameters_that_must_be_donated_,
      input_deps, input_deps, input_buffers, device_buffers,
      executable->device(), replica, partition,
      executable_->parameter_device_shapes_, is_error,
      /*allow_fallback_for_donation=*/true));

  CHECK(!is_error) << "CpuClient does not support is_error.";
  TF_RETURN_IF_ERROR(
      CheckBufferCompatibilities(device_buffers, argument_handles));

  auto cpu_executable =
      tsl::down_pointer_cast<cpu::CpuExecutable>(executable_->cpu_executable_);
  TF_ASSIGN_OR_RETURN(
      auto output_leaf_buffers,
      client_->AllocateOutputBuffersWithInputReuse(
          executable_->cpu_executable_->result_shape(), device_buffers,
          executable_->cpu_executable_->module().input_output_alias_config(),
          device, executable_->output_memory_space_kind_ids_));

  PjRtRawLoadedExecutable::RawExecuteResult result;
  absl::Status inline_result_status;

  TF_RETURN_IF_ERROR(std::move(*executable)
                         .Execute(options, result, inline_result_status,
                                  input_buffers, output_leaf_buffers,
                                  input_deps, fill_future));

  for (CommonPjRtBuffer::ScopedHold& b : device_buffers) {
    if (b.type() == CommonPjRtBuffer::ScopedHold::kUsage) {
      b.ConvertUsageHold(result.primary_execute_event);
    } else {
      CHECK(b.type() == CommonPjRtBuffer::ScopedHold::kDonation);
      b.ConfirmDonation();
    }
  }

  TF_RETURN_IF_ERROR(inline_result_status);

  auto res =
      client_->CreateOutputs(executable_->cpu_executable_->result_shape(),
                             std::move(result.primary_execute_event), device,
                             executable_->output_memory_space_kind_ids_,
                             std::move(output_leaf_buffers),
                             /*is_predetermined_error=*/false);

  return Result({std::move(result.future), std::move(res)});
}

absl::Status CpuPjRtRawLoadedExecutable::Execute(
    const ExecuteOptions& options,
    PjRtRawLoadedExecutable::RawExecuteResult& result,
    absl::Status& inline_result_status,
    absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>&
        input_buffers,
    absl::InlinedVector<tsl::RCReference<CommonPjRtRawBuffer>, 4>&
        output_leaf_buffers,
    PjRtDeviceEventSet& generic_input_deps, bool fill_future) && {
  // `returned_future_can_be_set_event` indicates when `returned_future` can be
  // set using `execute_event`. This is necessary to delay setting the
  // `returned_future` until all (async) execution activities are complete even
  // if `execute_event` itself may be set early due to execution poisoning. This
  // lets the user rely on `returned_future` when there is no more in-flight
  // executions and destroy any external resources such as loaded callbacks and
  // execute contexts.
  auto returned_future_can_be_set_event =
      tsl::MakeConstructedAsyncValueRef<CpuEvent>();

  auto& input_deps =
      *tensorflow::down_cast<CpuTrackedDeviceEventSet*>(&generic_input_deps);
  auto execute_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  MarkEventReadyOnExit ready_on_exit(execute_event);
  result.primary_execute_event =
      tsl::MakeRef<CpuTrackedDeviceEvent>(execute_event);

  auto cpu_executable =
      tsl::down_pointer_cast<cpu::CpuExecutable>(executable_->cpu_executable_);
  auto client = client_;

  // Tuplize the inputs if compiler expects a single tuple argument but runtime
  // gets many inputs that are not yet tupled.
  tsl::AsyncValueRef<CpuDeviceMemory> tuple_index_table;
  if (executable_->parameter_is_tupled_arguments_) {
    absl::InlinedVector<tsl::AsyncValueRef<CpuDeviceMemory>, 4> leaf_buffers;
    leaf_buffers.reserve(input_buffers.size());
    for (const auto& buffer : input_buffers) {
      leaf_buffers.push_back(
          tensorflow::down_cast<CpuRawBuffer*>(buffer.get())->buffer());
    }
    tuple_index_table = CpuDeviceMemory::CreateDelayedMemory();
    tsl::RunWhenReady(
        absl::MakeConstSpan(leaf_buffers),
        [buffers = leaf_buffers, tuple_index_table,
         allocator = client->allocator()]() mutable {
          size_t index_table_byte_size = buffers.size() * sizeof(void*);
          // We assume tuple table allocations will not fail.
          CHECK_OK(CpuDeviceMemory::AllocateInto(
              index_table_byte_size, tuple_index_table.AsPtr(), *allocator));
          uintptr_t* index_table =
              reinterpret_cast<uintptr_t*>(tuple_index_table->untyped_data());
          for (int i = 0; i < buffers.size(); ++i) {
            index_table[i] =
                absl::bit_cast<uintptr_t>(buffers[i]->untyped_data());
          }
        });
  }

  // `buffer_alloc` and `buffer_alloc_and_copy` are used to do real memory
  // allocation and copy work.
  BufferAlloc buffer_alloc;
  BufferAllocAndCopy buffer_alloc_and_copy;

  TF_ASSIGN_OR_RETURN(
      std::vector<tsl::AsyncValueRef<CpuDeviceMemory>> buffer_table,
      CreateBufferTable(cpu_executable->buffer_assignment(),
                        cpu_executable->constants(), input_buffers,
                        buffer_alloc, buffer_alloc_and_copy, tuple_index_table,
                        output_leaf_buffers, executable_->output_indices_));

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  auto compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
      device_->max_inflight_computations_semaphore().ScopedAcquire(1));

  ExecutableRunOptions run_options;
  run_options.set_run_id(run_id_);
  // Need to keep device_assignment alive until execution completes.
  run_options.set_device_assignment(device_assignment_.get());
  run_options.set_intra_op_thread_pool(client->eigen_intraop_device());

  auto cpu_run_options = std::make_unique<cpu::CpuExecutableRunOptions>();
  run_options.set_cpu_executable_run_options(cpu_run_options.get());

  const CpuExecuteContext* cpu_execute_context =
      options.context == nullptr
          ? nullptr
          : dynamic_cast<const CpuExecuteContext*>(options.context);
  if (cpu_execute_context != nullptr &&
      cpu_execute_context->process_index().has_value()) {
    run_options.set_device_ordinal(
        PackCpuDeviceId(*cpu_execute_context->process_index(),
                        UnpackCpuLocalDeviceId(device_->global_device_id()))
            .value());
  } else {
    run_options.set_device_ordinal(device_->global_device_id().value());
  }
  if (cpu_execute_context != nullptr &&
      cpu_execute_context->collectives() != nullptr) {
    cpu_run_options->set_collectives(cpu_execute_context->collectives());
  } else {
    cpu_run_options->set_collectives(client->collectives_.get());
  }

  // Schedule only one collective at a time.
  bool is_a_collective_launch =
      static_cast<bool>(last_collective_launch_event_.first);
  // Add additional dependency conditioned on whether this is a collective
  // launch or not.
  if (is_a_collective_launch) {
    input_deps.AddEvent(std::move(last_collective_launch_event_.first));
  } else {
    // This is a non-parallel computation. Add the last enqueue event as a
    // dependency with any error cleared.
    auto last_enqueue_event = device_->stream_event_map()->GetLastEnqueueEvent(
        options.execution_stream_id);
    if (!last_enqueue_event.IsAvailable()) {
      auto last_enqueue_done_event =
          tsl::MakeUnconstructedAsyncValueRef<CpuEvent>();
      last_enqueue_event.AndThen(
          [last_enqueue_done_event = last_enqueue_done_event.CopyRef()]() {
            last_enqueue_done_event.emplace();
          });
      input_deps.AddEvent(std::move(last_enqueue_done_event));
    }
  }
  if (options.context != nullptr) {
    run_options.set_ffi_execution_context(&options.context->ffi_context());
  }

  bool execute_inline = executable_->cheap_computation_ ||
                        !client->asynchronous_ ||
                        ThisThreadIsInsideHostCallback();

  // Overwrite `execute_inline` if it is specified in the ExecuteOptions.
  if (options.execution_mode == ExecuteOptions::ExecutionMode::kAsynchronous) {
    execute_inline = false;
  } else if (options.execution_mode ==
             ExecuteOptions::ExecutionMode::kSynchronous) {
    execute_inline = true;
  }

  auto execute_thunks = [cpu_executable, buffer_table = std::move(buffer_table),
                         eigen_device = client->eigen_intraop_device(),
                         run_options = std::move(run_options)]()
      -> absl::StatusOr<tsl::AsyncValueRef<cpu::Thunk::ExecuteEvent>> {
    // Set denormal and rounding behavior to match the default TF
    // ThreadPool behavior.
    tsl::port::ScopedFlushDenormal flush;
    tsl::port::ScopedSetRound round(FE_TONEAREST);

    // Immediately allocate memory and prepare for computation.
    for (const auto& buffer : buffer_table) {
      CHECK(buffer.IsAvailable());
      if (buffer.IsError()) {
        return buffer.GetError();
      }
    }

    if (!cpu_executable->has_thunks()) {
      return Internal("CpuExecutable has no thunks.");
    }
    // Call interpreted thunk sequence implementing XLA executable.
    absl::InlinedVector<MaybeOwningDeviceAddress, 8> buffer_device_mem;
    buffer_device_mem.reserve(buffer_table.size());
    for (const auto& buffer : buffer_table) {
      buffer_device_mem.emplace_back(
          se::DeviceAddressBase(buffer->untyped_data(), buffer->size_bytes()));
    }

    cpu::BufferAllocations allocations(buffer_device_mem);

    TF_ASSIGN_OR_RETURN(
        cpu::Thunk::CollectiveExecuteParams collective_params,
        cpu::Thunk::CollectiveExecuteParams::Create(&run_options));

    TF_ASSIGN_OR_RETURN(
        cpu::Thunk::CustomCallExecuteParams custom_call_execute_params,
        cpu::Thunk::CustomCallExecuteParams::Create(&run_options));

    std::optional<cpu::Thunk::YnnParams> ynn_params;
    if (cpu_executable->has_ynn_fusions()) {
      TF_ASSIGN_OR_RETURN(ynn_params,
                          cpu::Thunk::YnnParams::Create(&run_options));
    }

    cpu::ThreadPoolTaskRunner task_runner(
        run_options.intra_op_thread_pool()->getPool());

    cpu::Thunk::ExecuteParams execute_params = {
        cpu_executable->function_library(),
        &allocations,
        cpu::GetXfeedManager(run_options.device_ordinal()),
        run_options.intra_op_thread_pool(),
        &task_runner,
        &collective_params,
        &custom_call_execute_params,
        ynn_params ? &*ynn_params : nullptr,
        run_options.run_id().ToInt(),
        run_options.device_ordinal(),
    };

    auto thunks_execute_event =
        cpu_executable->thunks().Execute(execute_params);

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "ThunkExecutor::Execute (wait for completion)",
          {{"run_id", run_options.run_id().ToInt()},
           {"device_ordinal", run_options.device_ordinal()}});
    });
    tsl::BlockUntilReady(thunks_execute_event);
    return thunks_execute_event;
  };

  if (input_deps.events().empty() && execute_inline) {
    // Synchronously call generated function or thunk sequence.
    buffer_alloc.Allocate(*client->allocator());
    buffer_alloc_and_copy.AllocateAndCopy(*client->allocator());

    TF_ASSIGN_OR_RETURN(auto thunks_execute_event, execute_thunks());

    if (thunks_execute_event.IsError()) {
      inline_result_status = thunks_execute_event.GetError();
    } else {
      returned_future_can_be_set_event.SetStateConcrete();
    }

  } else {
    // Asynchronously call generated function.

    // We only created enough threads for one collective to complete.
    // The next collective launch will not be scheduled onto threadpool until
    // this one completes.
    if (is_a_collective_launch) {
      execute_event.AndThen(
          [count_down = last_collective_launch_event_.second]() mutable {
            count_down.CountDown();
          });
    } else {
      // This is a non-parallel computation. Set the execute event as the new
      // last enqueue event.
      auto* stream_event_map = device_->stream_event_map();
      stream_event_map->SetLastEnqueueEvent(options.execution_stream_id,
                                            execute_event.CopyRef());
      execute_event.AndThen([stream_event_map,
                             execution_stream_id = options.execution_stream_id,
                             self = execute_event.AsPtr()]() {
        stream_event_map->Clear(execution_stream_id, self);
      });
    }
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events_avs_ref =
        input_deps.events();
    CpuScopedAsyncExecution scoped_async_execution =
        device_->async_execution_tracker()->NewAsyncExecution(
            run_id_.ToInt(), std::move(ready_on_exit).Release());
    client->async_work_runner()->ScheduleWhenReady(
        events_avs_ref,
        [cpu_executable, buffer_alloc = std::move(buffer_alloc),
         buffer_alloc_and_copy = std::move(buffer_alloc_and_copy),
         execute_thunks = std::move(execute_thunks),
         device_assignment = std::move(device_assignment_),
         cpu_run_options = std::move(cpu_run_options),
         compute_reservation = std::move(compute_reservation),
         tuple_index_table = std::move(tuple_index_table),
         scoped_async_execution = std::move(scoped_async_execution),
         input_deps_avs = std::move(input_deps).Consume(),
         allocator = client->allocator(),
         returned_future_can_be_set_event =
             returned_future_can_be_set_event.CopyRef()]() mutable {
          // Because `input_deps` contains the definition events of all inputs,
          // when it is ready, all input buffers must have been allocated. So,
          // we are safe to allocate and copy memory here. Since `execute_event`
          // may error out, we need to do it early.
          buffer_alloc.Allocate(*allocator);
          buffer_alloc_and_copy.AllocateAndCopy(*allocator);

          for (const auto& av : input_deps_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              scoped_async_execution.SetError(Internal(
                  "Error dispatching computation: %s", error->message()));
              returned_future_can_be_set_event.SetStateConcrete();
              return;
            }
          }
          auto status = [&]() -> absl::Status {
            TF_ASSIGN_OR_RETURN(auto thunks_execute_event, execute_thunks());
            if (thunks_execute_event.IsError()) {
              return thunks_execute_event.GetError();
            }
            return absl::OkStatus();
          }();

          if (!status.ok()) {
            // CPU computation fails with an error.
            scoped_async_execution.SetError(std::move(status));
            returned_future_can_be_set_event.SetStateConcrete();
            return;
          }

          // CPU computation completes.
          scoped_async_execution.SetStateConcrete();
          returned_future_can_be_set_event.SetStateConcrete();
        });
  }

  if (fill_future && inline_result_status.ok()) {
    auto [promise, future] = MakePromise<>();
    returned_future_can_be_set_event.AndThen(
        [execute_event = std::move(execute_event),
         promise = std::move(promise)]() mutable {
          execute_event.AndThen([execute_event = execute_event.CopyRef(),
                                 promise = std::move(promise)]() mutable {
            if (auto* error = execute_event.GetErrorIfPresent()) {
              promise.Set(*error);
            } else {
              promise.Set();
            }
          });
        });
    result.future = std::move(future);
  }

  return absl::OkStatus();
}

static void MaybeDumpHloSnapshot(
    const HloModule& module, RunId run_id,
    const std::vector<PjRtBuffer*>& arguments,
    const std::vector<std::unique_ptr<PjRtBuffer>>& results,
    absl::string_view file_name_prefix = "") {
  if (!DumpingEnabledForHloModule(module)) {
    return;
  }
  if (!module.config().debug_options().xla_dump_hlo_snapshots()) {
    return;
  }
  xla::HloSnapshot hlo_snapshot;
  *hlo_snapshot.mutable_hlo()->mutable_hlo_module() = module.ToProto();

  for (auto* argument : arguments) {
    *hlo_snapshot.add_arguments() = (*argument->ToLiteral().Await())->ToProto();
  }

  // If there are multiple results, wrap them in a tuple.
  if (results.size() == 1) {
    *hlo_snapshot.mutable_result() =
        (*results[0]->ToLiteral().Await())->ToProto();
  } else {
    std::vector<Literal> result_literals;
    result_literals.reserve(results.size());
    for (auto& result : results) {
      result_literals.push_back(std::move(**result->ToLiteral().Await()));
    }
    *hlo_snapshot.mutable_result() =
        LiteralUtil::MakeTupleOwned(std::move(result_literals)).ToProto();
  }

  DumpToFileInDir(
      module, "",
      absl::StrCat(file_name_prefix, "snapshot.", run_id.ToInt(), ".pb"),
      hlo_snapshot.SerializeAsString());
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtCpuLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMe trace_me("PjRtCpuLoadedExecutable::Execute");
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
    MaybeDumpHloSnapshot(executable_->cpu_executable_->module(), run_id,
                         argument_handles[0], {});
    if (executable_->unoptimized_hlo_module_ != nullptr) {
      HloUnoptimizedSnapshot hlo_snapshot;
      *hlo_snapshot.mutable_hlo_module() =
          executable_->unoptimized_hlo_module_->ToProto();
      for (const auto& argument_handle : argument_handles) {
        HloInputs hlo_inputs;
        for (const auto& buffer : argument_handle) {
          TF_ASSIGN_OR_RETURN(auto literal, buffer->ToLiteral().Await());
          *hlo_inputs.add_arguments() = literal->ToProto();
        }
        *hlo_snapshot.add_partitions() = std::move(hlo_inputs);
      }

      DumpHloUnoptimizedSnapshotIfEnabled(
          hlo_snapshot,
          executable_->cpu_executable_->module().config().debug_options());
    }
    auto statusor = ExecuteHelper(
        argument_handles[0], replica, partition, run_id, options,
        /*last_collective_launch_event=*/{}, returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    MaybeDumpHloSnapshot(executable_->cpu_executable_->module(), run_id,
                         argument_handles[0], wrapped_results[0]);
  } else {
    // Gang schedule collectives to ensure that collectives with the same RunId
    // are run at the same time. We conservatively run only one collective at a
    // time, because we may not have enough threads to run arbitrary number of
    // collectives concurrently.
    PjRtCpuClient::CollectiveLaunchEvent last_collective_launch_event =
        client()->GetLastCollectiveLaunchEvent(num_addressable_devices);

    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;

      client()->async_work_runner()->Schedule([&, replica, partition, i] {
        auto statusor = ExecuteHelper(
            argument_handles[i], replica, partition, run_id, options,
            last_collective_launch_event, returned_futures.has_value());
        if (statusor.ok()) {
          wrapped_results[i] = std::move(statusor->buffers);
          if (returned_futures.has_value()) {
            (*returned_futures)[i] = std::move(*statusor->future);
          }
        }

        absl::MutexLock lock(mu);
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
      absl::MutexLock lock(mu);
      mu.Await(absl::Condition(&done_running));
    }

    if (!first_failure_status.ok()) return first_failure_status;
  }
  VLOG(1) << "Replicated execution complete.";

  return wrapped_results;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCpuLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMe trace_me("PjRtCpuLoadedExecutable::ExecuteSharded");
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
              addressable_device_logical_ids_[i].partition, run_id, options,
              /*last_collective_launch_event=*/{}, fill_future));
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
PjRtCpuLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMe trace_me("PjRtCpuLoadedExecutable::ExecutePortable");
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
      ExecuteHelper(argument_handles,
                    /*replica=*/0,
                    /*partition=*/0, run_id, options,
                    /*last_collective_launch_event=*/{}, fill_future,
                    tsl::down_cast<PjRtCpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}
}  // namespace xla
