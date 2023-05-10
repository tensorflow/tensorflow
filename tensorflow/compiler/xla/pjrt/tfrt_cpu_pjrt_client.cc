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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/compile_options.pb.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/runtime/cpu_event.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/casts.h"
#include "tensorflow/tsl/platform/denormal.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/setround.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/profiler/lib/connected_traceme.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace xla {
namespace {

using ::xla::runtime::CpuEvent;

StatusOr<std::unique_ptr<TfrtCpuBuffer>> AllocateDestinationBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events,
    TfrtCpuDevice* device, TfrtCpuClient* client) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      AbstractTfrtCpuBuffer::AllocateTrackedDeviceBuffer(
          on_device_shape, std::move(definition_events)));
  return std::make_unique<TfrtCpuBuffer>(
      on_device_shape, std::move(tracked_device_buffer), client, device);
}

StatusOr<std::unique_ptr<TfrtCpuBuffer>> AllocateDestinationBufferAndAvs(
    const Shape& shape,
    absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4>* avs,
    TfrtCpuDevice* device, TfrtCpuClient* client) {
  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
  AbstractTfrtCpuBuffer::AllocateAvsAndEvents(shape, avs, &definition_events);
  return AllocateDestinationBuffer(
      shape, std::move(definition_events),
      tensorflow::down_cast<TfrtCpuDevice*>(device), client);
}

const char kCpuPlatformName[] = "cpu";

void EnqueueWork(tsl::thread::ThreadPool* pool,
                 absl::AnyInvocable<void()> callee) {
  // TSL TheadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool->Schedule([ptr = new absl::AnyInvocable<void()>(std::move(callee))]() {
    (*ptr)();
    delete ptr;
  });
}

// Enqueue to PjRtClient pool when all `values` are ready.
void EnqueueWorkWhenReady(
    tsl::thread::ThreadPool* pool,
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
    absl::AnyInvocable<void()> callee) {
  RunWhenReady(values, [pool, callee = std::move(callee)]() mutable {
    EnqueueWork(pool, std::move(callee));
  });
}

class ThreadPoolAsyncWorkRunner : public AsyncWorkRunner {
 public:
  explicit ThreadPoolAsyncWorkRunner(tsl::thread::ThreadPool* pool)
      : pool_(pool) {}

  void Schedule(absl::AnyInvocable<void()> work) override {
    EnqueueWork(pool_, std::move(work));
  }

  void ScheduleWhenReady(
      absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
      absl::AnyInvocable<void()> work) override {
    EnqueueWorkWhenReady(pool_, values, std::move(work));
  }

 private:
  tsl::thread::ThreadPool* pool_;
};

class TfrtCpuAsyncHostToDeviceTransferManager
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static StatusOr<std::unique_ptr<TfrtCpuAsyncHostToDeviceTransferManager>>
  Create(absl::Span<const Shape> shapes, TfrtCpuDevice* device,
         TfrtCpuClient* client) {
    for (const Shape& shape : shapes) {
      if (shape.IsTuple()) {
        return Unimplemented(
            "Tuples are not supported by "
            "TfrtCpuAsyncHostToDeviceTransferManager");
      }
    }
    absl::InlinedVector<std::unique_ptr<TfrtCpuBuffer>, 4> buffers;
    buffers.reserve(shapes.size());
    absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4> avs;
    avs.reserve(shapes.size());
    for (const Shape& shape : shapes) {
      absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4> local_avs;
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<TfrtCpuBuffer> buffer,
          AllocateDestinationBufferAndAvs(shape, &local_avs, device, client));
      CHECK_EQ(local_avs.size(), 1);
      avs.push_back(std::move(local_avs[0]));
      buffers.push_back(std::move(buffer));
    }
    absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> device_buffers;
    device_buffers.reserve(buffers.size());
    for (const std::unique_ptr<TfrtCpuBuffer>& buffer : buffers) {
      auto usage_event = tfrt::MakeAvailableAsyncValueRef<CpuEvent>();
      auto* device_buffer = buffer->AcquireUsage(std::move(usage_event));
      CHECK(device_buffer);
      device_buffers.push_back(device_buffer);
    }

    absl::InlinedVector<size_t, 4> buffer_sizes;
    buffer_sizes.reserve(buffers.size());
    for (const std::unique_ptr<TfrtCpuBuffer>& buffer : buffers) {
      TF_ASSIGN_OR_RETURN(const size_t buffer_size,
                          buffer->GetOnDeviceSizeInBytes());
      buffer_sizes.push_back(buffer_size);
    }

    return absl::WrapUnique(new TfrtCpuAsyncHostToDeviceTransferManager(
        std::move(avs), std::move(buffers), std::move(device_buffers),
        std::move(buffer_sizes), DefaultThreadPoolSize(), device));
  }

  ~TfrtCpuAsyncHostToDeviceTransferManager() override {
    // Wait for in-flight transfers to finish.
    absl::Condition transfers_finished(
        +[](int* t) { return *t == 0; }, &transfers_in_flight_);
    absl::MutexLock l(&mu_);
    mu_.Await(transfers_finished);
    for (auto& avref : avs_) {
      auto av = avref.CopyRef();
      if (av && av->IsUnavailable()) {
        av->SetError(absl::InternalError(
            "Async transfer object was deleted before transfers completed."));
      }
    }
  }

  size_t buffer_count() const override { return buffer_sizes_.size(); };

  size_t buffer_size(int buffer_index) const override {
    CHECK_GE(buffer_index, 0);
    CHECK_LT(buffer_index, buffer_sizes_.size());
    return buffer_sizes_[buffer_index];
  }

  PjRtDevice* device() const override { return device_; }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    absl::MutexLock l(&mu_);
    CHECK_GE(buffer_index, 0);
    CHECK_LT(buffer_index, buffers_.size());
    return std::move(buffers_[buffer_index]);
  }

  Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, literal.untyped_data(), 0,
                                      literal.size_bytes(), true,
                                      std::move(on_done));
  }

  Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(), 0, data.size(),
                                      true, std::move(on_done));
  }

  Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    absl::MutexLock l(&mu_);
    CHECK_GE(buffer_index, 0);
    CHECK_LT(buffer_index, buffers_.size());
    CHECK_LE(transfer_size + offset, buffer_sizes_[buffer_index]);
    ++transfers_in_flight_;
    EnqueueWork(
        thread_pool_.get(),
        [this, device_buffer = device_buffers_[buffer_index],
         av = avs_[buffer_index].CopyRef(), data, offset, transfer_size,
         is_last_transfer, on_done = std::move(on_done)]() mutable -> void {
          absl::MutexLock l(&mu_);
          const std::shared_ptr<MaybeOwningCpuMemory>& b =
              device_buffer->Buffers()[0];
          std::memcpy(reinterpret_cast<char*>(b->data()) + offset, data,
                      transfer_size);
          std::move(on_done)();
          if (is_last_transfer) {
            av->SetStateConcrete();
          }
          --transfers_in_flight_;
        });
    return tsl::OkStatus();
  }

  void SetBufferError(int buffer_index, Status error) override {
    absl::MutexLock l(&mu_);
    avs_[buffer_index]->SetError(ToAbslStatus(error));
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {
    LOG(WARNING) << "AddTransferMetadata not implemented for TfrtCpuClient";
  }

 private:
  TfrtCpuAsyncHostToDeviceTransferManager(
      absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4> avs,
      absl::InlinedVector<std::unique_ptr<TfrtCpuBuffer>, 4> buffers,
      absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> device_buffers,
      absl::InlinedVector<size_t, 4> buffer_sizes, size_t num_threads,
      TfrtCpuDevice* device)
      : transfers_in_flight_(0),
        avs_(std::move(avs)),
        buffers_(std::move(buffers)),
        device_buffers_(std::move(device_buffers)),
        buffer_sizes_(std::move(buffer_sizes)),
        thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
            tsl::Env::Default(),
            "XLATfrtCpuTfrtCpuAsyncHostToDeviceTransferManager", num_threads)),
        device_(device) {}

  mutable absl::Mutex mu_;
  // The number of transfers that are currently in flight.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_);
  // AsyncValues used to mark buffers as ready for consumption.
  absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4> avs_
      ABSL_GUARDED_BY(mu_);
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<TfrtCpuBuffer>, 4> buffers_
      ABSL_GUARDED_BY(mu_);
  // Device buffers which we use to get the underlying memory to populate.
  absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> device_buffers_
      ABSL_GUARDED_BY(mu_);
  // Cached versions of the sizes of all the buffers. Not modified after
  // creation, so not guarded by mu_.
  absl::InlinedVector<size_t, 4> buffer_sizes_;

  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  TfrtCpuDevice* device_;  // not owned.
};

}  // namespace

TfrtCpuDeviceDescription::TfrtCpuDeviceDescription(int id) : id_(id) {
  debug_string_ = absl::StrCat("TFRT_CPU_", id);
  to_string_ = absl::StrCat("CpuDevice(id=", id, ")");
}

absl::string_view TfrtCpuDeviceDescription::device_kind() const {
  return kCpuPlatformName;
}

absl::string_view TfrtCpuDeviceDescription::DebugString() const {
  return debug_string_;
}

absl::string_view TfrtCpuDeviceDescription::ToString() const {
  return to_string_;
}

TfrtCpuDevice::TfrtCpuDevice(int id, bool asynchronous)
    : description_(id),
      max_inflight_computations_semaphore_(/*capacity=*/asynchronous ? 32 : 1) {
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
  // Need at least CpuDeviceCount threads to launch one collective.
  size_t num_threads = std::max(DefaultThreadPoolSize(), cpu_device_count);

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                      GetTfrtCpuDevices(asynchronous, cpu_device_count));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtCpuClient>(
      /*process_index=*/0, std::move(devices), num_threads));
}

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous) {
  return GetTfrtCpuClient(asynchronous, CpuDeviceCount());
}

TfrtCpuClient::TfrtCpuClient(
    int process_index, std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
    size_t num_threads)
    : process_index_(process_index),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      pjrt_client_thread_pool_(new tsl::thread::ThreadPool(
          tsl::Env::Default(), "XLATfrtCpuClient", num_threads)),
      async_work_runner_(std::make_unique<ThreadPoolAsyncWorkRunner>(
          pjrt_client_thread_pool_.get())),
      eigen_intraop_pool_(new tsl::thread::ThreadPool(
          tsl::Env::Default(), "XLAEigen", DefaultThreadPoolSize())),
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

StatusOr<std::unique_ptr<HloCostAnalysis>> TfrtCpuClient::GetHloCostAnalysis()
    const {
  return std::make_unique<HloCostAnalysis>(cpu::CpuExecutable::ShapeSizeBytes);
}

StatusOr<std::optional<std::string>> TfrtCpuClient::ExecutableFingerprint(
    const PjRtLoadedExecutable& executable) const {
  return std::optional<std::string>();
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

StatusOr<std::string> TfrtCpuExecutable::SerializeExecutable() const {
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

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtCpuClient::DeserializeExecutable(absl::string_view serialized,
                                     std::optional<CompileOptions> options) {
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
      aot_result->LoadExecutable(&compiler, /*executor=*/nullptr));

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
  bool allow_sparse_shapes =
      hlo_module->config().debug_options().xla_cpu_use_xla_runtime();
  cpu::CpuCompiler compiler(allow_sparse_shapes);
  xla::Compiler::CompileOptions dummy;
  TF_ASSIGN_OR_RETURN(hlo_module,
                      compiler.RunHloPasses(std::move(hlo_module),
                                            /*stream_exec=*/nullptr, dummy));

  // Run backend.
  return compiler.RunBackend(std::move(hlo_module), /*stream_exec=*/nullptr,
                             dummy);
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtCpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  tsl::profiler::TraceMe traceme("TfrtCpuClient::Compile");
  auto input_options = options;
  ExecutableBuildOptions& build_options = options.executable_build_options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

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
      options.parameter_is_tupled_arguments, std::move(input_options),
      std::move(cpu_executable), result_slice.index(),
      std::move(result_buffer_indices),
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
  tsl::profiler::TraceMe traceme("TfrtCpuClient::CreateUninitializedBuffer");
  VLOG(1) << "TfrtCpuClient::CreateUninitializedBuffer: shape: "
          << shape.DebugString() << " device: " << device->DebugString();
  return AllocateDestinationBuffer(
      shape, /*definition_events=*/{},
      tensorflow::down_cast<TfrtCpuDevice*>(device), this);
}

StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtCpuClient::CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                                 PjRtDevice* device) {
  auto* tfrt_device = tensorflow::down_cast<TfrtCpuDevice*>(device);
  return TfrtCpuAsyncHostToDeviceTransferManager::Create(shapes, tfrt_device,
                                                         this);
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  tsl::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostBuffer");
  Shape shape = ShapeUtil::MakeShape(type, dims);
  VLOG(2) << "TfrtCpuClient::BufferFromHostBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      AbstractTfrtCpuBuffer::BufferFromHostBufferHelper(
          data, type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), shape, async_work_runner(),
          &transpose_mu_, &transpose_cache_));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tensorflow::down_cast<TfrtCpuDevice*>(device)));
}

StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtDevice* device) {
  tsl::profiler::TraceMe traceme("TfrtCpuClient::BufferFromHostLiteral");
  VLOG(1) << "TfrtCpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().DebugString()
          << " device: " << device->DebugString();
  const Shape& shape = literal.shape();

  absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4> avs;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfrtCpuBuffer> output_buffer,
      AllocateDestinationBufferAndAvs(
          shape, &avs, tensorflow::down_cast<TfrtCpuDevice*>(device), this));

  output_buffer->CopyFromLiteral(literal, shape, &avs, async_work_runner());

  return std::unique_ptr<PjRtBuffer>(std::move(output_buffer));
}

TfrtCpuBuffer::TfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
    TfrtCpuClient* client, TfrtCpuDevice* device)
    : AbstractTfrtCpuBuffer(std::move(on_device_shape),
                            std::move(tracked_device_buffer)),
      client_(client),
      device_(device) {}

static std::vector<tfrt::RCReference<tfrt::AsyncValue>> CopyAsyncValues(
    absl::Span<const tfrt::RCReference<tfrt::AsyncValue>> events) {
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev.CopyRef());
  }
  return avs;
}

PjRtFuture<Status> TfrtCpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  return ToLiteralHelper(literal, client()->async_work_runner());
}

// TODO(zhangqiaorjc): Consider disallowing multiple CPU devices and assign
// multiple pmap replicas to the same CPU device for multi-CPU pmap testing.
StatusOr<std::unique_ptr<PjRtBuffer>> TfrtCpuBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  tsl::profiler::TraceMe traceme("TfrtCpuBuffer::CopyToDevice");
  // TODO(zhangqiaorjc): Remove this restriction after removing the test that
  // explicitly asserts this.
  if (dst_device == device_) {
    return InvalidArgument(
        "CopyToDevice cannot accept the same source and destination devices");
  }

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    return CopyToDeviceAcrossClients(dst_device);
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      CopyToDeviceHelper(client()->async_work_runner()));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtCpuBuffer>(
      on_device_shape_, std::move(tracked_device_buffer), client(),
      tensorflow::down_cast<TfrtCpuDevice*>(dst_device)));
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
    absl::Span<std::pair<bool, TrackedTfrtCpuDeviceBuffer*> const> arguments) {
  if (allocation.is_entry_computation_parameter()) {
    auto [can_donate, arg] = arguments[allocation.parameter_number()];
    std::shared_ptr<MaybeOwningCpuMemory> out =
        arg->Buffer(allocation.param_shape_index());
    CHECK_EQ(allocation.size(), out->size())
        << "Size mismatch on param " << allocation.parameter_number()
        << " at shape index " << allocation.param_shape_index().ToString();

    // If we don't own the buffer, we can't overwrite it or donate it. For
    // example we might be pointing to a buffer owned by the client whose
    // lifetime will not extend past the lifetime of the donated input buffer.
    if ((!can_donate || !out->owns_data()) && !allocation.is_readonly()) {
      TF_ASSIGN_OR_RETURN(
          auto copy, MaybeOwningCpuMemory::AllocateShared(allocation.size()));
      std::memcpy(copy->data(), out->data(), allocation.size());
      return copy;
    }
    return out;
  } else if (allocation.is_constant()) {
    return std::make_shared<MaybeOwningCpuMemory>();
  } else if (allocation.is_thread_local()) {
    return std::make_shared<MaybeOwningCpuMemory>();
  }

  // Output and temporary buffer.
  TF_ASSIGN_OR_RETURN(auto out,
                      MaybeOwningCpuMemory::AllocateShared(allocation.size()));

  // Since the output buffer and all the temporary buffers were written into
  // by the JITed code, msan has no way of knowing their memory was
  // initialized. Mark them initialized so that msan doesn't flag loads from
  // these buffers.
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(out->data(), allocation.size());
  return out;
}

static StatusOr<std::vector<std::shared_ptr<MaybeOwningCpuMemory>>>
CreateBufferTable(
    const BufferAssignment& assignment,
    absl::Span<std::pair<bool, TrackedTfrtCpuDeviceBuffer*> const> arguments) {
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
    absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> buffer_table) {
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> output_buffers;
  output_buffers.reserve(buffer_indices.size());
  for (int i = 0; i < buffer_indices.size(); ++i) {
    output_buffers.push_back(buffer_table[buffer_indices[i]]);
  }
  return output_buffers;
}

Status TfrtCpuExecutable::CheckBufferCompatibilities(
    absl::Span<std::pair<bool, TrackedTfrtCpuDeviceBuffer*> const>
        input_buffers) const {
  if (input_buffers.size() != input_buffer_sizes_in_bytes_.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes_.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i].second;
    if (input_buffer_sizes_in_bytes_[i] != buffer->Buffers()[0]->size()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with "
          "incompatible size %lld",
          i, input_buffer_sizes_in_bytes_[i], buffer->Buffers()[0]->size());
    }
  }
  return OkStatus();
}

// Create a descriptor table for XLA Runtime from a buffer table.
static std::vector<xla::cpu::BufferDesc> MakeXLARuntimeDescriptorTable(
    absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> buffer_table) {
  std::vector<xla::cpu::BufferDesc> descriptor_table;
  descriptor_table.reserve(buffer_table.size());
  for (const auto& buf : buffer_table) {
    descriptor_table.emplace_back(buf->data(), buf->size());
  }
  return descriptor_table;
}

StatusOr<PjRtLoadedExecutable::Result> TfrtCpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options,
    tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event,
    bool fill_future, TfrtCpuDevice* device) {
  tsl::profiler::TraceMe traceme("TfrtCpuExecutable::ExecuteHelper");

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

  absl::InlinedVector<std::pair<bool, TrackedTfrtCpuDeviceBuffer*>, 4>
      tracked_buffers;
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

    TrackedTfrtCpuDeviceBuffer* tracked_buffer;
    auto get_buffer = [&](int i) -> Status {
      bool must_donate = donate_it != parameters_that_must_be_donated_.end() &&
                         *donate_it == i;
      if (must_donate) {
        ++donate_it;
        StatusOr<TfrtCpuBuffer::DonationTransaction> donation_transaction =
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
          for (const auto& ev :
               donation_transaction->device_buffer()->UsageEvents()) {
            if (!ev.IsAvailable()) {
              input_deps.push_back(ev.CopyRCRef());
            }
          }
          tracked_buffer = donation_transaction->device_buffer();
          tracked_buffers.emplace_back(/*can_donate=*/true, tracked_buffer);
          donation_transactions.push_back(std::move(*donation_transaction));
          return OkStatus();
        }
      }
      tracked_buffer = tfrt_buffer->AcquireUsage(execute_event);
      if (!tracked_buffer)
        return InvalidArgument(
            "Invalid buffer passed: buffer has been deleted or donated.");
      tracked_buffers.emplace_back(/*can_donate=*/false, tracked_buffer);
      return OkStatus();
    };
    TF_RETURN_IF_ERROR(get_buffer(i));

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
      auto span = tracked_buffer.second->Buffers();
      leaf_buffers.insert(leaf_buffers.end(), span.begin(), span.end());
    }

    // Tuplize into a single input.
    tracked_buffers.clear();
    tuplized_arg = std::make_unique<TrackedTfrtCpuDeviceBuffer>(
        /*is_tuple=*/true, std::move(leaf_buffers),
        /*definition_event=*/tfrt::MakeAvailableAsyncValueRef<CpuEvent>());
    tracked_buffers.emplace_back(false, tuplized_arg.get());
  }

  auto* cpu_executable =
      tensorflow::down_cast<cpu::CpuExecutable*>(cpu_executable_.get());
  TF_ASSIGN_OR_RETURN(
      std::vector<std::shared_ptr<MaybeOwningCpuMemory>> buffer_table,
      CreateBufferTable(cpu_executable->buffer_assignment(), tracked_buffers));
  auto result_buffers =
      CreateResultShapedBuffer(result_buffer_indices_, buffer_table);

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
    tsl::port::ScopedFlushDenormal flush;
    tsl::port::ScopedSetRound round(FE_TONEAREST);

    XlaCustomCallStatus status;

    // Call generated function.
    if (cpu_executable->IsXlaRuntime()) {
      Status status = cpu_executable->ExecuteXlaRuntime(
          MakeXLARuntimeDescriptorTable(buffer_table), &run_options);
      if (!status.ok()) return status;
    } else {
      cpu_executable->compute_function()(result_buffer, &run_options, nullptr,
                                         buffer_pointers.data(), &status,
                                         nullptr);
    }

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
        client()->pjrt_client_thread_pool(), input_deps,
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
                  "Error dispatching computation: %s", error->message()));
              return;
            }
          }

          // Set denormal and rounding behavior to match the default TF
          // ThreadPool behavior.
          tsl::port::ScopedFlushDenormal flush;
          tsl::port::ScopedSetRound round(FE_TONEAREST);

          // Call generated function.
          std::optional<absl::string_view> error_message;
          if (cpu_executable->IsXlaRuntime()) {
            Status s = cpu_executable->ExecuteXlaRuntime(
                MakeXLARuntimeDescriptorTable(buffer_table), &run_options);
            if (!s.ok()) {
              // TODO(kramerb): Propagate custom call error messages.
              error_message = "XLA Runtime execution failed";
            }
          } else {
            XlaCustomCallStatus status;
            cpu_executable->compute_function()(result_buffer, &run_options,
                                               nullptr, buffer_pointers.data(),
                                               &status, nullptr);
            error_message = xla::CustomCallStatusGetMessage(&status);
          }

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
            s = InternalError("Compute error: %s", error->message());
          }
          done_event.emplace(std::move(s));
        });
    future = PjRtFuture<Status>(std::move(done_event));
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

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtCpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  tsl::profiler::TraceMe traceme("TfrtCpuExecutable::Execute");
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }

  RunId run_id;
  tsl::profiler::TraceMeProducer activity("TfrtCpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
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

    // Dump once before running, in case there's a crash.
    MaybeDumpHloSnapshot(cpu_executable_->module(), run_id, argument_handles[0],
                         {});
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

    MaybeDumpHloSnapshot(cpu_executable_->module(), run_id, argument_handles[0],
                         wrapped_results[0]);
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

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtCpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  tsl::profiler::TraceMe traceme("TfrtCpuExecutable::ExecuteSharded");
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
  tsl::profiler::TraceMe traceme("TfrtCpuExecutable::ExecutePortable");
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
