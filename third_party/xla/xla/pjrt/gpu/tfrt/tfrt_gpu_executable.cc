/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_executable.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/layout.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/utils.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/maybe_owning_device_address.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#if defined(PLATFORM_WINDOWS)
// Required to build successfully with Mingw
#undef CreateEvent
#endif

namespace xla {

class TfrtGpuCopyToDeviceStream : public CopyToDeviceStream {
 public:
  TfrtGpuCopyToDeviceStream(int64_t channel_id, se::Stream* stream,
                            se::DeviceAddressBase dst,
                            tsl::AsyncValueRef<std::unique_ptr<se::Event>> done)
      : CopyToDeviceStream(dst.size(), /*granule_bytes=*/1),
        channel_id_(channel_id),
        stream_(stream),
        dst_(dst),
        done_(std::move(done)) {}

  Future<> AddChunk(PjRtChunk chunk) final {
    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode("TfrtGpuCopyToDeviceStream::AddChunk",
                                          {{"channel_id", channel_id_}});
    });

    absl::ReleasableMutexLock lock(mu_);

    VLOG(4) << "Add chunk to a H2D channel #" << channel_id_ << ": "
            << "size=" << chunk.size() << ", "
            << "current_bytes=" << current_bytes_ << ", "
            << "total_bytes=" << total_bytes_;

    if (chunk.size() % granule_size_in_bytes() != 0) {
      done_.SetError(absl::InvalidArgumentError(absl::StrFormat(
          "Chunk size (%d) was not a multiple of the granule size (%d)",
          chunk.size(), granule_size_in_bytes())));
      return Future<>(done_.GetError());
    }

    if (current_bytes_ + chunk.size() > total_bytes_) {
      done_.SetError(absl::InvalidArgumentError(
          absl::StrFormat("Adding chunk of size %d would overflow buffer of "
                          "size %d (%d already transferred)",
                          chunk.size(), total_bytes_, current_bytes_)));
      return Future<>(done_.GetError());
    }

    se::DeviceAddressBase dst(
        reinterpret_cast<std::byte*>(dst_.opaque()) + current_bytes_,
        dst_.size() - current_bytes_);

    current_bytes_ += chunk.size();
    bool complete = IsCompleteLocked();
    lock.Release();

    VLOG(3) << "H2D copy: " << chunk.data() << " -> " << dst.opaque() << " ("
            << chunk.size() << " bytes)";
    auto copied = stream_->Memcpy(&dst, chunk.data(), chunk.size());
    if (!copied.ok()) {
      done_.SetError(copied);
      return Future<>(done_.GetError());
    }

    // Delete chunk once the memcpy operation completes.
    auto deleted = stream_->DoHostCallback(
        [chunk = std::move(chunk), buffer_opaque = dst.opaque()]() {
          VLOG(3) << "H2D copy done. " << buffer_opaque;
        });
    if (!deleted.ok()) {
      done_.SetError(deleted);
      return Future<>(done_.GetError());
    }

    // Record done event once processed the last chunk. It is the caller
    // responsibility to synchronize with this event before submitting any new
    // computations to the stream.
    if (complete) {
      auto recorded = stream_->RecordEvent(done_.get().get());
      if (!recorded.ok()) {
        done_.SetError(recorded);
        return Future<>(done_.GetError());
      }
      done_.SetStateConcrete();
    }

    return Future<>(absl::OkStatus());
  }

 private:
  int64_t channel_id_;
  se::Stream* stream_;
  se::DeviceAddressBase dst_;

  // Async value will become available after we'll submit the last memcpy
  // operation, and the event will be recorded on the stream.
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_;
};

absl::StatusOr<std::string> TfrtGpuExecutable::SerializeExecutable() const {
  if (executables_.size() != 1) {
    // TODO(b/382117736): Change SerializeExecutable interface to support
    // multiple partitions.
    return absl::FailedPreconditionError(
        "SerializeExecutable with >1 partitions not yet supported");
  }
  Executable* built_executable = executables_[0]->executable();
  Compiler* compiler = client_->xla_client()->backend().compiler();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<CompiledModule> aot_result,
                      compiler->Export(built_executable));
  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  if (serialized.empty()) {
    return Internal(
        "TfrtGpuExecutable::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  *proto.mutable_pjrt_client_name() = kPjRtClientName;
  return proto.SerializeAsString();
}

PjRtClient* TfrtGpuExecutable::client() const { return (PjRtClient*)client_; }

TfrtGpuExecutable::TfrtGpuExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    CompileOptions compile_options,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtGpuClient* client)
    : client_(client),
      device_assignment_(std::move(device_assignment)),
      compile_options_(std::move(compile_options)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  TransferManager* transfer_manager =
      client_->xla_client()->backend().transfer_manager();
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
  executables_.reserve(executables.size());
  for (auto& executable : executables) {
    const auto& computation_layout =
        executable->executable()->module().entry_computation_layout();
    std::vector<Shape> parameter_shapes;
    parameter_shapes.reserve(computation_layout.parameter_count());
    for (int i = 0; i < computation_layout.parameter_count(); ++i) {
      parameter_shapes.push_back(transfer_manager->HostShapeToDeviceShape(
          computation_layout.parameter_shape(i)));
    }
    on_device_executable_parameter_shapes_.push_back(
        std::make_shared<std::vector<Shape>>(std::move(parameter_shapes)));

    auto input_buffer_sizes_in_bytes = std::make_shared<std::vector<int64_t>>();

    // Assume compiled program expects either many non-tupled arguments or a
    // singled tupled argument, or no arguments. Nested tuple is not yet
    // supported.
    if (computation_layout.parameter_count() == 0) {
      // No arguments. Do nothing.
    } else if (computation_layout.parameter_count() == 1 &&
               computation_layout.parameter_shape(0).IsTuple()) {
      const std::vector<Shape>& tuple_shapes =
          computation_layout.parameter_shape(0).tuple_shapes();
      input_buffer_sizes_in_bytes->reserve(tuple_shapes.size());
      for (const Shape& shape : tuple_shapes) {
        input_buffer_sizes_in_bytes->push_back(ShapeUtil::ByteSizeOf(shape));
      }
    } else {
      const std::vector<ShapeLayout>& parameter_layouts =
          computation_layout.parameter_layouts();
      input_buffer_sizes_in_bytes->reserve(parameter_layouts.size());
      for (const ShapeLayout& layout : parameter_layouts) {
        input_buffer_sizes_in_bytes->push_back(
            ShapeUtil::ByteSizeOf(layout.shape()));
      }
    }
    input_buffer_sizes_in_bytes_.push_back(
        std::move(input_buffer_sizes_in_bytes));

    fingerprint = tsl::FingerprintCat128(
        fingerprint,
        tsl::Fingerprint128(executable->executable()->module().ToString(
            HloPrintOptions::ModuleFingerprint())));
    executables_.emplace_back(std::move(executable));
  }
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // This must go after `executables_` is initialized.
    VLOG(4) << "TfrtGpuExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    VLOG(4) << "TfrtGpuExecutable device_assignment:\n"
            << device_assignment_->ToString();

    if ((device_assignment_->replica_count() > 1 ||
         device_assignment_->computation_count() > 1) &&
        IsAllZeros(*device_assignment_)) {
      // This code path should only be triggered when we intentionally compile
      // an HLO without having enough devices to actually run it. See the
      // "--run=false" option in
      // tensorflow/compiler/xla/tools/multihost_hlo_runner/hlo_runner_main.cc.
      // That will help us debug the XLA compiler locally.
      LOG(INFO) << "A workaround is in effect to allow compiling multi-device "
                   "HLOs on machines with fewer devices. Don't run this "
                   "executable.";
    } else {
      CHECK_LE(addressable_devices_.size(), client_->addressable_device_count())
          << "Inconsistent local device count.";
    }

    num_partitions = device_assignment_->computation_count();
  }

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }
}

absl::StatusOr<PjRtLoadedExecutable::Result> TfrtGpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const ExecuteOptions& options, bool fill_future,
    TfrtGpuDevice* device) const {
  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
    VLOG(3) << "device_id: " << device_id;
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(PjRtGlobalDeviceId(device_id)));
    device = tsl::down_cast<TfrtGpuDevice*>(pjrt_device);
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

  tsl::profiler::TraceMeProducer activity(
      [&] {
        return tsl::profiler::TraceMeEncode("TfrtGpuExecutable::ExecuteHelper",
                                            {{"launch_id", options.launch_id},
                                             {"device_id", device->id()},
                                             {"name", name()}});
      },
      tsl::profiler::ContextType::kPjRt, options.launch_id);

  VLOG(1) << "ExecuteHelper " << name() << ": " << options.launch_id
          << "; replica: " << replica << "; partition: " << partition
          << "; mapped to device ordinal for execution: " << device->id();

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  std::unique_ptr<Semaphore::ScopedReservation> compute_reservation;
  {
    tsl::profiler::TraceMe traceme_compute_reservation(
        "TfrtGpuExecutable::ExecuteHelper::acquire_semaphore");

    VLOG(1) << "Trying to acquire semaphore for " << name() << " on device "
            << device->DebugString();
    compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
        device->max_inflight_computations_semaphore().ScopedAcquire(1));
    VLOG(1) << "Acquired semaphore for " << name() << " on device "
            << device->DebugString();
  }

  // Handle inputs.
  // SPMD sharding produces a single executable for multiple partitions.
  int executable_idx = executables_.size() > 1 ? partition : 0;

  TF_ASSIGN_OR_RETURN(std::vector<Shape> output_shapes, GetOutputShapes());
  const Shape& result_shape = output_shapes[executable_idx];

  // `scheduled_event` indicates whether gpu computation is dispatched to the
  // stream and whether there was an error.
  auto scheduled_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  // `complete_event` indicates whether gpu computation is complete and whether
  // there was an error.
  tsl::AsyncValueRef<GpuEvent> complete_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  absl::InlinedVector<TfrtGpuBuffer::DonationTransaction, 4>
      donation_transactions;

  absl::InlinedVector<TrackedGpuDeviceBuffer*, 4> tracked_buffers;
  absl::InlinedVector<bool, 4> buffer_is_donated;
  tracked_buffers.reserve(argument_handles.size());
  buffer_is_donated.reserve(argument_handles.size());
  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `complete_event` dominates all inputs'
  // events, and thus output buffer only need to contain `complete_event` as
  // the single definition event.
  std::vector<tsl::RCReference<tsl::AsyncValue>> prepare_input_deps;
  std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
  std::vector<tsl::RCReference<tsl::AsyncValue>> ready_deps;
  input_deps.reserve(argument_handles.size() + 1);

  absl::Span<int const> donated_params =
      parameters_that_must_be_donated_[executable_idx];
  auto donate_it = donated_params.begin();

  absl::flat_hash_map<const void*, std::pair<bool, int>> donation_clashes;
  donation_clashes.reserve(argument_handles.size());
  for (int i = 0; i < argument_handles.size(); ++i) {
    PjRtBuffer* handle = argument_handles[i];
    auto* tfrt_buffer = tsl::down_cast<TfrtGpuBuffer*>(handle);

    if (tfrt_buffer->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, tfrt_buffer->device()->DebugString(),
          device->DebugString());
    }
    bool donation_denied_at_runtime =
        options.non_donatable_input_indices.contains(i);
    bool must_donate = donate_it != donated_params.end() && *donate_it == i &&
                       !donation_denied_at_runtime;

    // Prepare the tracked buffer for the input.
    TrackedGpuDeviceBuffer* tracked_buffer = nullptr;
    if (must_donate) {
      VLOG(3) << "Buffer for argument_handles[" << i << "] is donated";

      ++donate_it;
      TF_RETURN_IF_ERROR(TestBufferDonationClashes(
          handle, donation_clashes, must_donate, i, replica, partition));
      TF_ASSIGN_OR_RETURN(auto donation_transaction,
                          tfrt_buffer->AcquireDonation());

      // After acquiring the buffer for donation, we retrieve the dependent
      // usage events. Note that we don't need any locking here as
      // AcquireDonation() is supposed to synchronize with other usages.
      input_deps.push_back(
          donation_transaction.device_buffer()->AfterAllUsageEvents());
      tracked_buffer = donation_transaction.device_buffer();
      donation_transactions.push_back(std::move(donation_transaction));
      buffer_is_donated.push_back(true);
    } else {
      tracked_buffer = tfrt_buffer->AcquireUsage(complete_event);
      if (!tracked_buffer) {
        return InvalidArgument(
            "Invalid buffer passed: buffer has been deleted or donated.");
      }
      buffer_is_donated.push_back(false);
    }

    // By now, the tracked buffer is guaranteed to be valid.
    tracked_buffers.push_back(tracked_buffer);
    prepare_input_deps.push_back(tracked_buffer->buffer().CopyRCRef());

    VLOG(3) << "argument_handles[" << i << "]: addr = "
            << (tracked_buffer->buffer().IsAvailable()
                    ? tracked_buffer->buffer()->buffer().opaque()
                    : "NotReady")
            << ", logical shape = "
            << tfrt_buffer->logical_on_device_shape()->ToString();

    // Definition events are never modified after buffer construction. If they
    // are available and have no error, they can be skipped in input deps.
    // In contrast, already known errors in the input are taken as deps so
    // that they can poison output buffers.
    const auto& definition_event = tracked_buffer->definition_event();
    if (!definition_event.IsAvailable() || definition_event.IsError()) {
      VLOG(3) << "definition_event is not available: AsyncValue pointer: "
              << definition_event.GetAsyncValue();
      input_deps.push_back(definition_event.CopyRCRef());
    }
    ready_deps.push_back(tracked_buffer->ready_event().CopyRCRef());
  }

  {
    // Schedule only one collective at a time.
    tsl::AsyncValueRef<GpuEvent> ordering_event =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
    tsl::AsyncValueRef<GpuEvent> last_collective_launch_event =
        device->SetLastCollectiveLaunchEvent(scheduled_event);
    // We don't use last_collective_launch_event directly because we don't
    // want the previous failure to be propagated to the current execution.
    last_collective_launch_event.AndThen(
        [event = ordering_event.CopyRef()]() { event.SetStateConcrete(); });
    input_deps.push_back(std::move(ordering_event));
  }

  // Call `CreateCudaEvent` on a thread pool to avoid calling CUDA API inline.
  // See the comments in `TfrtGpuStreamAccessorGuard` for more information about
  // why this is necessary.
  TF_ASSIGN_OR_RETURN(
      auto output_cuda_execute_event,
      RunOnAsyncWorkRunner(client_->non_blocking_thread_pool(),
                           [&]() { return CreateCudaEvent(device); }));

  std::vector<tsl::AsyncValueRef<GpuDeviceMemory>> output_buffers;
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  auto gpu_executable = executables_[executable_idx];
  bool result_is_tuple = result_shape.IsTuple();
  if (result_shape.IsTuple()) {
    output_buffers.reserve(result_shape.tuple_shapes().size());
    outputs.reserve(output_buffers.size());
    for (int i = 0; i < result_shape.tuple_shapes().size(); ++i) {
      output_buffers.push_back(
          tsl::MakeUnconstructedAsyncValueRef<GpuDeviceMemory>());
      // Program execution writes to output buffers so it's a definition
      // event.
      auto leaf_tracked_device_buffer =
          std::make_unique<TrackedGpuDeviceBuffer>(
              output_buffers.back().CopyRef(), scheduled_event.CopyRef(),
              complete_event.CopyRef(), nullptr, output_cuda_execute_event);
      VLOG(4) << "created leaf_tracked_device_buffer: "
              << leaf_tracked_device_buffer.get();

      const Shape& shape = result_shape.tuple_shapes(i);
      PjRtMemorySpace* memory_space =
          device->default_memory_space().value_or(nullptr);
      if (shape.has_layout() &&
          shape.layout().memory_space() == Layout::kHostMemorySpace) {
        TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind_id(
                                              PinnedHostMemorySpace::kKindId));
      }

      auto output = std::make_unique<TfrtGpuBuffer>(
          result_shape.tuple_shapes(i), std::move(leaf_tracked_device_buffer),
          client_, device, memory_space);
      outputs.push_back(std::move(output));
    }
  } else {
    output_buffers.push_back(
        tsl::MakeUnconstructedAsyncValueRef<GpuDeviceMemory>());
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
        output_buffers.back().CopyRef(),
        /*definition_event=*/scheduled_event.CopyRef(),
        complete_event.CopyRef(), nullptr, output_cuda_execute_event);
    VLOG(4) << "created tracked_device_buffer: " << tracked_device_buffer.get();

    const Shape& shape = result_shape;
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    if (shape.has_layout() &&
        shape.layout().memory_space() == Layout::kHostMemorySpace) {
      TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind_id(
                                            PinnedHostMemorySpace::kKindId));
    }

    auto tfrt_output_buffer = std::make_unique<TfrtGpuBuffer>(
        result_shape, std::move(tracked_device_buffer), client_, device,
        memory_space);
    outputs.push_back(std::move(tfrt_output_buffer));
  }

  auto ffi_context =
      options.context != nullptr ? &options.context->ffi_context() : nullptr;

  // Create a PjRt<->StreamExecutor adaptors to send/recv device memory as
  // PjRt chunks via the user-provided callbacks.
  SendDeviceMemoryFunction send_device_memory =
      ConvertSendCallbacksToSendFunction(replica, options,
                                         client_->non_blocking_thread_pool());
  RecvDeviceMemoryFunction recv_device_memory =
      ConvertRecvCallbacksToRecvFunction(replica, options);

  auto execute_fn = [replica, partition, device, launch_id(options.launch_id),
                     output_buffers(output_buffers),
                     complete_event(complete_event.CopyRef()),
                     scheduled_event(scheduled_event.CopyRef()),
                     result_is_tuple(result_is_tuple),
                     donation_transactions(std::move(donation_transactions)),
                     parameter_shapes(on_device_executable_parameter_shapes_
                                          [executable_idx]),
                     gpu_executable(std::move(gpu_executable)),
                     device_assignment(device_assignment),
                     executable_name(name()), ffi_context(ffi_context),
                     inputs_avs(CopyAsyncValues(input_deps)),
                     ready_deps(std::move(ready_deps)),
                     execution_profile(options.execution_profile),
                     send_device_memory(std::move(send_device_memory)),
                     recv_device_memory(std::move(recv_device_memory)),
                     output_cuda_execute_event(
                         std::move(output_cuda_execute_event)),
                     compute_reservation(std::move(compute_reservation)),
                     client = client_, task_incarnations = options.incarnations,
                     time_measurement_key = xla::GetDeviceTimeMeasurementKey()](
                        std::vector<ExecutionInput> execution_inputs) mutable {
    VLOG(1) << "execute_fn for " << executable_name
            << ", launch_id: " << launch_id << ", replica: " << replica
            << ", device: " << device->DebugString();

    tsl::profiler::TraceMeConsumer producer(
        [&] {
          return tsl::profiler::TraceMeEncode("execute_fn",
                                              {
                                                  {"launch_id", launch_id},
                                                  {"device_id", device->id()},
                                              });
        },
        tsl::profiler::ContextType::kPjRt, launch_id);

    auto set_error = [&](absl::Status status) {
      for (auto& output_buffer : output_buffers) {
        output_buffer.SetError(status);
      }
      complete_event.SetError(status);
      scheduled_event.SetError(status);
    };

    for (const auto& av : inputs_avs) {
      if (auto* error = av->GetErrorIfPresent()) {
        set_error(*error);
        return;
      }
    }

    // Set the incarnations in gpu_run_options.
    gpu::GpuExecutableRunOptions* gpu_run_options =
        CHECK_NOTNULL(client->gpu_run_options());
    if (!task_incarnations.empty()) {
      gpu_run_options->set_incarnations(
          GetLatestIncarnations(client->devices(), task_incarnations));
    }

    auto stream = device->stream();
    ExecutableRunOptions run_options;
    run_options.set_stream(stream);
    run_options.set_host_to_device_stream(stream);
    run_options.set_device_to_host_stream(stream);
    run_options.set_allocator(client->allocator());
    run_options.set_device_assignment(device_assignment.get());
    run_options.set_run_id(RunId(launch_id));
    run_options.set_rng_seed(device->GetNewPrngSeed());
    run_options.set_gpu_executable_run_options(gpu_run_options);
    run_options.set_launch_id(launch_id);
    run_options.set_local_device_count(client->device_count());
    run_options.set_device_ordinal(device->local_device_id().value());
    run_options.set_physical_device_ordinal(
        device->local_hardware_id().value());
    run_options.set_ffi_execution_context(ffi_context);
    run_options.set_intra_op_thread_pool(
        client->xla_client()->backend().eigen_intra_op_thread_pool_device());
    run_options.set_send_device_memory_function(&send_device_memory);
    run_options.set_recv_device_memory_function(&recv_device_memory);
    run_options.set_execution_profile(execution_profile);
    std::vector<std::unique_ptr<CliqueKey>> clique_keys;
    run_options.set_clique_keys(&clique_keys);

    // TODO(phawkins): *technically* this should probably happen after
    // calling RunAsync(). But that causes a large performance problem: it
    // prevents the main thread from freeing the buffer objects.
    for (auto& donation_transaction : donation_transactions) {
      VLOG(3) << "Committing donation transaction: "
              << donation_transaction.device_buffer();
      std::move(donation_transaction).Commit();
    }

    ////////////////////////////////////////////////////////////////////////
    // Record the start time of the execution by placing a callback on the
    // stream directly before the execution. If this callback is added,
    // another callback will be added directly after the execution to record
    // the elapsed device time.
    ////////////////////////////////////////////////////////////////////////
    auto start_time = std::make_shared<absl::Time>();
    if (time_measurement_key.has_value()) {
      absl::Status host_callback_status = stream->DoHostCallback(
          [start_time]() mutable { *start_time = absl::Now(); });

      if (!host_callback_status.ok()) {
        LOG(WARNING) << "Failed to do host callback for to register device "
                        "start time";
      }
    }

    ////////////////////////////////////////////////////////////////////////
    // Start calling RunAsync for the executable.
    ////////////////////////////////////////////////////////////////////////
    VLOG(1) << "Start calling RunAsync for " << executable_name
            << ", device=" << device->DebugString()
            << ", launch_id=" << launch_id << ", replica=" << replica
            << ", partition=" << partition;

    if (VLOG_IS_ON(2)) {
      absl::Status host_callback_status =
          stream->DoHostCallback([executable_name, launch_id, device]() {
            VLOG(1) << "Start device execution for " << executable_name
                    << ", launch_id: " << launch_id
                    << ", device: " << device->DebugString();
          });
      if (!host_callback_status.ok()) {
        LOG(WARNING)
            << "Failed to do host callback for start device execution for "
            << executable_name << ", status = " << host_callback_status;
      }
    }

    absl::StatusOr<ExecutionOutput> result_buffer_or_status =
        gpu_executable->RunAsync(std::move(execution_inputs), run_options);

    if (VLOG_IS_ON(2)) {
      absl::Status host_callback_status =
          stream->DoHostCallback([executable_name, launch_id, device]() {
            VLOG(1) << "Finish device execution for " << executable_name
                    << ", launch_id: " << launch_id
                    << ", device: " << device->DebugString();
          });
      if (!host_callback_status.ok()) {
        LOG(WARNING)
            << "Failed to do host callback for finish device execution for "
            << executable_name << ", status = " << host_callback_status;
      }
    }

    VLOG(1) << "Finish calling RunAsync for " << executable_name
            << ", device=" << device->DebugString()
            << ", launch_id=" << launch_id << ", replica=" << replica
            << ", partition=" << partition
            << ", completed, ok=" << result_buffer_or_status.ok();

    if (!result_buffer_or_status.ok()) {
      LOG(ERROR) << "Calling RunAsync failed for executable " << executable_name
                 << " on device " << device->DebugString()
                 << ", status = " << result_buffer_or_status.status();
      set_error(result_buffer_or_status.status());
      return;
    }

    auto record_event_status =
        stream->RecordEvent(output_cuda_execute_event.get());
    if (!record_event_status.ok()) {
      LOG(ERROR) << "Failed to record cuda event: " << record_event_status;
      scheduled_event.SetError(record_event_status);
      complete_event.SetError(record_event_status);
      return;
    }

    ////////////////////////////////////////////////////////////////////////
    // Record the end time of the execution by placing a callback on the
    // stream directly after the execution. If this callback is added,
    // another callback will be added directly before the execution to
    // record the elapsed device time.
    ////////////////////////////////////////////////////////////////////////
    if (time_measurement_key.has_value()) {
      absl::Status host_callback_status = stream->DoHostCallback(
          [executable_name, time_measurement_key, start_time]() mutable {
            auto elapsed = absl::Now() - *start_time;
            VLOG(1) << "Device execution time for " << executable_name << " is "
                    << elapsed;

            xla::RecordDeviceTimeMeasurement(
                *time_measurement_key, elapsed,
                DeviceTimeMeasurement::DeviceType::kGpu);
          });
      if (!host_callback_status.ok()) {
        LOG(WARNING) << "Failed to do host callback for to register device "
                        "time measurement";
      }
    }

    ExecutionOutput& execution_output = result_buffer_or_status.value();
    ScopedShapedBuffer output = execution_output.ConsumeResult();
    if (result_is_tuple) {
      for (int i = 0; i < output_buffers.size(); ++i) {
        ScopedShapedBuffer tuple_buffer = output.TakeSubTree({i});
        stream_executor::DeviceAddressBase* elem =
            tuple_buffer.buffers().mutable_element({});
        VLOG(3) << "untuple: output_buffers[" << i
                << "].emplace: " << elem->opaque();
        output_buffers[i].emplace(stream_executor::ScopedDeviceAddress<uint8_t>(
            *elem, device->local_device_id().value(), client->allocator()));
        *elem = se::DeviceAddressBase();
      }
    } else {
      CHECK_EQ(output_buffers.size(), 1);
      auto* elem = output.buffers().mutable_element({});
      VLOG(3) << "output_buffers[0].emplace: " << elem->opaque();
      output_buffers.front().emplace(
          stream_executor::ScopedDeviceAddress<uint8_t>(
              *elem, device->local_device_id().value(), client->allocator()));
      *elem = se::DeviceAddressBase();
    }

    // Set the scheduled event to concrete to indicate that the scheduling
    // has completed, so that the next execute_fn can start.
    scheduled_event.SetStateConcrete();

    absl::Status status = BlockHostUntilDoneWithHostCallback(stream);
    VLOG(1) << "execute_fn for " << executable_name
            << ", launch_id: " << launch_id << ", replica=" << replica
            << ", partition=" << partition
            << ", device: " << device->DebugString() << " is done with status "
            << status;

    if (!status.ok()) {
      LOG(ERROR) << "BlockHostUntilDoneWithHostCallback failed for executable "
                 << executable_name << " on device " << device->DebugString()
                 << ", status = " << status;
      complete_event.SetError(status);
      return;
    }

    // Propagate errors (if any) from dependencies.
    absl::Status ready_deps_status;
    for (const tsl::RCReference<tsl::AsyncValue>& ready : ready_deps) {
      tsl::BlockUntilReady(ready.get());
      if (!ready->IsError()) {
        continue;
      }
      absl::Status err = ready->GetError();
      LOG(ERROR) << "Computation has failed dependency: " << err;
      if (ready_deps_status.ok()) {
        ready_deps_status = err;
      } else {
        ready_deps_status = absl::Status(
            err.code(),
            absl::StrCat(ready_deps_status.message(), "; ", err.message()));
      }
    }
    if (!ready_deps_status.ok()) {
      complete_event.SetError(ready_deps_status);
      return;
    }

    // If any collective is stale, then the collective may have aborted.
    // Note that NCCL doesn't provide a way to *know* if the collective was
    // aborted, but we conservatively assume it was.
    for (const std::unique_ptr<CliqueKey>& clique_key : clique_keys) {
      gpu::GpuCliqueKey* gpu_clique_key = CHECK_NOTNULL(
          tensorflow::down_cast<gpu::GpuCliqueKey*>(clique_key.get()));
      if (absl::Status s = CheckCliqueIsntStale(*gpu_clique_key); !s.ok()) {
        VLOG(1) << "GPU clique key " << gpu_clique_key->ToString()
                << " is stale";
        complete_event.SetError(s);
        return;
      }
    }

    complete_event.SetStateConcrete();
  };

  auto prepare_inputs =
      [replica, client = client_, launch_id(options.launch_id),
       executable_name(name()), device,
       tracked_buffers(std::move(tracked_buffers)),
       buffer_is_donated(std::move(buffer_is_donated)),
       prepare_inputs_avs(CopyAsyncValues(prepare_input_deps)),
       complete_event(complete_event.CopyRef()),
       scheduled_event(scheduled_event.CopyRef()),
       output_buffers(std::move(output_buffers)),
       execute_fn(std::move(execute_fn)), input_deps(std::move(input_deps)),
       parameter_shapes(on_device_executable_parameter_shapes_[executable_idx]),
       parameter_is_tupled_arguments(parameter_is_tupled_arguments_),
       input_buffer_sizes_in_bytes(
           input_buffer_sizes_in_bytes_[executable_idx])]() mutable {
        tsl::profiler::TraceMeConsumer activity(
            [&] {
              return tsl::profiler::TraceMeEncode(
                  "prepare_inputs", {
                                        {"launch_id", launch_id},
                                        {"device_id", device->id()},
                                    });
            },
            tsl::profiler::ContextType::kPjRt, launch_id);

        auto set_error = [&](absl::Status status) {
          complete_event.SetError(status);
          scheduled_event.SetError(status);
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
        };

        for (const auto& av : prepare_inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            set_error(*error);
            return;
          }
        }

        VLOG(3) << "prepare_inputs for " << executable_name
                << ", launch_id: " << launch_id << ", replica: " << replica
                << ", device: " << device->DebugString();
        DCHECK_EQ(tracked_buffers.size(), buffer_is_donated.size());

        absl::Status status = CheckBufferCompatibilities(
            *input_buffer_sizes_in_bytes, tracked_buffers);
        if (!status.ok()) {
          set_error(status);
          return;
        }

        std::vector<ExecutionInput> inputs;
        if (parameter_is_tupled_arguments) {
          inputs.emplace_back(
              ShapeTree<MaybeOwningDeviceAddress>(&parameter_shapes->front()));
          ExecutionInput& input = inputs.back();
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            VLOG(4) << "tupled input[" << i
                    << "]: " << tracked_buffers[i]->buffer()->buffer().opaque();
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {i},
                  MaybeOwningDeviceAddress(se::ScopedDeviceAddress<uint8_t>(
                      tracked_buffers[i]->buffer()->buffer(),
                      device->local_hardware_id().value(),
                      client->allocator())));
            } else {
              input.SetBuffer({i}, MaybeOwningDeviceAddress(
                                       tracked_buffers[i]->buffer()->buffer()));
            }
          }
        } else {
          inputs.reserve(tracked_buffers.size());
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            VLOG(4) << "untupled input[" << i
                    << "]: " << tracked_buffers[i]->buffer()->buffer().opaque();
            inputs.emplace_back(
                ShapeTree<MaybeOwningDeviceAddress>(&(*parameter_shapes)[i]));
            ExecutionInput& input = inputs.back();
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {}, MaybeOwningDeviceAddress(se::ScopedDeviceAddress<uint8_t>(
                          tracked_buffers[i]->buffer()->buffer(),
                          device->local_hardware_id().value(),
                          client->allocator())));
            } else {
              input.SetBuffer({}, MaybeOwningDeviceAddress(
                                      tracked_buffers[i]->buffer()->buffer()));
            }
          }
        }

        client->blocking_thread_pool()->ScheduleWhenReady(
            input_deps, [execute_fn(std::move(execute_fn)),
                         inputs(std::move(inputs))]() mutable {
              execute_fn(std::move(inputs));
            });
      };
  client_->non_blocking_thread_pool()->ScheduleWhenReady(
      prepare_input_deps, std::move(prepare_inputs));

  // Create output TFRT buffers.
  std::optional<Future<>> future;
  if (fill_future) {
    future = CreateFutureForEvent(complete_event);
  }
  return Result({/*future=*/std::move(future),
                 /*buffers=*/std::move(outputs)});
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtGpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<Future<>>>& returned_futures) const {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
                                          options.launch_id);
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
  if (num_addressable_devices == 1 && !ThisThreadIsInsideHostCallback()) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = addressable_device_logical_ids_[0].replica;
    const int partition = addressable_device_logical_ids_[0].partition;

    // TODO(b/382117736): Dump HLO snapshot.
    // Dump once before running, in case there's a crash.
    // MaybeDumpHloSnapshot(gpu_executable_->module(), options.launch_id,
    //                      argument_handles[0], {});

    auto statusor = ExecuteHelper(argument_handles[0], replica, partition,
                                  options, returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    // TODO(b/382117736): Dump HLO snapshot.
    // MaybeDumpHloSnapshot(cpu_executable_->module(), options.launch_id,
    //                      argument_handles[0], wrapped_results[0]);
  } else {
    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;
      const int device_id = (*device_assignment_)(replica, partition);
      TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                          client_->LookupDevice(PjRtGlobalDeviceId(device_id)));
      TfrtGpuDevice* gpu_device =
          tensorflow::down_cast<TfrtGpuDevice*>(pjrt_device);

      VLOG(1) << "Try to run ExecuteHelper for " << name() << " on device "
              << gpu_device->DebugString()
              << ", launch_id: " << options.launch_id;

      // Gang schedule collectives to ensure that collectives with the same
      // launch_id are run at the same time. We conservatively run only one
      // collective at a time, because we may not have enough threads to run
      // arbitrary number of collectives concurrently.
      client_->non_blocking_thread_pool()->Schedule(
          [this, replica, partition, i, &argument_handles, &options,
           &returned_futures, &wrapped_results, &mu, &running, &failed,
           &first_failure_status] {
            auto statusor =
                ExecuteHelper(argument_handles[i], replica, partition, options,
                              returned_futures.has_value());
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
TfrtGpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecuteSharded",
                                          tsl::profiler::ContextType::kPjRt,
                                          options.launch_id);
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
          ExecuteHelper(argument_handles,
                        addressable_device_logical_ids_[i].replica,
                        addressable_device_logical_ids_[i].partition, options,
                        fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->global_device_id().value());
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtGpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<Future<>>& returned_future,
    bool fill_future) const {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecutePortable",
                                          tsl::profiler::ContextType::kPjRt,
                                          options.launch_id);
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
  TF_ASSIGN_OR_RETURN(auto result,
                      ExecuteHelper(argument_handles,
                                    /*replica=*/0,
                                    /*partition=*/0, options, fill_future,
                                    tsl::down_cast<TfrtGpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}

absl::string_view TfrtGpuExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    return "<unknown executable>";
  }
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
TfrtGpuExecutable::GetHloModules() const {
  std::vector<std::shared_ptr<HloModule>> modules;
  modules.reserve(executables_.size());
  for (const auto& local_exec : executables_) {
    if (!local_exec->executable()->has_module()) {
      return InvalidArgument("Executable does not have HLO modules.");
    }
    modules.push_back(local_exec->executable()->shared_module());
  }
  return std::move(modules);
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
TfrtGpuExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  if (addressable_devices().empty()) {
    return Unimplemented(
        "GetOutputMemoryKinds is not supported when there are no addressable "
        "devices in TfrtGpuExecutable.");
  }
  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * default_memory_space,
                      addressable_devices()[0]->default_memory_space());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(
        std::vector<absl::string_view> memory_kind,
        MemoryKindsFromShape(shape, default_memory_space->kind()));
    out.push_back(memory_kind);
  }
  return out;
}

absl::Status TfrtGpuExecutable::SetUpDonation(bool tuple_inputs) {
  parameters_that_must_be_donated_.reserve(executables_.size());
  for (auto& executable : executables_) {
    TF_ASSIGN_OR_RETURN(std::vector<int> parameters_to_donate,
                        ComputeParametersThatMustBeDonated(
                            executable->executable()->module(), tuple_inputs));
    parameters_that_must_be_donated_.emplace_back(
        std::move(parameters_to_donate));
  }
  return absl::OkStatus();
}

absl::StatusOr<CompiledMemoryStats> TfrtGpuExecutable::GetCompiledMemoryStats()
    const {
  if (executables_.size() != 1) {
    return Unimplemented(
        "Retrieving CompiledMemoryStats is not supported for multiple "
        "executables.");
  }
  CompiledMemoryStats memory_stats = CompiledMemoryStats();
  memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
  const BufferAssignmentProto* proto =
      executables_[0]->executable()->buffer_assignment_proto();
  if (proto != nullptr) {
    memory_stats.serialized_buffer_assignment = proto->SerializeAsString();
    TF_ASSIGN_OR_RETURN(memory_stats.peak_memory_in_bytes,
                        ComputePeakMemory(*proto));
  }
  memory_stats.PopulateBufferStatsFromAllocations(
      executables_[0]->executable()->GetAllocations());
  return memory_stats;
}
}  // namespace xla
