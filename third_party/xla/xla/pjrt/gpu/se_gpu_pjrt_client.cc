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

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/collectives/allocator_memory_registration.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/client/local_client.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/device_event_utils.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/coordination/coordination_service_agent.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_cross_host_transfer.pb.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_runtime_abi_version.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/host_to_device_transfer_manager.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se/buffer_sequencing_event.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/pjrt/se/pjrt_stream_executor_client.h"
#include "xla/pjrt/se/se_raw_buffer.h"
#include "xla/pjrt/se/tracked_device_buffer.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/hang_watchdog.h"
#include "xla/runtime/process_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/device_interconnect_resource.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "xla/tsl/util/env_var.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/traceme.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM) || \
    defined(TENSORFLOW_USE_SYCL)
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/pjrt/gpu/gpu_metrics.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/se/stream_executor_executable.pb.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/tsl/framework/scoped_allocation_trace.h"
#include "xla/xla.pb.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM || TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#include "xla/stream_executor/rocm/rocm_device_address_vmm_allocator.h"
#endif

#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/util.h"

namespace xla {

absl::Status RunCallbackOnStream(
    se::Stream* stream, AsyncWorkRunner* async_work_runner,
    absl::AnyInvocable<void() &&> callback,
    absl::AnyInvocable<void(absl::Status) &&> error_callback) {
  if (error_callback) {
    error_callback = [cb = std::move(error_callback),
                      worker = async_work_runner](absl::Status status) mutable {
      worker->Execute(
          [cb = std::move(cb), status]() mutable { std::move(cb)(status); });
    };
  }
  return stream->DoHostCallbackWithStatus(
      [cb = std::move(callback), worker = async_work_runner]() mutable {
        worker->Execute([cb = std::move(cb)]() mutable { std::move(cb)(); });
        return absl::OkStatus();
      },
      std::move(error_callback));
}

static std::shared_ptr<StreamExecutorGpuTopologyDescription>
CreateSEGpuTopology(absl::string_view platform_name,
                    std::shared_ptr<const GpuTopology> gpu_topology,
                    se::StreamExecutor* s) {
  std::optional<se::GpuTargetConfigProto> target_config;
  // Temporary ability to disable TargetConfig via env var until
  // internal tests can be fixed.
  const char* disable_target_config_str =
      std::getenv("PJRT_GPU_SE_DISABLE_TARGET_CONFIG");
  int disable_target_config = 0;
  if (s &&
      (!disable_target_config_str ||
       !absl::SimpleAtoi(disable_target_config_str, &disable_target_config) ||
       disable_target_config != 1)) {
    target_config = xla::gpu::GpuTargetConfig(s).ToProto();
  }
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attrs;
  if (target_config.has_value()) {
    std::string attr;
    if (tsl::protobuf::TextFormat::PrintToString(*target_config, &attr)) {
      attrs["target_config"] = std::move(attr);
    }
  }

  std::string host_target_machine_options;
  if (tsl::protobuf::TextFormat::PrintToString(
          xla::cpu::TargetMachineOptions().ToProto(),
          &host_target_machine_options)) {
    attrs["host_target_machine_options"] =
        std::move(host_target_machine_options);
  }

  return std::make_shared<xla::StreamExecutorGpuTopologyDescription>(
      tsl::Fingerprint64(platform_name), platform_name, std::move(gpu_topology),
      attrs, target_config);
}

static se::StreamExecutor* GetFirstExecutor(
    const std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>& devices) {
  for (const auto& d : devices) {
    if (auto* local_device_state = d.get()->local_device_state()) {
      return local_device_state->executor();
    }
  }
  return nullptr;
}

StreamExecutorGpuClient::StreamExecutorGpuClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::unique_ptr<StreamExecutorGpuRawClient> raw_client,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    bool abort_collectives_on_failure,
    std::shared_ptr<xla::StreamExecutorGpuTopologyDescription> topology,
    std::optional<int> num_nodes,
    std::shared_ptr<gpu::AllocatorMemoryRegistration> memory_registration)
    : xla::PjRtStreamExecutorClient(
          platform_name, client, std::move(devices), process_index,
          /*memory_spaces=*/{},  // Initialized below.
          std::move(topology), std::move(raw_client), std::move(kv_store)),
      abort_collectives_on_failure_(abort_collectives_on_failure),
      memory_registration_(std::move(memory_registration)) {
  VLOG(1) << absl::StreamFormat(
      "Constructed StreamExecutor GPU client: #devices=%d #num_nodes=%d",
      devices_.size(), num_nodes.value_or(1));
  const int basePinnedId = device_count();
  for (auto* device : addressable_devices()) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    const int id = device->id();
    auto memory_space =
        std::make_unique<StreamExecutorGpuHbmMemorySpace>(id, device);
    tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)->AttachMemorySpace(
        memory_space.get(), /*is_default=*/true);
    owned_memory_spaces_.push_back(std::move(memory_space));
    auto pinned =
        std::make_unique<PinnedHostMemorySpace>(basePinnedId + id, device);
    tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)->AttachMemorySpace(
        pinned.get());
    owned_memory_spaces_.push_back(std::move(pinned));
  }
  for (const std::unique_ptr<PjRtMemorySpace>& memory_space :
       owned_memory_spaces_) {
    memory_spaces_.push_back(memory_space.get());
  }

  // We don't promise anything about the order of memory spaces, but this
  // sorting is done for consistency with the device list that's sorted above.
  absl::c_sort(memory_spaces_,
               [](const PjRtMemorySpace* a, const PjRtMemorySpace* b) {
                 return a->id() < b->id();
               });
}

absl::string_view StreamExecutorGpuClient::platform_version() const {
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#if TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)  // rocm
  // TF_ROCM_VERSION format may change in future. Use it
  // cautiously
  return "rocm " STRINGIFY(TF_ROCM_VERSION);
#elif GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
  return "cuda " STRINGIFY(CUDART_VERSION);
// TODO(intel-tf): Oneapi multiple platform version support
// will be added in future, for now, oneapi 2026.0 is supported
#elif TENSORFLOW_USE_SYCL                     // oneapi
  return "oneapi 2026.0";
#else
  return "<unknown>";
#endif  // TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)
}

std::optional<PjRtPluginAttributes> StreamExecutorGpuClient::plugin_attributes()
    const {
  PjRtPluginAttributes attrs;
  attrs.pjrt_c_api_major_version = 0;
  attrs.pjrt_c_api_minor_version = 0;
  attrs.attributes["supports_cross_host_transfers"] = PjRtValueType(true);
  return attrs;
}

void StreamExecutorGpuClient::UpdateGlobalProcessInfo(
    absl::Span<xla::coordination::TaskInfo> infos) {
  if (!abort_collectives_on_failure_) {
    return;
  }
  absl::Status s = ::xla::gpu::UpdateGlobalProcessInfo(infos);
  if (!s.ok()) {
    LOG(WARNING) << s;
  }
}

absl::StatusOr<xla::DeviceAssignment>
StreamExecutorGpuClient::GetDefaultDeviceAssignment(int num_replicas,
                                                    int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

absl::Status StreamExecutorGpuClient::UpdateCompileOptionsInternal(
    CompileOptions* options, ExecutableExtras* returned_extras,
    bool lookup_addressable_devices) {
  ABSL_RETURN_IF_ERROR(PjRtStreamExecutorClient::UpdateCompileOptionsInternal(
      options, returned_extras, lookup_addressable_devices));
  options->executable_build_options.set_slice_size(
      tensorflow::down_cast<const StreamExecutorGpuTopologyDescription*>(
          topology())
          ->gpu_topology()
          .slice_size());
  return absl::OkStatus();
}

// ==== Start cross-host transfer implementations ==== //

// Anonymous namespace for se_gpu_pjrt_client.cc internal cross-host transfer
// helpers.
namespace {

// Get the local device state for a given PjRtDevice.
absl::StatusOr<LocalDeviceState*> GetLocalDeviceState(PjRtDevice* device) {
  PjRtStreamExecutorDevice* pjrt_se_device =
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device);
  return pjrt_se_device->GetLocalDeviceState();
}

// Creates a communicator for a cross-host transfer; used by the original
// cross-host transfers API.
absl::StatusOr<std::unique_ptr<Communicator>> CreateTransferCommunicator(
    LocalDeviceState* local_device, gpu::GpuCollectives* gpu_collectives,
    CliqueId clique_id, bool is_sender) {
  VLOG(3) << "Creating a new communicator for cross host transfer, is_sender = "
          << is_sender;

  // Create the communicator.
  //
  // TODO(mwhittaker): The way we are constructing GpuCliqueKeys is a
  // big hack. This code doesn't know the GlobalDeviceId of the sending
  // process. Instead, we use two arbitrary GlobalDeviceIds. This
  // works because NcclCommunicators don't actually use the
  // GlobalDeviceIds. Instead, they just need to the know the number
  // of devices (2 in this case).
  gpu::GpuCliqueKey clique_key(
      /*devices=*/{GlobalDeviceId(0), GlobalDeviceId(1)},
      /*num_local_participants=*/1);
  CliqueIds clique_ids(clique_id);
  gpu::GpuCollectives::Device collectives_device(local_device->executor());
  std::vector<Collectives::DeviceRank> ranks = {
      Collectives::DeviceRank(&collectives_device, RankId(is_sender ? 1 : 0))};
  gpu::GpuCollectives::Config config;

  ABSL_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> communicators,
                   gpu_collectives->CreateCommunicators(clique_key, clique_ids,
                                                        ranks, config));
  CHECK_EQ(communicators.size(), 1);

  return std::move(communicators[0]);
}

// Helper structs / classes for second cross-host transfers API.
using AcquiredCliqueAndCommunicator =
    std::pair<std::shared_ptr<gpu::LockableGpuClique::Lock>,
              gpu::GpuCommunicator*>;

class PreparedTransfer {
 public:
  StreamExecutorGpuRawClient* client_;
  gpu::GpuCliqueKey clique_key_;
  PjRtRawBufferRef raw_buffer_;
  tsl::AsyncValueRef<BufferSequencingEvent> transfer_event_;
  AcquiredCliqueAndCommunicator clique_and_communicator_;
  bool is_sender_;

  PreparedTransfer(StreamExecutorGpuRawClient* client,
                   gpu::GpuCliqueKey clique_key, PjRtRawBufferRef raw_buffer,
                   tsl::AsyncValueRef<BufferSequencingEvent> transfer_event,
                   AcquiredCliqueAndCommunicator clique_and_communicator,
                   bool is_sender)
      : client_(client),
        clique_key_(std::move(clique_key)),
        raw_buffer_(std::move(raw_buffer)),
        transfer_event_(std::move(transfer_event)),
        clique_and_communicator_(std::move(clique_and_communicator)),
        is_sender_(is_sender) {}

  PreparedTransfer(PreparedTransfer&&) = default;
  PreparedTransfer& operator=(PreparedTransfer&&) = default;

  ~PreparedTransfer() {
    if (!transfer_event_ || transfer_event_->IsDefined()) {
      return;
    }
    LOG(WARNING)
        << "PreparedTransfer destroyed with unfulfilled transfer_event.";
    client_->SetEventAsError(
        transfer_event_,
        absl::InternalError(
            "PreparedTransfer destroyed without fulfilling transfer_event."));
  }
};

static absl::Duration PjRtClientWatchdogTimeout() {
  const auto& debug_options = GetDebugOptionsFromFlags();
  absl::Duration timeout = absl::InfiniteDuration();
  if (!debug_options.xla_gpu_execution_terminate_timeout().empty()) {
    if (!absl::ParseDuration(
            debug_options.xla_gpu_execution_terminate_timeout(), &timeout)) {
      LOG(WARNING) << "Failed to parse xla_gpu_execution_terminate_timeout: "
                   << debug_options.xla_gpu_execution_terminate_timeout();
      return absl::InfiniteDuration();
    }
  }
  return timeout;
}

// Acquire the GPU clique and communicator for a given clique key.
absl::StatusOr<AcquiredCliqueAndCommunicator> AcquireCliqueAndCommunicator(
    StreamExecutorGpuRawClient* client, gpu::GpuCollectives* gpu_collectives,
    const gpu::GpuCliqueKey& clique_key,
    absl::Span<const std::vector<GlobalDeviceId>> device_groups,
    gpu::AcquiredCliquesMap& acquired_cliques_map, RankId rank_id,
    se::Stream* stream) {
  gpu::CliqueIdCallback clique_id_callback =
      client->gpu_run_options()->clique_id_callback();

  // Acquire the GPU clique for this receive. Guard the acquisition with a hang
  // watchdog because AcquireGpuClique can block indefinitely inside
  // ncclCommInit waiting for remote nodes during NCCL communicator setup.
  if (!acquired_cliques_map.contains(clique_key)) {
    int32_t device_ordinal = stream->parent()->device_ordinal();
    absl::Duration watchdog_timeout = PjRtClientWatchdogTimeout();

    std::shared_ptr<HangWatchdog::Guard> guard = nullptr;
    if (watchdog_timeout < absl::InfiniteDuration()) {
      std::string watchdog_name =
          absl::StrFormat("[%d] PjRt GPU client AcquireGpuClique for %v",
                          device_ordinal, clique_key);
      guard = HangWatchdog::Global().Watch(
          watchdog_name, watchdog_timeout,
          HangWatchdog::Abort(watchdog_name, watchdog_timeout));
    }

    ABSL_ASSIGN_OR_RETURN(
        acquired_cliques_map[clique_key],
        AcquireGpuClique(gpu_collectives,
                         /*device=*/stream->parent(), RunId(0), clique_key,
                         device_groups, clique_id_callback, rank_id,
                         acquired_cliques_map,
                         /*max_nchannels=*/0));
  }
  std::shared_ptr<gpu::LockableGpuClique::Lock> clique =
      acquired_cliques_map[clique_key];

  // Get the communicator to use for this receive.
  std::optional<Communicator*> maybe_communicator = (*clique)->comm(rank_id);
  if (!maybe_communicator.has_value()) {
    return absl::InternalError(
        "AcquireCliqueAndCommunicator: Unable to get communicator from "
        "acquired GPU clique.");
  }

  return AcquiredCliqueAndCommunicator{
      std::move(clique),
      absl::down_cast<gpu::GpuCommunicator*>(*maybe_communicator)};
}

// Create a `PreparedTransfer` object bundling together state needed to perform
// a transfer.
absl::StatusOr<PreparedTransfer> PrepareTransfer(
    StreamExecutorGpuRawClient* client, gpu::GpuCollectives* gpu_collectives,
    se::Stream* stream, GlobalDeviceId src_global_device_id,
    GlobalDeviceId dst_global_device_id, PjRtRawBufferRef raw_buffer,
    gpu::AcquiredCliquesMap& acquired_cliques_map,
    tsl::AsyncValueRef<BufferSequencingEvent> transfer_event, bool is_sender) {
  GlobalDeviceId src_device(src_global_device_id.value());
  GlobalDeviceId dst_device(dst_global_device_id.value());

  tsl::profiler::TraceMe trace([&] {
    return absl::StrFormat("PrepareTransfer: src=%v dst=%v", src_device,
                           dst_device);
  });

  // Form the GPU clique key.
  // TODO(asrao, mwhittaker): Supply correct incarnations when creating the
  // clique key.
  const gpu::GpuCliqueKey clique_key = gpu::GpuCliqueKey(
      /*devices=*/{src_device, dst_device},
      /*num_local_participants=*/1);

  // Get the clique and communicator for the transfer.
  ABSL_ASSIGN_OR_RETURN(
      AcquiredCliqueAndCommunicator clique_and_communicator,
      AcquireCliqueAndCommunicator(client, gpu_collectives, clique_key,
                                   /*device_groups=*/{{src_device, dst_device}},
                                   acquired_cliques_map,
                                   RankId(is_sender ? 0 : 1), stream));

  // Return the result.
  return PreparedTransfer(client, std::move(clique_key), std::move(raw_buffer),
                          std::move(transfer_event),
                          std::move(clique_and_communicator), is_sender);
}

// Groups transfers by clique key, preserving the order in which each clique
// key first appears in `prepared_transfers`.
//
// Cross-host transfers deadlock unless all participating ranks process clique
// groups in the same order. We guarantee that if every rank submits its
// transfers in a mutually consistent order, the returned grouping will be
// consistent across ranks too.
std::vector<std::pair<gpu::GpuCliqueKey, std::vector<PreparedTransfer>>>
GroupTransfersByCliqueKey(std::vector<PreparedTransfer>&& prepared_transfers) {
  std::vector<std::pair<gpu::GpuCliqueKey, std::vector<PreparedTransfer>>>
      grouped_results;
  absl::flat_hash_map<gpu::GpuCliqueKey, size_t> key_to_group_idx;
  for (auto& prepared_transfer : prepared_transfers) {
    auto [it, inserted] = key_to_group_idx.try_emplace(
        prepared_transfer.clique_key_, grouped_results.size());
    if (inserted) {
      grouped_results.emplace_back(prepared_transfer.clique_key_,
                                   std::vector<PreparedTransfer>{});
    }
    size_t group_idx = it->second;
    grouped_results[group_idx].second.push_back(std::move(prepared_transfer));
  }
  return grouped_results;
}

void FulfillDeviceEvent(PjRtStreamExecutorRawClient* client,
                        LocalDeviceState* local_device_state,
                        se::Stream* stream,
                        tsl::AsyncValueRef<BufferSequencingEvent> device_event,
                        const absl::Status& status) {
  if (!status.ok()) {
    client->SetEventAsError(device_event, status);
    return;
  }
  absl::Status s = client->AllocateAndRecordEvent(
      device_event, local_device_state, stream, "CrossHostTransferBuffers");
  if (!s.ok()) {
    client->SetEventAsError(device_event, s);
  }
}

absl::Status WaitForDeviceEventRefsOnStream(
    PjRtDeviceEventSpan device_event_refs, se::Stream* stream) {
  for (size_t i = 0; i < device_event_refs.size(); ++i) {
    PjRtDeviceEventPtr event = device_event_refs[i];
    tsl::AsyncValuePtr<BufferSequencingEvent> event_ref =
        event.down_cast<BufferSequencingEvent>();
    if (!event_ref) {
      return InvalidArgument(
          "WaitForDeviceEventRefsOnStream assumes that all input "
          "PjRtDeviceEventRefs are backed by BufferSequencingEventRefs.");
    }
    event_ref->WaitForEventOnStream(stream);
    if (auto* status = event_ref.value()->GetErrorIfPresent();
        status != nullptr) {
      return *status;
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<PjRtDeviceEventRefVector>
StreamExecutorGpuRawClient::CrossHostTransferBuffers(
    PjRtDeviceEventRefVector transfer_dependencies,
    std::vector<CommonPjRtClient::CrossHostTransferSpec> transfer_specs) {
  // Validate arguments.
  for (int i = 0; i < transfer_specs.size(); ++i) {
    if (transfer_specs[i].raw_buffer->memory_space()->devices().size() != 1) {
      return InvalidArgument(
          "CrossHostTransferBuffers: Received a raw buffer with a memory space "
          "that is not attached to exactly 1 device.");
    }
    PjRtDevice* buffer_device =
        transfer_specs[i].raw_buffer->memory_space()->devices()[0];
    if (!buffer_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostTransferBuffers: raw buffer %d is on non-addressable "
          "device with global device id %d.",
          i, buffer_device->global_device_id().value());
    }
    // Each transfer must be between an addressable and a non-addressable
    // device. If both devices are addressable, then both a data transfer and a
    // 'normal' XLA SPMD executable may try to acquire the same GPU clique,
    // causing issues.
    GlobalDeviceId remote_id = (transfer_specs[i].src_global_device_id ==
                                buffer_device->global_device_id())
                                   ? transfer_specs[i].dst_global_device_id
                                   : transfer_specs[i].src_global_device_id;
    ABSL_ASSIGN_OR_RETURN(PjRtDevice * remote_device,
                     buffer_device->client()->LookupDevice(remote_id));
    if (remote_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostTransferBuffers: remote device for buffer %d is "
          "addressable (global device id %d), but cross-host transfers must "
          "be between an addressable and a non-addressable device.",
          i, remote_id.value());
    }
  }

  // Group the transfers by their buffers' device.
  absl::flat_hash_map<PjRtDevice*, std::vector<int>> transfers_by_device;
  for (int i = 0; i < transfer_specs.size(); ++i) {
    PjRtDevice* buffer_device =
        transfer_specs[i].raw_buffer->memory_space()->devices()[0];
    transfers_by_device[buffer_device].push_back(i);
  }

  // We will register a single transfer event for all transfers to/from the same
  // device. We will collect the references to those events inside
  // output_transfer_events. This will eventually be returned to the user.
  std::vector<PjRtDeviceEventRef> output_transfer_events(transfer_specs.size(),
                                                         PjRtDeviceEventRef());

  // Schedule transfers.
  for (const auto& [device, transfer_idxs] : transfers_by_device) {
    const GlobalDeviceId device_id = device->global_device_id();

    // Create a transfer event for transfers on this device.
    tsl::AsyncValueRef<BufferSequencingEvent> transfer_event =
        BufferSequencingEvent::Create(this->async_work_runner());

    // Form transfer specs.
    std::vector<CommonPjRtClient::CrossHostTransferSpec> curr_transfer_specs;
    curr_transfer_specs.reserve(transfer_idxs.size());

    for (int idx : transfer_idxs) {
      // Keep raw_buffer alive until transfer_event completes, preventing
      // the allocation from being freed while the transfer is in-flight.
      transfer_event.AndThen(
          [raw_buffer = transfer_specs[idx].raw_buffer]() {});
      curr_transfer_specs.push_back(std::move(transfer_specs[idx]));
      output_transfer_events[idx] = PjRtDeviceEventRef(transfer_event);
    }

    // Get the local_device_state and use it to schedule transfers. Fail
    // transfers early if we cannot get the local_device_state.
    absl::StatusOr<LocalDeviceState*> local_device_state =
        tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
            ->GetLocalDeviceState();
    if (!local_device_state.ok()) {
      SetEventAsError(transfer_event, local_device_state.status());
      continue;
    }

    // Launch ScheduleTransfersOnLocalDevice on either the async dispatch thread
    // of the calling thread.
    if ((*local_device_state)->async_dispatch_thread()) {
      (*local_device_state)
          ->async_dispatch_thread()
          ->Schedule(tsl::WithCurrentContext(
              [this, local_device_state, device_id, transfer_dependencies,
               curr_transfer_specs = std::move(curr_transfer_specs),
               transfer_event = std::move(transfer_event)]() mutable {
                ScheduleTransfersOnLocalDevice(*local_device_state, device_id,
                                               std::move(transfer_event),
                                               std::move(transfer_dependencies),
                                               std::move(curr_transfer_specs));
              }));
    } else {
      ScheduleTransfersOnLocalDevice(
          *local_device_state, device_id, std::move(transfer_event),
          transfer_dependencies, std::move(curr_transfer_specs));
    }
  }

  PjRtDeviceEventRefVector result;
  result.reserve(output_transfer_events.size());
  for (auto& ev : output_transfer_events) {
    result.push_back(std::move(ev));
  }
  return result;
}

void StreamExecutorGpuRawClient::ScheduleTransfersOnLocalDevice(
    LocalDeviceState* local_device_state, GlobalDeviceId device_id,
    tsl::AsyncValueRef<BufferSequencingEvent> transfer_event,
    PjRtDeviceEventRefVector transfer_dependencies,
    std::vector<CommonPjRtClient::CrossHostTransferSpec> transfer_specs) {
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat(
            "[%v] StreamExecutorGpuClient::ScheduleTransfersOnLocalDevice",
            local_device_state->local_device_id()),
        {{"num_buffers", transfer_specs.size()}});
  });

  se::Stream* stream = local_device_state->GetDeviceToDeviceStream();
  std::vector<PreparedTransfer> prepared_transfers;
  prepared_transfers.reserve(transfer_specs.size());

  auto prepare_transfers = [&]() -> absl::Status {
    gpu::GpuCollectives* gpu_collectives =
        gpu::GpuCollectives::Default(stream->parent()->GetPlatform()->Name());

    gpu::AcquiredCliquesMap acquired_cliques_map;
    for (int i = 0; i < transfer_specs.size(); ++i) {
      const bool is_sender =
          device_id == transfer_specs[i].src_global_device_id;
      ABSL_ASSIGN_OR_RETURN(
          PreparedTransfer prepared_transfer,
          PrepareTransfer(this, gpu_collectives, stream,
                          transfer_specs[i].src_global_device_id,
                          transfer_specs[i].dst_global_device_id,
                          std::move(transfer_specs[i].raw_buffer),
                          acquired_cliques_map, transfer_event, is_sender));

      prepared_transfers.push_back(std::move(prepared_transfer));
    }

    return absl::OkStatus();
  };

  if (absl::Status status = prepare_transfers(); !status.ok()) {
    FulfillDeviceEvent(this, local_device_state, stream, transfer_event,
                       status);
    return;
  }

  // Form the closure called for each group of transfers.
  auto launch_transfer_group =
      [](gpu::GpuCommunicator* gpu_communicator,
         absl::Span<PreparedTransfer> prepared_transfers,
         se::Stream* stream) -> absl::Status {
    for (PreparedTransfer& prepared_transfer : prepared_transfers) {
      // Launch the transfer.
      auto mem = prepared_transfer.raw_buffer_
                     ->down_cast<PjRtStreamExecutorRawBuffer>()
                     ->device_buffer();

      // We always set `peer` to RankId(1) if we are the sender, and RankId(0)
      // if we are the receiver. This is because `PrepareTransfer()` always
      // acquires a GPU clique where the sender is rank 0 and the receiver is
      // rank 1.
      if (prepared_transfer.is_sender_) {
        ABSL_RETURN_IF_ERROR(gpu_communicator->LaunchSend(
            /*send_buffer=*/mem->mem(),
            /*dtype=*/U8,
            /*count=*/mem->mem().size(),
            /*peer=*/RankId(1),
            /*executor=*/gpu::GpuCollectives::On(*stream)));
      } else {
        ABSL_RETURN_IF_ERROR(gpu_communicator->LaunchRecv(
            /*recv_buffer=*/mem->mem(),
            /*dtype=*/U8,
            /*count=*/mem->mem().size(),
            /*peer=*/RankId(0),
            /*executor=*/gpu::GpuCollectives::On(*stream)));
      }
    }
    return absl::OkStatus();
  };

  // Form the closure to schedule on the device's execute thread.
  auto execute_transfers_fn =
      [this, local_device_state, stream,
       transfer_dependencies = std::move(transfer_dependencies),
       prepared_transfers = std::move(prepared_transfers),
       launch_transfer_group = std::move(launch_transfer_group),
       transfer_event = std::move(transfer_event)]() mutable {
        // Wait for transfer dependencies.
        if (auto status =
                WaitForDeviceEventRefsOnStream(transfer_dependencies, stream);
            !status.ok()) {
          FulfillDeviceEvent(this, local_device_state, stream, transfer_event,
                             status);
          return;
        }

        // Group transfers by GPU clique.
        std::vector<std::pair<gpu::GpuCliqueKey, std::vector<PreparedTransfer>>>
            grouped_transfers =
                GroupTransfersByCliqueKey(std::move(prepared_transfers));

        // Transfers for a particular clique are executed as a group. This
        // vector holds group futures for each clique_key in grouped_transfers.
        std::vector<Future<>> group_futures;
        group_futures.reserve(grouped_transfers.size());

        for (auto& [clique_key, curr_transfers] : grouped_transfers) {
          tsl::profiler::TraceMe trace([&k = clique_key] {
            return tsl::profiler::TraceMeEncode("LaunchTransfer",
                                                {{"clique", k}});
          });

          // Get the communicator on which we will execute this group of
          // transfers. We assume each clique key is associated with a unique
          // communicator, so we just take the communicator of the first
          // transfer_idx of this clique key.
          gpu::GpuCommunicator* gpu_communicator =
              curr_transfers[0].clique_and_communicator_.second;

          // Launch the group of transfers.
          group_futures.push_back(gpu_communicator->GroupExecute(
              [&launch_transfer_group, &curr_transfers = curr_transfers, stream,
               gpu_communicator]() {
                return launch_transfer_group(
                    gpu_communicator, absl::MakeSpan(curr_transfers), stream);
              }));
        }

        // On a separate thread pool, await group futures and fulfill buffer
        // sequencing events and promises.
        Future<> all_transfers_future = JoinFutures(group_futures);

        all_transfers_future.OnReady(
            *async_work_runner(),
            [this, local_device_state, stream, transfer_event,
             grouped_transfers = std::move(grouped_transfers)](
                const absl::Status& status) mutable {
              // Add transfer_event onto the stream.
              FulfillDeviceEvent(this, local_device_state, stream,
                                 transfer_event, status);
            });
      };

  // Schedule transfers on the execute thread.
  local_device_state->execute_thread()->Schedule(
      std::move(execute_transfers_fn));
}

void StreamExecutorGpuRawClient::ScheduleRemoteSend(
    PjRtMemorySpace* memory_space, PjRtRawBufferRef raw_buffer,
    PjRtDeviceEventRefVector definition_events,
    PjRtDeviceEventPromiseRef usage_event_promise,
    Future<std::string> serialized_descriptor,
    PjRtBuffer::RemoteSendCallback on_done) {
  BufferSequencingEventRef usage_event =
      BufferSequencingEvent::Create(this->async_work_runner());

  // Keep memory alive until the event is done.
  usage_event.AndThen([raw_buffer]() {});

  serialized_descriptor.OnReady(
      [this, on_done = std::move(on_done),
       definition_events = std::move(definition_events),
       raw_buffer = std::move(raw_buffer),
       usage_event](absl::StatusOr<std::string> serialized_descriptor) mutable {
        if (!serialized_descriptor.ok()) {
          on_done(serialized_descriptor.status(),
                  /*sends_were_enqueued=*/false);
          SetEventAsError(usage_event, serialized_descriptor.status());
          return;
        }
        PjRtDeviceEventSpan definition_events_span(definition_events);
        ExecuteWhenReady(
            definition_events_span, async_work_runner(),
            [this, on_done = std::move(on_done),
             definition_events = std::move(definition_events),
             raw_buffer = std::move(raw_buffer),
             usage_event = std::move(usage_event),
             serialized_descriptor =
                 *std::move(serialized_descriptor)]() mutable {
              bool sends_were_enqueued = false;
              auto status = [&]() -> absl::Status {
                ABSL_RETURN_IF_ERROR(GetErrors(definition_events));

                auto* se_raw_buffer =
                    raw_buffer->down_cast<PjRtStreamExecutorRawBuffer>();
                auto* local_device = se_raw_buffer->local_device();
                auto* executor = absl::down_cast<se::gpu::GpuExecutor*>(
                    local_device->executor());
                auto* stream = local_device->GetDeviceToDeviceStream();

                auto mem = se_raw_buffer->device_buffer()->mem();
                const size_t size = se_raw_buffer->GetOnDeviceSizeInBytes();
                ABSL_RETURN_IF_ERROR(WaitForAllocation(stream, *raw_buffer));

                StreamExecutorGpuCrossHostRecvDescriptor desc;
                if (!desc.ParseFromString(serialized_descriptor)) {
                  return xla::Internal("Failed to parse serialized descriptor");
                }

                // Import the receiver's allocations. Note that the returned
                // addresses are the beginning of the allocation range, so we
                // need to add the offset to get the actual address.
                if (!desc.buffer_handle().empty()) {
                  ABSL_ASSIGN_OR_RETURN(
                      std::shared_ptr<se::DeviceAddressBase> dst_base,
                      GetOrImportFabricHandle(executor, desc.buffer_handle()));

                  // Copy the source buffer into the destination buffer.
                  se::DeviceAddressBase dst(
                      static_cast<char*>(dst_base->opaque()) +
                          desc.buffer_offset(),
                      size);
                  ABSL_RETURN_IF_ERROR(stream->Memcpy(&dst, mem, size));
                }

                // Signal transfer completion by setting the value to 1. The
                // receiver polls this value.
                ABSL_ASSIGN_OR_RETURN(
                    std::shared_ptr<se::DeviceAddressBase> flag_base,
                    GetOrImportFabricHandle(executor, desc.flag_handle()));
                se::DeviceAddressBase flag(
                    static_cast<char*>(flag_base->opaque()) +
                        desc.flag_offset(),
                    sizeof(uint32_t));
                ABSL_RETURN_IF_ERROR(stream->Memset32(&flag, 1, sizeof(uint32_t)));

                // At this point, we must return `sends_were_enqueued = true` to
                // indicate that the send has been successfully enqueued.
                sends_were_enqueued = true;

                return AllocateAndRecordEvent(usage_event, local_device, stream,
                                              "CrossHostSendBuffers");
              }();
              std::move(on_done)(status, sends_were_enqueued);
              if (!status.ok()) {
                VLOG(2) << "CrossHostSendBuffers failed: " << status;
                SetEventAsError(usage_event, status);
              }
            });
      });
  usage_event_promise.Set(PjRtDeviceEventRef(std::move(usage_event)));
}

namespace {

// Keeps track of the state of an in-flight cross-host recv.
class CrossHostRecvState {
 public:
  CrossHostRecvState(int num_buffers, se::DeviceAddressBase flag,
                     HostMemoryAllocator* host_memory_allocator)
      : num_buffers_(num_buffers),
        flag_(std::move(flag)),
        host_memory_allocator_(host_memory_allocator),
        cancellation_statuses_(num_buffers) {}

  // Waits until all transfers in the batch are complete or cancelled. Returns
  // a list of cancellations statuses, one for each buffer.
  absl::StatusOr<std::vector<absl::Status>> Wait(se::Stream* stream) {
    static constexpr absl::Duration kInitialDelay = absl::Microseconds(2);
    static constexpr absl::Duration kPeriod = absl::Microseconds(10);

    // Keeps track of indices of buffers that are still pending.
    std::vector<int> buffer_indices;
    buffer_indices.reserve(num_buffers_);
    for (int i = 0; i < num_buffers_; ++i) {
      buffer_indices.push_back(i);
    }

    absl::SleepFor(kInitialDelay);

    // Allocate host memory to copy the flags into from the pinned host memory
    // allocator to optimize DMA.
    auto values_storage =
        host_memory_allocator_->Allocate(sizeof(uint32_t) * num_buffers_);
    uint32_t* const values = reinterpret_cast<uint32_t*>(values_storage.get());

    while (true) {
      // Poll the flags to see if any transfers are complete. We may consider
      // replacing with a 1-SM polling kernel that listens to both completion
      // signals from the sender and cancellation signals from the receiver.
      ABSL_RETURN_IF_ERROR(
          stream->Memcpy(values, flag_, sizeof(uint32_t) * num_buffers_));
      ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

      std::vector<int> pending_buffer_indices;
      for (const int index : buffer_indices) {
        {
          absl::MutexLock l(mu_);
          if (!cancellation_statuses_[index].ok()) {
            continue;
          }
        }
        if (values[index] == 0) {
          pending_buffer_indices.push_back(index);
        } else if (values[index] != 1) {
          return xla::Internal(
              "Unexpected cross-host recv flag value (potentially a bug or a "
              "memory corruption): %u",
              values[index]);
        }
      }
      if (pending_buffer_indices.empty()) {
        break;
      }

      buffer_indices = std::move(pending_buffer_indices);
      absl::SleepFor(kPeriod);
    }

    absl::MutexLock l(mu_);
    return cancellation_statuses_;
  }

  // Cancels the transfer associated with the given serialized descriptor. Per
  // the cross-host send/recv API contract, a given transfer must either
  // complete successfully or cancelled exactly once.
  void NotifyCancellation(absl::string_view serialized_descriptor,
                          absl::Status reason,
                          std::function<void(absl::Status)> on_canceled) {
    auto status = [&]() -> absl::Status {
      if (reason.ok()) {
        return xla::InvalidArgument("Cancellation reason must be non-OK");
      }
      StreamExecutorGpuCrossHostRecvDescriptor desc;
      if (!desc.ParseFromString(serialized_descriptor)) {
        return xla::Internal("Failed to parse serialized descriptor");
      }
      if (desc.buffer_index() < 0 || desc.buffer_index() >= num_buffers_) {
        return xla::Internal("Buffer index out of range: [0, %d, %d)",
                             desc.buffer_index(), num_buffers_);
      }
      {
        absl::MutexLock l(mu_);
        auto& status = cancellation_statuses_[desc.buffer_index()];
        if (!status.ok()) {
          return xla::Internal(
              "Received multiple cancellations for the same cross-host recv");
        }
        status = reason;
      }
      VLOG(3) << "Received cancellation request for buffer "
              << desc.buffer_index() << ": " << reason;
      return absl::OkStatus();
    }();
    on_canceled(status);
  }

 private:
  int num_buffers_;
  se::DeviceAddressBase flag_;
  HostMemoryAllocator* host_memory_allocator_;

  absl::Mutex mu_;

  // Cancellation status of each buffer. `OkStatus` indicates that the buffer
  // has not received a cancellation request.
  std::vector<absl::Status> cancellation_statuses_ ABSL_GUARDED_BY(mu_);
};

}  // namespace

absl::StatusOr<PjRtDeviceEventRefVector>
StreamExecutorGpuRawClient::CrossHostReceiveBuffersInto(
    absl::Span<const PjRtRawBufferRef> buffers,
    PjRtCrossHostRecvNotifier notifier,
    PjRtDeviceEventSpan transfer_dependency_avs) {
  // Validate arguments.
  if (buffers.empty()) {
    return InvalidArgument(
        "buffers parameter empty in CrossHostReceiveBuffersInto");
  }

  auto* const memory_space = buffers[0]->memory_space();
  if (memory_space->kind_id() != StreamExecutorGpuHbmMemorySpace::kKindId) {
    return xla::InvalidArgument(
        "Cross-host transfers are only supported for HBM buffers, but got "
        "buffers on memory space '%s'",
        memory_space->kind());
  }

  std::vector<PjRtRawBufferRef> raw_buffers;
  raw_buffers.reserve(buffers.size());
  for (const PjRtRawBufferRef& raw_buffer : buffers) {
    if (raw_buffer->memory_space() != memory_space) {
      return xla::InvalidArgument(
          "Cross-host transfers require all buffers in a batch to be on the "
          "same device");
    }
    raw_buffers.push_back(raw_buffer);
  }

  // All buffers are on the same device.
  LocalDeviceState* local_device =
      raw_buffers[0]->down_cast<PjRtStreamExecutorRawBuffer>()->local_device();
  se::Stream* stream = local_device->GetDeviceToDeviceStream();

  std::vector<BufferSequencingEventRef> buffer_sequencing_events;
  buffer_sequencing_events.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    buffer_sequencing_events.push_back(
        BufferSequencingEvent::Create(this->async_work_runner()));
  }

  // Allocate a `uint32_t` flag per buffer. The sender flips the flags from 0 to
  // 1 to signal transfer completion. Uses uint32_t because that's the smallest
  // unit that can be set atomically, e.g., Memset32.
  ABSL_ASSIGN_OR_RETURN(
      PjRtRawBufferRef flag_buffer,
      AllocateRawBuffer(memory_space, sizeof(uint32_t) * raw_buffers.size(),
                        /*retry_on_oom=*/true, {}));

  auto recv = [this, raw_buffers = std::move(raw_buffers),
               flag_buffer = std::move(flag_buffer), buffer_sequencing_events,
               notifier = std::move(notifier), local_device, stream]() mutable {
    auto results = [&]() -> absl::StatusOr<std::vector<absl::Status>> {
      auto* executor =
          absl::down_cast<se::gpu::GpuExecutor*>(local_device->executor());

      se::DeviceAddressBase flag_address;
      {
        auto* se_flag_buffer =
            flag_buffer->down_cast<PjRtStreamExecutorRawBuffer>();
        tsl::AsyncValueRef<RawSEDeviceMemory> flag_mem =
            se_flag_buffer->device_buffer();
        ABSL_RETURN_IF_ERROR(WaitForAllocation(stream, *se_flag_buffer));
        flag_address = flag_mem->mem();

        // Keep flags alive until the transfer is done.
        for (const auto& buffer_sequencing_event : buffer_sequencing_events) {
          buffer_sequencing_event.AndThen([flag_mem]() {});
        }
      }

      // Set all flags to 0 before starting the transfer.
      ABSL_RETURN_IF_ERROR(stream->Memset32(&flag_address, 0,
                                       sizeof(uint32_t) * raw_buffers.size()));

      // Export the address range which contains the flag buffer. The flags are
      // addressed as offsets from the beginning of this range.
      ABSL_ASSIGN_OR_RETURN(
          const std::string flag_handle,
          GetOrExportFabricHandle(executor, flag_address.opaque()));
      ABSL_ASSIGN_OR_RETURN(auto flag_range,
                       executor->GetAllocationRange(flag_address.opaque()));
      const int64_t flag_offset =
          reinterpret_cast<intptr_t>(flag_address.opaque()) -
          reinterpret_cast<intptr_t>(flag_range.opaque());

      StreamExecutorGpuCrossHostRecvDescriptor desc;
      std::vector<PjRtCrossHostRecvDescriptors> descriptors;
      descriptors.reserve(raw_buffers.size());

      for (int i = 0; i < raw_buffers.size(); ++i) {
        auto* se_raw_buffer =
            raw_buffers[i]->down_cast<PjRtStreamExecutorRawBuffer>();

        tsl::AsyncValueRef<RawSEDeviceMemory> mem =
            se_raw_buffer->device_buffer();
        ABSL_RETURN_IF_ERROR(WaitForAllocation(stream, *raw_buffers[i]));

        // Keep mem alive until the Recv has finished executing.
        buffer_sequencing_events[i].AndThen([mem]() {});

        // Export the buffer/flag fabric handles and use them as descriptors to
        // be sent to the sender.
        desc.Clear();
        desc.set_buffer_index(i);
        if (mem->mem().size() > 0) {
          ABSL_ASSIGN_OR_RETURN(
              *desc.mutable_buffer_handle(),
              GetOrExportFabricHandle(executor, mem->mem().opaque()));
          ABSL_ASSIGN_OR_RETURN(auto range,
                           executor->GetAllocationRange(mem->mem().opaque()));
          desc.set_buffer_offset(
              reinterpret_cast<intptr_t>(mem->mem().opaque()) -
              reinterpret_cast<intptr_t>(range.opaque()));
        }
        desc.set_flag_handle(flag_handle);
        desc.set_flag_offset(flag_offset + sizeof(uint32_t) * i);

        descriptors.push_back(
            PjRtCrossHostRecvDescriptors{{desc.SerializeAsString()}});
      }

      auto state = std::make_shared<CrossHostRecvState>(
          raw_buffers.size(), flag_address, GetHostMemoryAllocator());

      // Notify the receiver of the descriptors and cancellation callback. The
      // caller is responsible for sending the descriptors to the sender and/or
      // cancelling the transfer if needed.
      VLOG(3) << "Notifying receiver of descriptors for cross-host recv of "
              << descriptors.size() << " buffers";
      notifier(PjRtCrossHostRecvState{
          /*descriptors=*/std::move(descriptors),
          /*cancel_notifier=*/
          absl::bind_front(&CrossHostRecvState::NotifyCancellation, state),
      });

      VLOG(3) << "Waiting for cross-host recv completion";
      return state->Wait(stream);
    }();
    if (!results.ok()) {
      VLOG(2) << "CrossHostReceiveBuffersInto failed: " << results.status();
      for (const auto& buffer_sequencing_event : buffer_sequencing_events) {
        SetEventAsError(buffer_sequencing_event, results.status());
      }
      return;
    }

    VLOG(3) << "Cross-host recv completed";
    CHECK_EQ(results->size(), buffer_sequencing_events.size());
    for (int i = 0; i < buffer_sequencing_events.size(); ++i) {
      absl::Status status = (*results)[i];
      if (status.ok()) {
        status =
            AllocateAndRecordEvent(buffer_sequencing_events[i], local_device,
                                   stream, "CrossHostReceiveBuffersInto");
      }
      if (!status.ok()) {
        SetEventAsError(buffer_sequencing_events[i], status);
      }
    }
  };
  async_work_runner()->Execute(std::move(recv));

  PjRtDeviceEventRefVector definition_events;
  for (const auto& buffer_sequencing_event : buffer_sequencing_events) {
    definition_events.push_back(PjRtDeviceEventRef(buffer_sequencing_event));
  }
  return definition_events;
}

absl::StatusOr<std::string> StreamExecutorGpuRawClient::GetOrExportFabricHandle(
    se::StreamExecutor* executor, void* ptr) {
  if (!cache_fabric_handles_) {
    return absl::down_cast<se::gpu::GpuExecutor*>(executor)->ExportFabricHandle(
        ptr);
  }

  absl::MutexLock l(mu_);

  auto key = std::make_pair(executor, ptr);
  const auto it = exported_fabric_handles_.find(key);
  if (it != exported_fabric_handles_.end()) {
    return it->second;
  }

  ABSL_ASSIGN_OR_RETURN(
      std::string handle,
      absl::down_cast<se::gpu::GpuExecutor*>(executor)->ExportFabricHandle(
          ptr));
  exported_fabric_handles_[key] = handle;
  return handle;
}

absl::StatusOr<std::shared_ptr<se::DeviceAddressBase>>
StreamExecutorGpuRawClient::GetOrImportFabricHandle(
    se::StreamExecutor* executor, absl::string_view fabric_handle) {
  auto* const gpu_executor = absl::down_cast<se::gpu::GpuExecutor*>(executor);
  auto import_fabric_handle =
      [&]() -> absl::StatusOr<std::shared_ptr<se::DeviceAddressBase>> {
    ABSL_ASSIGN_OR_RETURN(se::DeviceAddressBase address,
                     gpu_executor->ImportFabricHandle(fabric_handle));
    return std::shared_ptr<se::DeviceAddressBase>(
        new se::DeviceAddressBase(address),
        [gpu_executor](se::DeviceAddressBase* address) {
          gpu_executor->Deallocate(address);
          delete address;
        });
  };

  if (!cache_fabric_handles_) {
    return import_fabric_handle();
  }

  absl::MutexLock l(mu_);

  auto key = std::make_pair(executor, std::string(fabric_handle));
  const auto it = imported_fabric_handles_.find(key);
  if (it != imported_fabric_handles_.end()) {
    return it->second;
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<se::DeviceAddressBase> address,
                   import_fabric_handle());
  return imported_fabric_handles_[key] = std::move(address);
}

// ==== End cross-host transfer implementations ==== //

void StreamExecutorGpuClient::RecordMemoryStats() {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM) || \
    defined(TENSORFLOW_USE_SYCL)
  for (const PjRtDevice* device : addressable_devices()) {
    LocalDeviceState* local_device_state =
        tensorflow::down_cast<const PjRtStreamExecutorDevice*>(device)
            ->local_device_state();
    int64_t free_memory, total_memory;
    if (local_device_state != nullptr) {
      se::StreamExecutor* executor = local_device_state->executor();
      int device_ordinal = executor->device_ordinal();
      if (executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
        gpu_metrics::RecordFreeGpuSystemMemory(device_ordinal, free_memory);
      } else {
        LOG(ERROR) << "Failed to query available memory for GPU "
                   << device_ordinal;
      }
    }
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM || TENSORFLOW_USE_SYCL
}

namespace {

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::shared_ptr<se::GpuCudaMallocAsyncAllocator>>
CreateCudaAsyncAllocator(const LocalDeviceState& device, double memory_fraction,
                         bool reserve_memory, bool create_new_pool,
                         bool sync_mode, bool compute_stats = true) {
  se::StreamExecutor* executor = device.executor();
  int device_ordinal = executor->device_ordinal();

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return Unavailable("Failed to query available memory from device %i",
                       device_ordinal);
  }
  // To allow full GPU memory to be visible to the Cuda Async allocator
  // if using unified memory.
  // When unified memory is enabled, allow GPU memory oversubscription by
  // setting memory_fraction > 1.
  size_t allocator_memory = total_memory * memory_fraction;
  if (reserve_memory) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CudaAsyncAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CudaAsyncAllocator.";
  }

  auto allocator = std::make_shared<se::GpuCudaMallocAsyncAllocator>(
      /*platform_device_id*/ tsl::PlatformDeviceId(device_ordinal),
      /*create_new_pool*/ create_new_pool,
      /*new_pool_size*/ allocator_memory,
      /*reserve_memory*/ reserve_memory,
      /*reserve_memory_size*/ reserve_memory ? allocator_memory : 0,
      /*sync_mode*/ sync_mode,
      /*compute_stats*/ compute_stats);

  allocator->SetStreamAndPreallocateMemory(
      device.compute_stream()->platform_specific_handle().stream);

  return allocator;
}

#else  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
absl::StatusOr<std::shared_ptr<tsl::Allocator>> CreateCudaAsyncAllocator(
    const LocalDeviceState& device, double memory_fraction, bool reserve_memory,
    bool create_new_pool, bool sync_mode, bool compute_stats = true) {
  return FailedPrecondition("CUDA async allocator requires CUDA >= 11.2");
}

#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

// Builds a LocalDeviceState for each GPU present.
absl::StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client, bool schedule_async,
                       std::optional<int> max_inflight_computations) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            max_inflight_computations, /*allow_event_reuse=*/true,
            /*use_callback_stream=*/true, /*device_ordinal=*/-1,
            /*stream_options=*/std::nullopt, schedule_async));
  }
  return std::move(addressable_devices);
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
absl::StatusOr<std::unique_ptr<se::DeviceAddressAllocator>>
GetStreamExecutorGpuDeviceAllocator(
    se::Platform* platform, const GpuAllocatorConfig& allocator_config,
    const std::map<int, std::unique_ptr<LocalDeviceState>>& addressable_devices,
    bool preallocate_host_memory) {
  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  const DebugOptions& debug_options = xla::GetDebugOptionsFromFlags();
  GpuAllocatorConfig::Kind effective_kind = allocator_config.kind;
  if (debug_options.xla_gpu_command_buffer_update_mode() !=
          DebugOptions::ALWAYS_UPDATE &&
      effective_kind != GpuAllocatorConfig::Kind::kVmm) {
    LOG(WARNING) << "xla_gpu_command_buffer_update_mode requires the "
                    "VMM allocator. Overriding allocator kind to kVmm.";
    effective_kind = GpuAllocatorConfig::Kind::kVmm;
  }

  // Set when a single preallocated BFC allocator serves both default and
  // collective memory via spatial partitioning; suppresses the separate
  // collective allocator below.
  bool shared_collective_pool = false;
  switch (effective_kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync: {
      for (const auto& ordinal_and_device : addressable_devices) {
        ABSL_ASSIGN_OR_RETURN(
            auto async_allocator,
            CreateCudaAsyncAllocator(
                *(ordinal_and_device.second), allocator_config.memory_fraction,
                allocator_config.preallocate, false, false, true));
        allocators.push_back(
            {std::move(async_allocator),
             ordinal_and_device.second->compute_stream(),
             /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kDefault});
      }
      break;
    }

    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      // With the spatial-partitioning flag enabled, preallocation lets one BFC
      // allocator over a fixed address range serve both default (lower end) and
      // collective (upper end) memory, so no separate collective allocator is
      // created. Otherwise, use the separate collective allocator below.
      shared_collective_pool =
          allocator_config.preallocate &&
          debug_options.xla_gpu_enable_allocator_spatial_partitioning();
      for (const auto& ordinal_and_device : addressable_devices) {
        ABSL_ASSIGN_OR_RETURN(
            auto bfc_allocator,
            CreateBFCAllocator(ordinal_and_device.second->executor(),
                               allocator_config.memory_fraction,
                               allocator_config.preallocate,
                               allocator_config.gpu_system_memory_size,
                               allocator_config.sub_allocator_alloc_visitors,
                               allocator_config.sub_allocator_free_visitors,
                               /*enable_spatial_partitioning=*/
                               shared_collective_pool));
        allocators.push_back(
            {bfc_allocator, ordinal_and_device.second->compute_stream(),
             /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kDefault});
        if (shared_collective_pool) {
          uint64_t collective_memory_alignment =
              tsl::Allocator::kAllocatorAlignment;
          absl::StatusOr<uint64_t> granularity =
              ordinal_and_device.second->executor()
                  ->GetCollectiveMemoryGranularity();
          if (granularity.ok()) {
            collective_memory_alignment = *granularity;
          }
          allocators.push_back(
              {std::move(bfc_allocator),
               ordinal_and_device.second->compute_stream(),
               /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kCollective,
               /*device_ordinal=*/std::nullopt,
               /*platform=*/nullptr,
               /*min_alignment=*/collective_memory_alignment,
               /*allocation_end=*/tsl::AllocationEnd::kUpper});
        }
      }
      break;
    }

    case GpuAllocatorConfig::Kind::kPlatform: {
      LOG(INFO) << "Using platform (synchronous passthrough) allocator.";
      if (allocator_config.collective_memory_size != 0) {
        LOG(WARNING)
            << "collective_memory_size is non-zero, but allocator kind is set "
               "to \"platform\". Collective memory will not be preallocated.";
      }
      for (const auto& [ordinal, device] : addressable_devices) {
        auto* executor = device->executor();
        auto* stream = device->compute_stream();

        // Default device memory space (XLA color 0 -> StreamExecutor
        // MemorySpace::kDevice = 0)
        auto default_allocator =
            std::make_shared<se::StreamExecutorMemoryAllocator>(
                executor, static_cast<int64_t>(se::MemorySpace::kDevice));
        allocators.push_back(
            {std::move(default_allocator), stream,
             /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kDefault});

        // Collective memory space (XLA color 1 -> StreamExecutor
        // MemorySpace::kCollective = 2)
        auto collective_allocator =
            std::make_shared<se::StreamExecutorMemoryAllocator>(
                executor, static_cast<int64_t>(se::MemorySpace::kCollective));
        allocators.push_back(
            {std::move(collective_allocator), stream,
             /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kCollective});

        // Temp buffer memory space (XLA color 2 -> StreamExecutor
        // MemorySpace::kDevice = 0)
        auto temp_allocator =
            std::make_shared<se::StreamExecutorMemoryAllocator>(
                executor, static_cast<int64_t>(se::MemorySpace::kDevice));
        allocators.push_back(
            {std::move(temp_allocator), stream,
             /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kTempBuffer});

        // Host memory space (StreamExecutor MemorySpace::kHost = 5)
        ABSL_ASSIGN_OR_RETURN(
            auto host_allocator,
            GetGpuHostAllocator(executor, preallocate_host_memory));
        allocators.push_back(
            {std::move(host_allocator), stream,
             /*memory_space=*/static_cast<int>(se::MemorySpace::kHost)});
      }
      return std::make_unique<se::MultiDeviceAdapter>(platform,
                                                      std::move(allocators));
    }

    case GpuAllocatorConfig::Kind::kVmm: {
#if GOOGLE_CUDA
      std::vector<std::pair<se::StreamExecutor*, se::Stream*>> executor_streams;
      executor_streams.reserve(addressable_devices.size());
      for (const auto& [ordinal, device] : addressable_devices) {
        executor_streams.push_back(
            {device->executor(), device->compute_stream()});
      }
      return se::gpu::CudaDeviceAddressVmmAllocator::Create(
          platform, allocator_config.memory_fraction,
          allocator_config.gpu_system_memory_size, executor_streams,
          /*reclaim_exempt_memory_space=*/
          static_cast<int64_t>(gpu::MemorySpaceColor::kCollective));
#elif TENSORFLOW_USE_ROCM
      std::vector<std::pair<se::StreamExecutor*, se::Stream*>> executor_streams;
      executor_streams.reserve(addressable_devices.size());
      for (const auto& [ordinal, device] : addressable_devices) {
        executor_streams.push_back(
            {device->executor(), device->compute_stream()});
      }
      return se::gpu::RocmDeviceAddressVmmAllocator::Create(
          platform, allocator_config.memory_fraction,
          allocator_config.gpu_system_memory_size, executor_streams,
          /*reclaim_exempt_memory_space=*/
          static_cast<int64_t>(gpu::MemorySpaceColor::kCollective));
#else
      return absl::UnimplementedError(
          "VMM allocator is only supported with CUDA or ROCm.");
#endif  // GOOGLE_CUDA
    }
  }

  // Add a separate collective allocator unless the default BFC allocator
  // already serves collective memory from its shared, spatially partitioned
  // pool.
  if (!shared_collective_pool) {
    for (const auto& ordinal_and_device : addressable_devices) {
      ABSL_ASSIGN_OR_RETURN(
          auto collective_bfc_allocator,
          CreateCollectiveBFCAllocator(
              ordinal_and_device.second->executor(),
              /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
              allocator_config.collective_memory_size));
      allocators.push_back(
          {std::move(collective_bfc_allocator),
           ordinal_and_device.second->compute_stream(),
           /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kCollective});
    }
  }

  for (const auto& ordinal_and_device : addressable_devices) {
    ABSL_ASSIGN_OR_RETURN(auto host_allocator,
                     GetGpuHostAllocator(ordinal_and_device.second->executor(),
                                         preallocate_host_memory));
    allocators.push_back(
        {std::move(host_allocator), ordinal_and_device.second->compute_stream(),
         /*memory_space=*/static_cast<int>(se::MemorySpace::kHost)});
  }

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
  if (debug_options.xla_gpu_temp_buffer_use_separate_color()) {
    // Add memory allocator to allocate memory buffers with persistent temp
    // memory space color.
    for (const auto& ordinal_and_device : addressable_devices) {
      ABSL_ASSIGN_OR_RETURN(auto async_allocator,
                       CreateCudaAsyncAllocator(*(ordinal_and_device.second),
                                                1.0, false, true, true, true));
      allocators.push_back(
          {std::move(async_allocator),
           ordinal_and_device.second->compute_stream(),
           /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kTempBuffer});
    }
  }
#endif
  return std::make_unique<se::MultiDeviceAdapter>(platform,
                                                  std::move(allocators));
}

// Name the devices and threads that launch work on them. Note: the launcher
// thread is only used if there are multiple devices driven by a single process.
void NameDeviceAndLauncherThread(const LocalTopologyProto& node,
                                 const DeviceProto& device_proto,
                                 WorkerThread* launcher_thread) {
  auto suffix = absl::StrFormat(
      ":#global=%d,local=%d,process=%d,partition=%d#",
      device_proto.global_device_id(), device_proto.local_device_ordinal(),
      node.process_id(), device_proto.partition_index());
  // Name the device.
  tsl::profiler::NameDevice(device_proto.local_device_ordinal(),
                            absl::StrCat("Xla", suffix));
  // Name the thread that launches work on this device. This is deferred
  // until after ExchangeTopologies has been called so the global device
  // id and partition index are known. These are not available when the thread
  // is created.
  launcher_thread->Schedule([name = absl::StrCat("XlaLauncher", suffix)] {
    tsl::profiler::NameCurrentThread(name);
  });
}

}  // namespace

// Creates allocator memory registration and adds the required suballocator
// visitors to `allocator_config`. Allocators that do not use suballocator
// visitors simply ignore them.
std::shared_ptr<gpu::AllocatorMemoryRegistration>
CreateAllocatorMemoryRegistration(GpuAllocatorConfig* allocator_config) {
  const DebugOptions& debug_options = GetDebugOptionsFromFlags();
  // TODO(b/530631424): Enable by default once bug is fixed.
  if (!debug_options.xla_gpu_enable_nccl_user_buffers_in_default_space()) {
    return nullptr;
  }
  // Automatic memory registration is only safe for preallocated BFC arenas.
  // If BFC grows later, ranks may not see a consistent set of registered
  // backing allocations, which can lead to undefined behavior or deadlocks.
  if (!allocator_config->preallocate) {
    return nullptr;
  }

  auto memory_registration =
      std::make_shared<gpu::AllocatorMemoryRegistration>();
  gpu::RegisterOnGpuCliqueCreatedCallback(
      memory_registration->CliqueCreatedCallback());
  allocator_config->sub_allocator_alloc_visitors.push_back(
      memory_registration->alloc_visitor());
  allocator_config->sub_allocator_free_visitors.push_back(
      memory_registration->free_visitor());

  return memory_registration;
}

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int process_id, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology = std::nullopt,
    std::optional<int> partition_index = std::nullopt,
    absl::Duration get_local_topology_timeout = absl::Minutes(2),
    absl::Duration get_global_topology_timeout = absl::Minutes(5)) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  LocalTopologyProto local_topology;
  local_topology.set_process_id(process_id);

  // If partition index is defined set it for local topology, otherwise it will
  // by assigned later based on the boot/fabric ids and network nodes.
  if (partition_index.has_value()) {
    local_topology.set_partition_index(*partition_index);
  }

  // Boot id is optional, we leave it empty if we can't get it at run time.
  absl::StatusOr<std::string> boot_id = GetBootIdString();
  if (boot_id.ok()) {
    local_topology.set_boot_id(*boot_id);
  } else {
    LOG(INFO) << "Failed to get boot id: " << boot_id.status();
  }

  // Network nodes also optional, they are needed for global device assignment
  // optimized for network locality.
  absl::StatusOr<std::vector<std::string>> network_nodes = GetNetworkNodes();
  if (network_nodes.ok()) {
    for (auto& network_node : *network_nodes) {
      *local_topology.add_network_nodes() = std::move(network_node);
    }
  } else {
    LOG(INFO) << "Failed to get network nodes: " << network_nodes.status();
  }

  std::optional<gpu::GpuTargetConfig> gpu_target_config;

  for (const auto& [ordinal, device] : local_device_states) {
    // We expect all devices on a host to have the same target config, so we
    // only need to get the target config for the first device.
    if (!gpu_target_config.has_value()) {
      gpu_target_config.emplace(device->executor());
    }
    const se::Platform* platform = device->executor()->GetPlatform();
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(device->local_hardware_id().value()));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(ordinal);
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
    auto compute_capability = MakeComputeCapabilityAttributeString(*desc);
    device_proto->set_compute_capability(compute_capability);
    device_proto->set_core_count(desc->core_count());
    device_proto->set_device_memory_bytes_limit(desc->device_memory_size());
    device_proto->set_shared_memory_per_block_optin(
        desc->shared_memory_per_block_optin());
    device_proto->set_numa_node(desc->numa_node());
    const se::DeviceInterconnectInfo& info = desc->device_interconnect_info();
    if (!info.cluster_uuid.empty() && !info.clique_id.empty()) {
      device_proto->set_fabric_uuid(
          absl::StrCat(info.cluster_uuid, "/", info.clique_id));
    }
  }

  if (!gpu_target_config.has_value()) {
    // A PjRtClient without any devices makes no sense, but we need to support
    // it for compatibility with Tensorflow. So we create an empty GPU target
    // config.
    stream_executor::GpuTargetConfigProto gpu_target_config_proto;
    gpu_target_config_proto.set_platform_name(platform_name);
    ABSL_ASSIGN_OR_RETURN(gpu_target_config,
                     gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));
  }

  GlobalTopologyProto global_topology;
  if (enable_mock_nccl) {
    TopologySizes sizes;
    if (mock_gpu_topology.has_value()) {
      ABSL_ASSIGN_OR_RETURN(sizes, TopologySizes::FromString(*mock_gpu_topology));
    } else {
      // If there is no topology spec, we assume that each node is a partition,
      // there is one process (host) on each partition and each host
      // has all the local devices.
      sizes.num_partitions = num_nodes;
      sizes.num_hosts_per_partition = 1;
      sizes.num_devices_per_host = local_topology.devices().size();
    }

    if (sizes.num_devices_per_host != local_topology.devices().size()) {
      return absl::InternalError(
          "The number of devices per host in 'mock_gpu_topology' "
          "must be the same as the number of devices in the local topology");
    }

    if (sizes.num_partitions * sizes.num_hosts_per_partition != num_nodes) {
      return absl::InternalError(
          "The number of hosts in 'mock_gpu_topology' "
          "must be the same as 'num_nodes'");
    }

    std::vector<LocalTopologyProto> local_topologies(num_nodes, local_topology);
    for (int i = 0; i < sizes.num_partitions; ++i) {
      for (int j = 0; j < sizes.num_hosts_per_partition; j++) {
        int process_id = i * sizes.num_hosts_per_partition + j;
        local_topologies[process_id].set_process_id(process_id);
        local_topologies[process_id].set_boot_id(absl::StrCat(i));
        local_topologies[process_id].set_partition_index(i);
      }
    }
    ABSL_ASSIGN_OR_RETURN(global_topology,
                     BuildGlobalTopology(absl::MakeSpan(local_topologies),
                                         /*assign_global_device_ids=*/true));
  } else {
    ABSL_RETURN_IF_ERROR(ExchangeTopologies(
        platform_name, process_id, num_nodes, get_local_topology_timeout,
        get_global_topology_timeout, kv_store.get(), local_topology,
        &global_topology, /*assign_global_device_ids=*/true));
  }

  auto device_interconnect_info_map =
      std::make_shared<se::DeviceInterconnectResource::InfoMap>();
  absl::btree_map<LocalDeviceId, GlobalDeviceId> gpu_device_ids;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process;
  int curr_partition_index = -1;
  int curr_process_index = -1;
  int curr_process_index_in_partition = 0;
  for (const LocalTopologyProto& node : global_topology.processes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      // The devices in the global topology are ordered by `partition_index`,
      // this is guaranteed by the `BuildGlobalTopology` function and the
      // `ExchangeTopologies` function.
      if (curr_partition_index != device_proto.partition_index()) {
        curr_partition_index = device_proto.partition_index();
        curr_process_index = node.process_id();
        curr_process_index_in_partition = 0;
      }
      if (curr_process_index != node.process_id()) {
        curr_process_index = node.process_id();
        curr_process_index_in_partition++;
      }

      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_process[global_device_id] = node.process_id();
      std::unique_ptr<LocalDeviceState> local_device;

      // Prepare DeviceInterconnectInfoMap.
      se::DeviceInterconnectInfo device_interconnect_info;
      {
        std::vector<std::string> parts =
            absl::StrSplit(device_proto.fabric_uuid(), '/');
        if (parts.size() == 2) {
          device_interconnect_info.cluster_uuid = parts[0];
          device_interconnect_info.clique_id = parts[1];
        }
      }
      device_interconnect_info_map->insert(
          {global_device_id.value(), std::move(device_interconnect_info)});

      if (node.process_id() == process_id) {
        auto it = local_device_states.find(device_proto.local_device_ordinal());
        TF_RET_CHECK(it != local_device_states.end())
            << device_proto.local_device_ordinal();
        TF_RET_CHECK(it->second != nullptr);
        local_device = std::move(it->second);

        // Attach Resource with shared DeviceInterconnectInfoMap to each
        // StreamExecutor.
        local_device->executor()
            ->GetOrCreateResource<se::DeviceInterconnectResource>(
                [device_interconnect_info_map] {
                  return std::make_unique<se::DeviceInterconnectResource>(
                      device_interconnect_info_map);
                });

        gpu_device_ids[LocalDeviceId(device_proto.local_device_ordinal())] =
            global_device_id;
        // Assign some descriptive names for profiling tools.
        NameDeviceAndLauncherThread(node, device_proto,
                                    local_device->execute_thread());
      }
      auto device = std::make_unique<StreamExecutorGpuDevice>(
          device_proto.global_device_id(), std::move(local_device),
          device_proto.name(), device_proto.vendor(),
          device_proto.compute_capability(), device_proto.core_count(),
          device_proto.device_memory_bytes_limit(),
          device_proto.shared_memory_per_block_optin(),
          device_proto.local_device_ordinal(), node.process_id(),
          curr_process_index_in_partition, device_proto.partition_index(),
          device_proto.numa_node(), device_proto.fabric_uuid());
      devices.push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device.second == nullptr);
  }

  VLOG(3) << absl::StreamFormat(
      "Set GPU device id map for process %d: %s", process_id,
      absl::StrJoin(gpu_device_ids, ",", absl::PairFormatter("->")));
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));

  auto* gpu_collectives = gpu_executable_run_options->collectives();
  if (gpu_collectives == nullptr) {
    gpu_collectives = gpu::GpuCollectives::Resolve(platform_name);
  }

  size_t num_processes = global_topology.processes().size();
  if (gpu_collectives->IsImplemented()) {
    ABSL_ASSIGN_OR_RETURN(
        auto clique_id_callback,
        gpu_collectives->InitializeTopology(
            {ProcessId(process_id), num_processes, local_device_states.size(),
             kv_store, device_to_process}));
    gpu_executable_run_options->set_clique_id_callback(
        std::move(clique_id_callback));
  }

  ABSL_ASSIGN_OR_RETURN(GpuTopologyProto gpu_topology,
                   BuildGpuTopology(global_topology, *gpu_target_config,
                                    cpu::TargetMachineOptions()));
  return std::make_pair(std::move(devices), gpu_topology);
}

StreamExecutorGpuDevice::StreamExecutorGpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor,
    std::string compute_capability, int core_count,
    int64_t device_memory_bytes_limit, int64_t shared_memory_per_block_optin,
    int local_device_id, int process_index, int process_index_in_partition,
    int partition_index, int numa_node, std::string fabric_uuid)
    : PjRtStreamExecutorDevice(
          id, std::move(local_device_state), local_device_id, process_index,
          process_index_in_partition, partition_index, std::move(device_kind)),
      device_vendor_(std::move(device_vendor)) {
  VLOG(1) << absl::StreamFormat(
      "Constructed StreamExecutor GPU device: compute_capability=%s "
      "core_count=%d device_memory_bytes_limit=%d shmem_per_block=%d "
      "local_device_id=%d process_index=%d "
      "process_index_in_partition=%d partition_index=%d numa_node=%d "
      "fabric_uuid=%s",
      compute_capability, core_count, device_memory_bytes_limit,
      shared_memory_per_block_optin, local_device_id, process_index,
      process_index_in_partition, partition_index, numa_node, fabric_uuid);

  StreamExecutorGpuTopologyDescription::SetupDeviceDescription(
      description(), device_vendor_, compute_capability, core_count,
      device_memory_bytes_limit,
      static_cast<int64_t>(shared_memory_per_block_optin), partition_index,
      fabric_uuid);
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes =
      description().Attributes();
  if (numa_node != tsl::port::kNUMANoAffinity) {
    attributes["numa_node"] = static_cast<int64_t>(numa_node);
  }
  SetAttributes(std::move(attributes));
}

absl::string_view StreamExecutorGpuDevice::device_vendor() const {
  return device_vendor_;
}

absl::StatusOr<tsl::AllocatorStats> StreamExecutorGpuDevice::GetAllocatorStats()
    const {
  if (!IsAddressable()) {
    return FailedPrecondition(
        "GetAllocatorStats() is allowed only for addressable devices");
  }

  auto* allocator_adapter = dynamic_cast<se::MultiDeviceAdapter*>(
      tensorflow::down_cast<PjRtStreamExecutorClient*>(client())->allocator());
  if (!allocator_adapter) {
    return Unimplemented(
        "GetAllocatorStats() is only implemented with MultiDeviceAdapter "
        "allocator");
  }

  ABSL_ASSIGN_OR_RETURN(auto allocator,
                   allocator_adapter->GetAllocator(local_device_id().value()));

  auto stats = allocator->GetStats();
  if (!stats.has_value()) {
    return Unimplemented(
        "GetAllocatorStats() is not supported by this allocator");
  }
  return *stats;
}

absl::Status StreamExecutorGpuDevice::ClearMemoryStats() {
  if (!IsAddressable()) {
    return absl::FailedPreconditionError(
        "ClearMemoryStats() is allowed only for addressable devices");
  }

  auto* allocator_adapter = dynamic_cast<se::MultiDeviceAdapter*>(
      tensorflow::down_cast<PjRtStreamExecutorClient*>(client())->allocator());
  if (!allocator_adapter) {
    return absl::UnimplementedError(
        "ClearMemoryStats() is only implemented with MultiDeviceAdapter "
        "allocator");
  }

  ABSL_ASSIGN_OR_RETURN(auto allocator,
                   allocator_adapter->GetAllocator(local_device_id().value()));

  // Call the ClearStats() method on the underlying tsl::Allocator
  // (BFCAllocator)
  if (allocator->ClearStats()) {
    return absl::OkStatus();
  }

  return absl::UnavailableError(
      "ClearStats not supported by the underlying allocator");
}

absl::Span<int const> StreamExecutorGpuDevice::coords() const {
  return description().coords();
}

absl::StatusOr<PjRtMemorySpace*> StreamExecutorGpuDevice::default_memory_space()
    const {
  return memory_space_by_kind_id(StreamExecutorGpuHbmMemorySpace::kKindId);
}

const int StreamExecutorGpuHbmMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(StreamExecutorGpuHbmMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

StreamExecutorGpuHbmMemorySpace::StreamExecutorGpuHbmMemorySpace(
    int id, PjRtDevice* device)
    : PjRtStreamExecutorMemorySpace(id, device, kKind, kKindId) {}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    const GpuClientOptions& options) {
#if TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::RocmName();
#elif TENSORFLOW_USE_SYCL
  auto pjrt_platform_name = xla::OneapiName();
#else   // TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  bool use_async_dispatch = false;
  if (options.use_async_dispatch.has_value()) {
    use_async_dispatch = *options.use_async_dispatch;
  } else if (const char* v = std::getenv("PJRT_GPU_ENABLE_ASYNC_DISPATCH")) {
    use_async_dispatch = absl::string_view(v) == "1";
  }

  ABSL_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  ABSL_ASSIGN_OR_RETURN(local_device_states,
                   BuildLocalDeviceStates(xla_client, use_async_dispatch,
                                          options.max_inflight_computations));
  EnablePeerAccess(xla_client->backend().stream_executors());

  GpuAllocatorConfig allocator_config = options.allocator_config;
  bool preallocate_device_memory = allocator_config.preallocate;
  auto memory_registration =
      CreateAllocatorMemoryRegistration(&allocator_config);

  bool preallocate_host_memory;
  ABSL_RETURN_IF_ERROR(tsl::ReadBoolFromEnvVar(
      "XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE", false, &preallocate_host_memory));

  ABSL_ASSIGN_OR_RETURN(auto allocator,
                   GetStreamExecutorGpuDeviceAllocator(
                       xla_client->platform(), std::move(allocator_config),
                       local_device_states, preallocate_host_memory));

  std::unique_ptr<HostMemoryAllocator> host_memory_allocator;
  if (options.host_memory_allocator_factory != nullptr) {
    if (preallocate_host_memory) {
      // Since `GetStreamExecutorGpuDeviceAllocator()` always creates a host
      // memory allocator, using both default host memory allocator and custom
      // allocator is wasteful if the default allocator is configured to
      // preallocate memory. We ask users to disable preallocation if they want
      // to use a custom host memory allocator instead.
      LOG(WARNING)
          << "Ignoring the custom host memory allocator factory given to PjRt "
             "GPU client creation since preallocation is also enabled; disable "
             "preallocation via XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE=false if "
             "you want to use a custom host allocator factory";
    } else {
      se::StreamExecutor* const stream_executor =
          local_device_states.begin()->second->compute_stream()->parent();
      HostMemoryAllocator::Options allocator_options;
      allocator_options.alignment = tsl::Allocator::kAllocatorAlignment;
      allocator_options.map_fn =
          [stream_executor](std::optional<LocalDeviceId> local_device_id,
                            void* data, size_t size) {
            bool success = stream_executor->HostMemoryRegister(data, size);
            if (!success) {
              return absl::InternalError(absl::StrFormat(
                  "Failed to register host memory at address: %ps", data));
            }
            return absl::OkStatus();
          };
      allocator_options.unmap_fn =
          [stream_executor](std::optional<LocalDeviceId> local_device_id,
                            void* data) {
            bool success = stream_executor->HostMemoryUnregister(data);
            if (!success) {
              return absl::InternalError(absl::StrFormat(
                  "Failed to unregister host memory at address: %ps", data));
            }
            return absl::OkStatus();
          };
      ABSL_ASSIGN_OR_RETURN(
          host_memory_allocator,
          options.host_memory_allocator_factory(std::move(allocator_options)));
    }
  }
  if (host_memory_allocator == nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        auto allocator,
        GetGpuHostAllocator(local_device_states.begin()->second->executor(),
                            preallocate_host_memory));
    host_memory_allocator = std::make_unique<BasicHostMemoryAllocator>(
        std::move(allocator), tsl::Allocator::kAllocatorAlignment);
  }

  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_collectives();
  }

  if (options.abort_collectives_on_failure) {
    gpu_run_options->set_execution_timeout_handler(
        [process_index = options.node_id,
         distributed_client = options.distributed_client](
            absl::string_view action, absl::Duration timeout) {
          absl::Status error = absl::DeadlineExceededError(
              absl::StrFormat("%s failed to finish in %v", action, timeout));

          if (absl::Status s =
                  gpu::AbortCollectivesOnTaskFailure(process_index, error);
              !s.ok()) {
            LOG(WARNING) << s;
          }

          if (distributed_client != nullptr) {
            absl::StatusOr<CoordinationServiceAgent*> agent =
                distributed_client->GetCoordinationServiceAgent();
            if (agent.ok()) {
              if (absl::Status s = (*agent)->ReportError(error); !s.ok()) {
                LOG(WARNING) << "Failed to report execution timeout to "
                                "coordination service: "
                             << s;
              }
            } else {
              LOG(WARNING) << "Failed to get coordination service agent: "
                           << agent.status();
            }
          } else {
            LOG(INFO) << "Skipping coordination service error report: "
                         "distributed client is not available.";
          }
        });
  }

  static const bool xla_gpu_require_exclusive_lock =
      xla::GetDebugOptionsFromFlags().xla_gpu_require_exclusive_lock();
  if (xla_gpu_require_exclusive_lock) {
    gpu_run_options->set_requires_exclusive_lock_on_gpu();
  }

  std::shared_ptr<KeyValueStoreInterface> kv_store = options.kv_store;
  if (options.enable_mock_nccl) {
    kv_store = std::make_shared<InMemoryKeyValueStore>();
  }
  TF_RET_CHECK(options.num_nodes == 1 || kv_store != nullptr);
  ABSL_ASSIGN_OR_RETURN(
      DeviceTopologyPair device_topology_pair,
      BuildDistributedDevices(
          pjrt_platform_name, std::move(local_device_states), options.node_id,
          options.num_nodes, gpu_run_options.get(), kv_store,
          options.enable_mock_nccl, options.mock_gpu_topology,
          options.partition_index));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<const GpuTopology> gpu_topology,
                   GpuTopology::FromProto(device_topology_pair.second));
  auto se_gpu_topology =
      CreateSEGpuTopology(pjrt_platform_name, std::move(gpu_topology),
                          GetFirstExecutor(device_topology_pair.first));
  auto raw_client = std::make_unique<StreamExecutorGpuRawClient>(
      std::move(allocator), xla_client, std::move(host_memory_allocator),
      options.should_stage_host_to_device_transfers,
      /*async_work_runner=*/nullptr,
      GetFirstExecutor(device_topology_pair.first), preallocate_device_memory,
      options.abort_collectives_on_failure, std::move(gpu_run_options));
  return std::make_unique<StreamExecutorGpuClient>(
      pjrt_platform_name, xla_client, std::move(device_topology_pair.first),
      options.node_id, std::move(raw_client), std::move(kv_store),
      options.abort_collectives_on_failure, std::move(se_gpu_topology),
      options.num_nodes, std::move(memory_registration));
}

static std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int process_id) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& desc =
        ordinal_and_device.second->executor()->GetDeviceDescription();
    const se::DeviceInterconnectInfo& info = desc.device_interconnect_info();
    int local_device_id = ordinal_and_device.second->local_device_id().value();
    auto device = std::make_unique<StreamExecutorGpuDevice>(
        /*id=*/ordinal_and_device.first,
        /*local_device_state=*/std::move(ordinal_and_device.second),
        /*device_kind=*/desc.name(),
        /*device_vendor=*/desc.device_vendor(),
        /*compute_capability=*/MakeComputeCapabilityAttributeString(desc),
        /*core_count=*/desc.core_count(),
        /*device_memory_bytes_limit=*/desc.device_memory_size(),
        /*shared_memory_per_block_optin=*/desc.shared_memory_per_block_optin(),
        /*local_device_id=*/local_device_id,
        /*process_index=*/process_id,
        /*process_index_in_partition=*/0,
        /*partition_index=*/0,
        /*numa_node=*/desc.numa_node(),
        /*fabric_uuid=*/absl::StrCat(info.cluster_uuid, "/", info.clique_id));
    devices.push_back(std::move(device));
  }
  return devices;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetSharedStreamExecutorGpuClient(
    const GpuClientOptions& options, LocalClient* local_client,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    std::unique_ptr<se::DeviceAddressAllocator> allocator,
    std::unique_ptr<HostMemoryAllocator> host_memory_allocator) {
  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
#if TENSORFLOW_USE_ROCM
  auto platform_name = RocmName();
#elif TENSORFLOW_USE_SYCL
  auto platform_name = SyclName();
#else   // TENSORFLOW_USE_ROCM
  auto platform_name = CudaName();
#endif  // TENSORFLOW_USE_ROCM
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> pjrt_devices;
  ABSL_ASSIGN_OR_RETURN(
      auto device_topology_pair,
      BuildDistributedDevices(platform_name, std::move(local_device_states),
                              options.node_id, options.num_nodes,
                              gpu_run_options.get(), options.kv_store,
                              /*enable_mock_nccl=*/false));

  VLOG(2) << "Distributed devices built with size=" << pjrt_devices.size();
  int i = 0;
  for (const auto& pjrt_device : pjrt_devices) {
    if (pjrt_device != nullptr) {
      VLOG(2) << "  pjrt_device " << i++ << ":"
              << pjrt_device->description().DebugString();
    } else {
      VLOG(2) << "  pjrt_device " << i++ << ":"
              << "nullptr";
    }
  }
  ABSL_ASSIGN_OR_RETURN(auto gpu_topology,
                   absl::StatusOr<std::shared_ptr<const GpuTopology>>(
                       GpuTopology::FromProto(device_topology_pair.second)));
  auto se_gpu_topology =
      CreateSEGpuTopology(platform_name, std::move(gpu_topology),
                          GetFirstExecutor(device_topology_pair.first));
  auto raw_client = std::make_unique<StreamExecutorGpuRawClient>(
      std::move(allocator), local_client, std::move(host_memory_allocator),
      /*should_stage_host_to_device_transfers=*/true,
      /*async_work_runner=*/nullptr,
      GetFirstExecutor(device_topology_pair.first),
      /*cache_fabric_handles=*/false,
      /*abort_collectives_on_failure=*/false, std::move(gpu_run_options));
  return std::make_unique<StreamExecutorGpuClient>(
      platform_name, local_client, std::move(device_topology_pair.first),
      /*process_index=*/options.node_id, std::move(raw_client),
      options.kv_store, /*abort_collectives_on_failure=*/false,
      /*topology=*/std::move(se_gpu_topology),
      /*num_nodes=*/options.num_nodes);
}

absl::Status ExchangeEmptyStreamExecutorGpuTopology(
    int process_id, int num_nodes,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout) {
#if TENSORFLOW_USE_ROCM
  auto platform_name = xla::RocmName();
#elif TENSORFLOW_USE_SYCL
  auto platform_name = xla::SyclName();
#else   // TENSORFLOW_USE_ROCM
  auto platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM
  LocalTopologyProto local_topology;
  local_topology.set_process_id(process_id);
  GlobalTopologyProto global_topology;
  return ExchangeTopologies(
      platform_name, process_id, num_nodes, get_local_topology_timeout,
      get_global_topology_timeout, kv_store.get(), local_topology,
      &global_topology, /*assign_global_device_ids=*/true);
}

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM) || \
    defined(TENSORFLOW_USE_SYCL)

static absl::StatusOr<PjRtStreamExecutorExecutionOutput> RunGpuAsync(
    LocalExecutable& exec, PjRtDevice* device,
    absl::Span<const PjRtRawBufferRef> flat_arguments,
    absl::Span<const PjRtRawBufferRef> results,
    ExecutableRunOptions run_options_inp, bool parameter_is_tupled_arguments) {
  std::vector<const Shape*> argument_shapes;
  if (exec.executable() != nullptr) {
    const auto& layout = exec.executable()->module().entry_computation_layout();
    argument_shapes.reserve(layout.parameter_count());
    for (int i = 0; i < layout.parameter_count(); ++i) {
      argument_shapes.push_back(&layout.parameter_shape(i));
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto options_and_stream,
                   exec.RunHelper(argument_shapes, run_options_inp));
  auto* gpu_exec =
      tensorflow::down_cast<xla::gpu::GpuExecutable*>(exec.executable());
  const ServiceExecutableRunOptions* run_options = &options_and_stream.first;
  se::DeviceAddressAllocator* const memory_allocator = run_options->allocator();

  se::StreamExecutor* executor = run_options->stream()->parent();

  // Use the `device_ordinal` from the `run_options` if it is provided. This is
  // the ordinal of the logical devices (e.g., virtual GPUs). If it is not
  // provided, the ordinals of the logical and physical devices are the same.
  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();

  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "[", device_ordinal, "] GpuExecutable::ExecuteAsyncOnStreamImpl(",
      gpu_exec->name(), ")"));

  // Attribute all device memory allocations to the gpu executable.
  tsl::ScopedAllocationTrace allocation_trace(
      "xla.execute", {{"executable", gpu_exec->name()}});

  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  auto activation = executor->Activate();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> gpu_lock(
      std::in_place_index_t<0>{}, &gpu::GetGpuMutex(executor));

  // Maybe update to a writer lock to get exclusive access to underlying GPU.
  if (auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
      gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    gpu_lock.emplace<1>(&gpu::GetGpuMutex(executor));
  }

  const gpu::GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    ABSL_ASSIGN_OR_RETURN(globals,
                     gpu_exec->ResolveConstantGlobals(run_options->stream()));
  }

  absl::Span<const BufferAllocation* const> allocations =
      gpu_exec->GetAllocations();

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<gpu::GpuExecutableBufferAllocator::ParameterBuffer> {
    int64_t param_no;
    if (parameter_is_tupled_arguments) {
      // TODO(parkers): Change compiler to not even pretend to read the tuple
      // index tables (also GPU shouldn't tuple ever).
      if (allocation.param_shape_index().empty()) {
        return gpu::GpuExecutableBufferAllocator::ParameterBuffer{
            se::DeviceAddressBase(), 0, /*allow_null_buffer=*/true};
      }
      param_no = allocation.param_shape_index()[0];
    } else {
      param_no = allocation.parameter_number();
    }
    return gpu::GpuExecutableBufferAllocator::ParameterBuffer{
        absl::down_cast<const xla::PjRtStreamExecutorRawBuffer*>(
            flat_arguments[param_no]
                ->down_cast<xla::PjRtStreamExecutorRawBuffer>())
            ->device_buffer()
            ->mem(),
        param_no};
  };

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<gpu::GpuExecutableBufferAllocator::ExecutionScope>
          allocation_scope,
      gpu_exec->buffer_allocator().CreateExecutionScope(
          run_options, memory_allocator, device_ordinal));

  ABSL_ASSIGN_OR_RETURN(xla::gpu::BufferAllocations buffer_allocations,
                   allocation_scope->GenerateBufferAllocations(
                       run_options, get_parameter_buffer, globals,
                       memory_allocator, device_ordinal));
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Buffer allocations: " << buffer_allocations.ToString();

  std::set<se::DeviceAddressBase> buffers_in_result;

  auto set_result = [&](const ShapeIndex& index, int i) -> absl::Status {
    const gpu::GpuExecutable::OutputInfo& output_info =
        gpu_exec->output_info().at(index);
    const BufferAllocation* allocation =
        allocations[output_info.allocation_index];
    se::DeviceAddressBase result_buffer;

    XLA_VLOG_DEVICE(4, device_ordinal)
        << "Looking at: allocation " << output_info.allocation_index
        << " @ index: " << index.ToString();

    auto buf = results[i]
                   .get()
                   ->down_cast<PjRtStreamExecutorRawBuffer>()
                   ->device_buffer();
    if (output_info.alias_config) {
      auto input = flat_arguments[parameter_is_tupled_arguments
                                      ? allocation->param_shape_index()[0]
                                      : allocation->parameter_number()]
                       ->down_cast<PjRtStreamExecutorRawBuffer>()
                       ->device_buffer();
      bool is_donated = input == buf;
      if (output_info.alias_config->must_alias() && !is_donated) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (is_donated) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        buffers_in_result.insert(input->mem());
        return absl::OkStatus();
      } else if (!ShapeUtil::GetSubshape(gpu_exec->result_shape(), index)
                      .IsTuple()) {
        ABSL_ASSIGN_OR_RETURN(
            result_buffer,
            allocation_scope->AllocateCopyProtectedOutputBuffer(
                run_options, buffer_allocations, index, *allocation,
                device_ordinal, memory_allocator, [&](absl::Status status) {
                  return ResourceExhausted(
                      "%s\n%s\n", status.message(),
                      gpu_exec->buffer_allocations_debug_summary());
                }));
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);
    }
    buffers_in_result.insert(result_buffer);

    RawSEDeviceMemory::ConstructDelayed(
        buf, result_buffer,
        tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
            ->local_device_state(),
        memory_allocator);
    return absl::OkStatus();
  };

  if (gpu_exec->result_shape().IsTuple()) {
    int tuple_count = gpu_exec->result_shape().tuple_shapes().size();
    for (int i = 0; i < tuple_count; ++i) {
      ABSL_RETURN_IF_ERROR(set_result({i}, i));
    }
  } else {
    ABSL_RETURN_IF_ERROR(set_result({}, 0));
  }

  absl::Status execute_status = allocation_scope->ExecuteWithBufferAllocations(
      buffer_allocations, device_ordinal,
      [&](const gpu::BufferAllocations& execution_buffers,
          std::optional<absl::Span<const BufferAllocation::Index>>
              persistent_alloc_indices) {
        return gpu_exec->ExecuteThunks(execution_buffers, run_options,
                                       persistent_alloc_indices);
      });
  absl::Status teardown_status = buffer_allocations.TearDown(
      buffers_in_result, gpu_exec->GetAllocations());

  ABSL_RETURN_IF_ERROR(execute_status);
  ABSL_RETURN_IF_ERROR(teardown_status);

  std::vector<tsl::AsyncValueRef<RawSEDeviceMemory>> to_be_released;

  // When the device uses compute-synchronized allocation, any foreign input
  // buffer must be explicitly kept alive until execution is complete because a
  // foreign buffer has its own `on_delete_callback` and may not follow the
  // compute synchronization model.
  if (tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
          ->local_device_state()
          ->allocation_model() == LocalDeviceState::kComputeSynchronized) {
    for (const auto& argument : flat_arguments) {
      const auto& device_buffer =
          absl::down_cast<const xla::PjRtStreamExecutorRawBuffer*>(
              argument->down_cast<xla::PjRtStreamExecutorRawBuffer>())
              ->device_buffer();
      if (dynamic_cast<ForeignRawSEDeviceMemory*>(&device_buffer.get()) !=
          nullptr) {
        to_be_released.push_back(device_buffer);
      }
    }
  }

  return PjRtStreamExecutorExecutionOutput({std::move(to_be_released), {}});
}

static bool register_gpu_run_async = []() {
  xla::RegisterRunAsyncHandler(std::type_index(typeid(xla::gpu::GpuExecutable)),
                               &RunGpuAsync);
  return true;
}();

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM || TENSORFLOW_USE_SYCL

absl::StatusOr<std::unique_ptr<PjRtRuntimeAbiVersion>>
StreamExecutorGpuClient::RuntimeAbiVersion() const {
  ABSL_ASSIGN_OR_RETURN(auto se_runtime_abi_version,
                   client_->platform()->GetRuntimeAbiVersion());
  return std::make_unique<StreamExecutorGpuPjRtRuntimeAbiVersion>(
      platform_id_, std::move(se_runtime_abi_version));
}

}  // namespace xla
