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
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
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
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/buffer_sequencing_event.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/host_to_device_transfer_manager.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/se_raw_buffer.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/traceme.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "xla/debug_options_flags.h"
#include "xla/pjrt/gpu/gpu_metrics.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/xla.pb.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/util.h"

namespace xla {

absl::Status RunCallbackOnStream(se::Stream* stream,
                                 AsyncWorkRunner* async_work_runner,
                                 absl::AnyInvocable<void() &&> callback) {
  return stream->DoHostCallbackWithStatus(
      [cb = std::move(callback), async_work_runner]() mutable {
        async_work_runner->Schedule(
            [cb_ptr = new absl::AnyInvocable<void() &&>(std::move(cb))]() {
              std::move (*cb_ptr)();
              delete cb_ptr;
            });
        return absl::OkStatus();
      });
}

static std::optional<stream_executor::GpuTargetConfigProto>
GetTargetConfigForDevices(absl::Span<PjRtDevice* const> devices) {
  // Temporary ability to disable TargetConfig via env var until
  // internal tests can be fixed.
  const char* disable_target_config_str =
      std::getenv("PJRT_GPU_SE_DISABLE_TARGET_CONFIG");
  int disable_target_config = 0;
  if (disable_target_config_str &&
      absl::SimpleAtoi(disable_target_config_str, &disable_target_config)) {
    if (disable_target_config == 1) {
      return std::nullopt;
    }
  }
  for (const PjRtDevice* device : devices) {
    LocalDeviceState* local_device_state =
        tensorflow::down_cast<const PjRtStreamExecutorDevice*>(device)
            ->local_device_state();
    if (local_device_state != nullptr) {
      return xla::gpu::GpuTargetConfig(local_device_state->executor())
          .ToProto();
    }
  }
  return std::nullopt;
}

static absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    absl::Span<PjRtDevice* const> devices) {
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attrs;
  auto target_config = GetTargetConfigForDevices(devices);
  if (target_config.has_value()) {
    std::string attr;
    if (tsl::protobuf::TextFormat::PrintToString(*target_config, &attr)) {
      attrs["target_config"] = std::move(attr);
    }
  }
  return attrs;
}

StreamExecutorGpuClient::StreamExecutorGpuClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::unique_ptr<se::DeviceAddressAllocator> allocator,
    std::unique_ptr<HostMemoryAllocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    bool abort_collectives_on_failure,
    std::shared_ptr<const GpuTopology> gpu_topology,
    std::optional<int> num_nodes)
    : xla::PjRtStreamExecutorClient(
          platform_name, client, std::move(devices), process_index,
          /*memory_spaces=*/{},  // Initialized below.
          std::move(allocator), std::move(host_memory_allocator),
          should_stage_host_to_device_transfers, std::move(gpu_run_options)),
      num_nodes_(num_nodes),
      abort_collectives_on_failure_(abort_collectives_on_failure),
      kv_store_(std::move(kv_store)) {
  if (gpu_topology != nullptr) {
    topology_.emplace(tsl::Fingerprint64(platform_name), platform_name,
                      std::move(gpu_topology),
                      GetAttrsForDevices(addressable_devices()),
                      GetTargetConfigForDevices(addressable_devices()));
  }
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
    absl::Span<tensorflow::CoordinatedTaskStateInfo> infos) {
  if (!abort_collectives_on_failure_) {
    return;
  }
  absl::Status s = ::xla::gpu::UpdateGlobalProcessInfo(infos);
  if (!s.ok()) {
    LOG(WARNING) << s;
  }
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
StreamExecutorGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  return xla::CreateAsyncHostToDeviceTransferManager(
      shape_specs, std::move(device_layouts), memory_space);
}

absl::flat_hash_map<GlobalDeviceId, IncarnationId>
StreamExecutorGpuClient::GetLatestIncarnations(const ExecuteOptions& options) {
  // Map every device to its incarnation.
  absl::flat_hash_map<GlobalDeviceId, IncarnationId> device_incarnations;
  for (const PjRtDevice* device : devices()) {
    int task_id = device->process_index();
    GlobalDeviceId device_id(device->global_device_id().value());

    auto it = options.incarnations.find(task_id);
    if (it == options.incarnations.end()) {
      // The task might be dead.
      LOG(WARNING) << "Incarnation for task " << task_id << " not found";
      continue;
    }
    device_incarnations[device_id] = it->second;
  }
  return device_incarnations;
}

gpu::GpuExecutableRunOptions* StreamExecutorGpuClient::gpu_run_options(
    const ExecuteOptions& options) {
  if (!options.incarnations.empty()) {
    absl::flat_hash_map<GlobalDeviceId, IncarnationId> incarnations =
        GetLatestIncarnations(options);
    gpu_run_options_->set_incarnations(std::move(incarnations));
  }
  return gpu_run_options_.get();
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
  TF_RETURN_IF_ERROR(PjRtStreamExecutorClient::UpdateCompileOptionsInternal(
      options, returned_extras, lookup_addressable_devices));
  if (topology_) {
    options->executable_build_options.set_slice_size(
        topology_->gpu_topology().slice_size());
  }
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

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> communicators,
                      gpu_collectives->CreateCommunicators(
                          clique_key, clique_ids, ranks, config));
  CHECK_EQ(communicators.size(), 1);

  return std::move(communicators[0]);
}

// Helper structs / classes for second cross-host transfers API.
using AcquiredCliqueAndCommunicator =
    std::pair<std::shared_ptr<gpu::LockableGpuClique::Lock>,
              gpu::GpuCommunicator*>;

class PreparedSend {
 public:
  StreamExecutorGpuClient* client_;
  gpu::GpuCliqueKey clique_key_;
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer_;
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events_;
  tsl::RCReference<PjRtStreamExecutorDeviceEvent> usage_event_;
  AcquiredCliqueAndCommunicator clique_and_communicator_;
  std::shared_ptr<Promise<>> promise_;

  PreparedSend(StreamExecutorGpuClient* client, gpu::GpuCliqueKey clique_key,
               tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
               std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events,
               tsl::RCReference<PjRtStreamExecutorDeviceEvent> usage_event,
               AcquiredCliqueAndCommunicator clique_and_communicator,
               std::shared_ptr<Promise<>> promise)
      : client_(client),
        clique_key_(std::move(clique_key)),
        raw_buffer_(std::move(raw_buffer)),
        definition_events_(std::move(definition_events)),
        usage_event_(std::move(usage_event)),
        clique_and_communicator_(std::move(clique_and_communicator)),
        promise_(std::move(promise)) {}

  PreparedSend(PreparedSend&&) = default;
  PreparedSend& operator=(PreparedSend&&) = default;

  ~PreparedSend() {
    if (!usage_event_ || usage_event_->event()->IsDefined()) {
      return;
    }
    LOG(WARNING) << "PreparedSend destroyed with unfulfilled usage_event";
    client_->SetEventAsError(
        usage_event_->event(),
        absl::InternalError("PreparedSend destroyed without fulfilling "
                            "usage_event"));
  }
};

class PreparedReceive {
 public:
  StreamExecutorGpuClient* client_;
  gpu::GpuCliqueKey clique_key_;
  std::unique_ptr<PjRtBuffer> buffer_;
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer_;
  tsl::RCReference<PjRtStreamExecutorDeviceEvent> definition_event_;
  AcquiredCliqueAndCommunicator clique_and_communicator_;

  PreparedReceive(
      StreamExecutorGpuClient* client, gpu::GpuCliqueKey clique_key,
      std::unique_ptr<PjRtBuffer> buffer,
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      tsl::RCReference<PjRtStreamExecutorDeviceEvent> definition_event,
      AcquiredCliqueAndCommunicator clique_and_communicator)
      : client_(client),
        clique_key_(std::move(clique_key)),
        buffer_(std::move(buffer)),
        raw_buffer_(std::move(raw_buffer)),
        definition_event_(std::move(definition_event)),
        clique_and_communicator_(std::move(clique_and_communicator)) {}

  PreparedReceive(PreparedReceive&&) = default;
  PreparedReceive& operator=(PreparedReceive&&) = default;

  ~PreparedReceive() {
    if (!definition_event_ || definition_event_->event()->IsDefined()) {
      return;
    }
    LOG(WARNING)
        << "PreparedReceive destroyed with unfulfilled definition_event";
    client_->SetEventAsError(
        definition_event_->event(),
        absl::InternalError("PreparedReceive destroyed without fulfilling "
                            "definition_event"));
  }
};

// Acquire the GPU clique and communicator for a given clique key.
absl::StatusOr<AcquiredCliqueAndCommunicator> AcquireCliqueAndCommunicator(
    StreamExecutorGpuClient* client, gpu::GpuCollectives* gpu_collectives,
    const gpu::GpuCliqueKey& clique_key,
    gpu::AcquiredCliquesMap& acquired_cliques_map, RankId rank_id,
    se::Stream* stream) {
  // Get the clique ID callback.
  const ExecuteOptions dummy_execute_options;
  gpu::GpuExecutableRunOptions* dummy_gpu_run_options =
      client->gpu_run_options(dummy_execute_options);
  gpu::CliqueIdCallback clique_id_callback =
      dummy_gpu_run_options->clique_id_callback();

  // Acquire the GPU clique for this receive.
  if (!acquired_cliques_map.contains(clique_key)) {
    TF_ASSIGN_OR_RETURN(
        acquired_cliques_map[clique_key],
        AcquireGpuClique(gpu_collectives,
                         /*device=*/stream->parent(), RunId(0), clique_key,
                         clique_id_callback, rank_id, acquired_cliques_map,
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
      tsl::down_cast<gpu::GpuCommunicator*>(*maybe_communicator)};
}

// Create a `PreparedSend` object bundling together state needed to perform a
// send.
absl::StatusOr<PreparedSend> PrepareSend(
    StreamExecutorGpuClient* client, gpu::GpuCollectives* gpu_collectives,
    se::Stream* stream, PjRtBuffer* buffer,
    PjRtGlobalDeviceId dst_global_device_id, CrossHostTransferKey transfer_key,
    gpu::AcquiredCliquesMap& acquired_cliques_map,
    std::shared_ptr<Promise<>> promise,
    tsl::RCReference<PjRtStreamExecutorDeviceEvent> usage_event) {
  // Form the GPU clique key.
  // TODO(asrao, mwhittaker): Supply correct incarnations when creating the
  // clique key.
  gpu::GpuCliqueKey clique_key = gpu::GpuCliqueKey(
      /*devices=*/{GlobalDeviceId(buffer->device()->global_device_id().value()),
                   GlobalDeviceId(dst_global_device_id.value())},
      /*num_local_participants=*/1);

  // Get the clique and communicator for the send.
  TF_ASSIGN_OR_RETURN(
      AcquiredCliqueAndCommunicator clique_and_communicator,
      AcquireCliqueAndCommunicator(client, gpu_collectives, clique_key,
                                   acquired_cliques_map, RankId(0), stream));

  // Acquire a hold on this buffer. The hold is held as long as usage_event is
  // not fulfilled; this behavior is achieved by registering a 'dummy' closure
  // capturing raw_buffer that executes after usage_event is fulfilled.
  // definition_events can be used to track when the buffer data is ready.
  tsl::RCReference<CommonPjRtRawBuffer> raw_buffer;
  std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events;

  TF_RETURN_IF_ERROR(
      tensorflow::down_cast<CommonPjRtBufferImpl*>(buffer)
          ->AcquireScopedRawBuffer(
              [&](tsl::RCReference<CommonPjRtRawBuffer> buf_raw_buffer,
                  std::vector<tsl::RCReference<tsl::AsyncValue>>
                      buf_definition_events) mutable
                  -> absl::StatusOr<tsl::RCReference<PjRtDeviceEvent>> {
                raw_buffer = std::move(buf_raw_buffer);
                usage_event->AndThen([raw_buffer]() {});
                definition_events = std::move(buf_definition_events);
                return usage_event;
              },
              "PrepareSend"));

  // Return the result.
  return PreparedSend(client, std::move(clique_key), std::move(raw_buffer),
                      std::move(definition_events), std::move(usage_event),
                      std::move(clique_and_communicator), std::move(promise));
}

// Create a `PreparedReceive` object bundling together state needed to perform a
// receive.
absl::StatusOr<PreparedReceive> PrepareReceive(
    StreamExecutorGpuClient* client, gpu::GpuCollectives* gpu_collectives,
    se::Stream* stream, PjRtDevice* device, PjRtMemorySpace* memory_space,
    PjRtGlobalDeviceId src_global_device_id, CrossHostTransferKey transfer_key,
    Shape shape, gpu::AcquiredCliquesMap& acquired_cliques_map,
    tsl::RCReference<PjRtStreamExecutorDeviceEvent> definition_event) {
  // Form the GPU clique key.
  // TODO(asrao, mwhittaker): Supply correct incarnations when creating the
  // clique key.
  gpu::GpuCliqueKey clique_key = gpu::GpuCliqueKey(
      /*devices=*/{GlobalDeviceId(src_global_device_id.value()),
                   GlobalDeviceId(device->global_device_id().value())},
      /*num_local_participants=*/1);

  // Get the clique and communicator for the receive.
  TF_ASSIGN_OR_RETURN(
      AcquiredCliqueAndCommunicator clique_and_communicator,
      AcquireCliqueAndCommunicator(client, gpu_collectives, clique_key,
                                   acquired_cliques_map, RankId(1), stream));

  // Allocate an uninitialized buffer. The buffer will be populated with data
  // received from the sending process.
  TF_ASSIGN_OR_RETURN(
      Shape on_device_shape,
      client->MakeDefaultShapeForMemorySpace(
          memory_space, shape, shape.has_layout() ? &shape.layout() : nullptr));
  TF_ASSIGN_OR_RETURN(
      size_t on_device_bytes_count,
      client->GetOnDeviceBytesCount(memory_space, on_device_shape));
  TF_ASSIGN_OR_RETURN(
      tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
      client->AllocateRawBuffer(memory_space, on_device_bytes_count,
                                /*retry_on_oom=*/true,
                                /*allocate_after=*/{}));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer,
                      client->DefineBuffer(on_device_shape, memory_space,
                                           raw_buffer, {definition_event}));
  definition_event->AndThen([raw_buffer]() {});

  return PreparedReceive(client, std::move(clique_key), std::move(buffer),
                         std::move(raw_buffer), std::move(definition_event),
                         std::move(clique_and_communicator));
}

absl::flat_hash_map<gpu::GpuCliqueKey, std::vector<PreparedSend>>
GroupSendsByCliqueKey(std::vector<PreparedSend>&& prepared_sends) {
  absl::flat_hash_map<gpu::GpuCliqueKey, std::vector<PreparedSend>> grouped;
  grouped.reserve(prepared_sends.size());
  for (auto&& prepared_send : prepared_sends) {
    grouped[prepared_send.clique_key_].push_back(std::move(prepared_send));
  }
  return grouped;
}

absl::flat_hash_map<gpu::GpuCliqueKey, std::vector<PreparedReceive>>
GroupReceivesByCliqueKey(std::vector<PreparedReceive>&& prepared_receives) {
  absl::flat_hash_map<gpu::GpuCliqueKey, std::vector<PreparedReceive>> grouped;
  grouped.reserve(prepared_receives.size());
  for (auto&& prepared_receive : prepared_receives) {
    grouped[prepared_receive.clique_key_].push_back(
        std::move(prepared_receive));
  }
  return grouped;
}

absl::Status FulfillDeviceEvent(
    PjRtStreamExecutorClient* client, LocalDeviceState* local_device_state,
    se::Stream* stream,
    tsl::RCReference<PjRtStreamExecutorDeviceEvent> device_event,
    const absl::Status& status) {
  if (!status.ok()) {
    client->SetEventAsError(device_event->event(), status);
    return absl::OkStatus();
  }
  absl::Status s = client->AllocateAndRecordEvent(device_event->event(),
                                                  local_device_state, stream);
  if (!s.ok()) {
    client->SetEventAsError(device_event->event(), s);
  }
  return s;
}

void FulfillPromises(std::vector<std::shared_ptr<Promise<>>>& promises,
                     absl::Status status) {
  for (std::shared_ptr<Promise<>>& promise : promises) {
    promise->Set(status);
  }
}
}  // namespace

// Send functionality for second cross-host transfers API.
absl::StatusOr<std::vector<Future<>>>
StreamExecutorGpuClient::CrossHostSendBuffers(
    absl::Span<PjRtBuffer* const> buffers,
    absl::Span<const PjRtGlobalDeviceId> dst_global_device_ids,
    std::vector<CrossHostTransferKey> transfer_keys) {
  // Validate arguments.
  if (dst_global_device_ids.size() != buffers.size() ||
      transfer_keys.size() != buffers.size()) {
    return InvalidArgument(
        "CrossHostSendBuffers: buffers, "
        "dst_global_device_ids, and transfer_keys "
        "must have the same length, but got %d, %d, and %d.",
        buffers.size(), dst_global_device_ids.size(), transfer_keys.size());
  }
  for (int i = 0; i < buffers.size(); ++i) {
    // Each transfer must be between an addressable and a non-addressable
    // device. If both devices are addressable, then both a data transfer and a
    // 'normal' XLA SPMD executable may try to acquire the same GPU clique,
    // causing issues.
    PjRtDevice* src_device = buffers[i]->device();
    if (!src_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostSendBuffers: buffer %d is on non-addressable device with "
          "global device id %d.",
          i, src_device->global_device_id().value());
    }
    TF_ASSIGN_OR_RETURN(PjRtDevice * dst_device,
                        LookupDevice(dst_global_device_ids[i]));
    if (dst_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostSendBuffers: destination device for buffer %d is "
          "addressable (global device id %d), but cross-host transfers must "
          "be between an addressable and a non-addressable device.",
          i, dst_global_device_ids[i].value());
    }
  }

  // Create futures and promises.
  std::vector<Future<>> futures;
  std::vector<std::shared_ptr<Promise<>>> promises;
  futures.reserve(buffers.size());
  promises.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    auto [promise, future] = MakePromise<>();
    futures.push_back(std::move(future));
    promises.push_back(std::move(promise).ToShared());
  }

  // Group the sends by local device.
  absl::flat_hash_map<PjRtDevice*, std::vector<int>> sends_by_device;
  for (int i = 0; i < buffers.size(); ++i) {
    sends_by_device[buffers[i]->device()].push_back(i);
  }

  // Execute sends for each local device.
  for (auto& [device, send_idxs] : sends_by_device) {
    // Execute sends.
    std::vector<PjRtBuffer*> curr_buffers;
    std::vector<PjRtGlobalDeviceId> curr_dst_ids;
    std::vector<CrossHostTransferKey> curr_transfer_keys;
    std::vector<std::shared_ptr<Promise<>>> curr_promises;
    for (int idx : send_idxs) {
      curr_buffers.push_back(buffers[idx]);
      curr_dst_ids.push_back(dst_global_device_ids[idx]);
      curr_transfer_keys.push_back(transfer_keys[idx]);
      curr_promises.push_back(promises[idx]);
    }

    ScheduleSendsOnLocalDevice(
        device, std::move(curr_buffers), std::move(curr_dst_ids),
        std::move(curr_transfer_keys), std::move(curr_promises));
  }

  return futures;
}

void StreamExecutorGpuClient::ScheduleSendsOnLocalDevice(
    PjRtDevice* device, std::vector<PjRtBuffer*> buffers,
    const std::vector<PjRtGlobalDeviceId> dst_global_device_ids,
    const std::vector<CrossHostTransferKey> transfer_keys,
    std::vector<std::shared_ptr<Promise<>>> promises) {
  // Get the local device state, transfer stream, and prepare the send
  // buffers. We associate the group of sends with a single usage_event.
  LocalDeviceState* local_device_state;
  se::Stream* stream;
  std::vector<PreparedSend> prepared_sends;
  prepared_sends.reserve(buffers.size());
  tsl::RCReference<PjRtStreamExecutorDeviceEvent> usage_event;

  auto setup_sends = [&]() -> absl::Status {
    TF_ASSIGN_OR_RETURN(local_device_state, GetLocalDeviceState(device));
    stream = local_device_state->GetDeviceToDeviceStream();
    gpu::GpuCollectives* gpu_collectives =
        gpu::GpuCollectives::Default(stream->parent()->GetPlatform()->Name());
    usage_event = tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(
        BufferSequencingEvent::Create(this->async_work_runner()));

    gpu::AcquiredCliquesMap acquired_cliques_map;
    for (int i = 0; i < buffers.size(); ++i) {
      absl::StatusOr<PreparedSend> prepared_send = PrepareSend(
          this, gpu_collectives, stream, buffers[i], dst_global_device_ids[i],
          transfer_keys[i], acquired_cliques_map, promises[i], usage_event);

      if (!prepared_send.ok()) {
        SetEventAsError(usage_event->event(), prepared_send.status());
        return prepared_send.status();
      }

      prepared_sends.push_back(*std::move(prepared_send));
    }

    return absl::OkStatus();
  };

  if (absl::Status status = setup_sends(); !status.ok()) {
    FulfillPromises(promises, status);
    return;
  }

  // Form the closure called for each group of sends.
  auto launch_send_group = [](gpu::GpuCommunicator* gpu_communicator,
                              absl::Span<PreparedSend> prepared_sends,
                              se::Stream* stream) -> absl::Status {
    for (PreparedSend& prepared_send : prepared_sends) {
      // Wait until the buffer we want to send is fully materialized.
      for (const auto& event : prepared_send.definition_events_) {
        if (event->IsType<BufferSequencingEvent>()) {
          tsl::AsyncValueRef<BufferSequencingEvent> event_ref(event);
          event_ref->WaitForEventOnStream(stream);
        } else {
          tsl::BlockUntilReady(event.get());
        }
        if (auto* status = event->GetErrorIfPresent(); status != nullptr) {
          return *status;
        }
      }
      // Launch the send.
      auto mem = tensorflow::down_cast<PjRtStreamExecutorRawBuffer*>(
                     prepared_send.raw_buffer_.get())
                     ->device_buffer();
      TF_RETURN_IF_ERROR(gpu_communicator->LaunchSend(
          /*send_buffer=*/mem->mem(),
          /*dtype=*/xla::PrimitiveType::U8,
          /*count=*/mem->mem().size(),
          /*peer=*/RankId(1),
          /*executor=*/gpu::GpuCollectives::On(*stream)));
    }
    return absl::OkStatus();
  };

  // Form the closure to schedule on the device's execute thread.
  auto execute_sends_fn = [this, local_device_state, stream,
                           promises = std::move(promises),
                           prepared_sends = std::move(prepared_sends),
                           launch_send_group = std::move(launch_send_group),
                           usage_event = std::move(usage_event)]() mutable {
    // Group transfers by GPU clique.
    absl::flat_hash_map<gpu::GpuCliqueKey, std::vector<PreparedSend>>
        grouped_sends = GroupSendsByCliqueKey(std::move(prepared_sends));

    // Transfers for a particular clique are executed as a group. This vector
    // holds group futures for each clique_key in grouped_sends.
    std::vector<Future<>> group_futures;
    group_futures.reserve(grouped_sends.size());

    for (auto& [clique_key, curr_sends] : grouped_sends) {
      // Get the communicator on which we will execute this group of
      // transfers. We assume each clique key is associated with a unique
      // communicator, so we just take the communicator of the first
      // transfer_idx of this clique key.
      gpu::GpuCommunicator* gpu_communicator =
          curr_sends[0].clique_and_communicator_.second;

      // Launch the group of transfers.
      group_futures.push_back(gpu_communicator->GroupExecute(
          [&launch_send_group, &curr_sends = curr_sends,
           stream](gpu::GpuCommunicator* gpu_comm) -> absl::Status {
            return launch_send_group(gpu_comm, absl::MakeSpan(curr_sends),
                                     stream);
          }));
    }

    // On a separate thread pool, await group futures and fulfill buffer
    // sequencing events and promises.
    Future<> all_sends_future = JoinFutures(group_futures);

    all_sends_future.OnReady(
        this->async_work_runner()->AsExecutor(),
        [this, local_device_state, stream, promises = std::move(promises),
         usage_event, grouped_sends = std::move(grouped_sends)](
            const absl::Status& status) mutable {
          // Add usage_event onto the stream.
          absl::Status fulfill_status = FulfillDeviceEvent(
              this, local_device_state, stream, usage_event, status);

          // Fail promises early if there was an issue.
          if (!status.ok() || !fulfill_status.ok()) {
            FulfillPromises(promises, status);
            return;
          }

          // Asynchronously fulfill promises via a host callback, failing them
          // early if there is an issue registering the callback.
          absl::Status callback_status = RunCallbackOnStream(
              stream, this->async_work_runner(), [promises]() mutable {
                FulfillPromises(promises, absl::OkStatus());
              });

          if (!callback_status.ok()) {
            FulfillPromises(promises, callback_status);
          }
        });
  };

  // Schedule transfers on the execute thread.
  local_device_state->execute_thread()->Schedule(std::move(execute_sends_fn));
}

// Prepare a receive buffer on a given device for receiving data as part of a
// cross-host transfer. This function (and the PrepareReceiveBufferResult helper
// struct) is used by the original cross-host transfers API
// (MakeCrossHostReceiveBuffers).
absl::StatusOr<StreamExecutorGpuClient::PrepareReceiveBufferResult>
StreamExecutorGpuClient::PrepareReceiveBuffer(PjRtDevice* device, Shape shape) {
  TF_ASSIGN_OR_RETURN(auto* memory_space, device->default_memory_space());
  TF_ASSIGN_OR_RETURN(
      Shape on_device_shape,
      MakeDefaultShapeForMemorySpace(
          memory_space, shape, shape.has_layout() ? &shape.layout() : nullptr));
  TF_ASSIGN_OR_RETURN(size_t on_device_bytes_count,
                      GetOnDeviceBytesCount(memory_space, on_device_shape));

  // Allocate an uninitialized buffer. The buffer will be populated with data
  // received from the sending process.
  TF_ASSIGN_OR_RETURN(tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
                      AllocateRawBuffer(memory_space, on_device_bytes_count,
                                        /*retry_on_oom=*/true,
                                        /*allocate_after=*/{}));
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());

  se::Stream* stream = local_device->GetDeviceToDeviceStream();

  BufferSequencingEventRef definition_event =
      BufferSequencingEvent::Create(this->async_work_runner());
  TF_ASSIGN_OR_RETURN(
      auto buffer,
      DefineBuffer(
          on_device_shape, memory_space, raw_buffer,
          {tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(definition_event)}));

  return PrepareReceiveBufferResult{std::move(buffer), std::move(raw_buffer),
                                    local_device, stream,
                                    std::move(definition_event)};
}

// Receive functionality for second cross-host transfers API.
absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
StreamExecutorGpuClient::CrossHostReceiveBuffers(
    xla::PjRtDevice* device, absl::Span<const xla::Shape> shapes,
    absl::Span<const PjRtGlobalDeviceId> src_global_device_ids,
    std::vector<CrossHostTransferKey> transfer_keys) {
  // Validate arguments.
  if (shapes.empty()) {
    return InvalidArgument("shapes parameter empty in CrossHostReceiveBuffers");
  }
  if (src_global_device_ids.size() != shapes.size() ||
      transfer_keys.size() != shapes.size()) {
    return InvalidArgument(
        "CrossHostReceiveBuffers: shapes, src_global_device_ids, and "
        "transfer_keys must have the same length, but got %d, %d, and %d.",
        shapes.size(), src_global_device_ids.size(), transfer_keys.size());
  }
  // Each transfer must be between an addressable and a non-addressable
  // device. If both devices are addressable, then both a data transfer and a
  // 'normal' XLA SPMD executable may try to acquire the same GPU clique,
  // causing issues.
  if (!device->IsAddressable()) {
    return InvalidArgument(
        "CrossHostReceiveBuffers: destination device is non-addressable "
        "(global device id %d), but cross-host transfers must be between an "
        "addressable and a non-addressable device.",
        device->global_device_id().value());
  }
  for (int i = 0; i < src_global_device_ids.size(); ++i) {
    TF_ASSIGN_OR_RETURN(PjRtDevice * src_device,
                        LookupDevice(src_global_device_ids[i]));
    if (src_device->IsAddressable()) {
      return InvalidArgument(
          "CrossHostReceiveBuffers: source device for buffer %d is "
          "addressable (global device id %d), but cross-host transfers must "
          "be between an addressable and a non-addressable device.",
          i, src_global_device_ids[i].value());
    }
  }

  // Get the local device state, transfer stream, and prepare the receive
  // buffers. We associate the group of receives with a single definition_event.
  LocalDeviceState* local_device_state;
  se::Stream* stream;
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(shapes.size());
  std::vector<PreparedReceive> prepared_receives;
  prepared_receives.reserve(shapes.size());
  tsl::RCReference<PjRtStreamExecutorDeviceEvent> definition_event;

  auto setup_receives = [&]() -> absl::Status {
    TF_ASSIGN_OR_RETURN(local_device_state, GetLocalDeviceState(device));
    stream = local_device_state->GetDeviceToDeviceStream();
    TF_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                        device->default_memory_space());
    gpu::GpuCollectives* gpu_collectives =
        gpu::GpuCollectives::Default(stream->parent()->GetPlatform()->Name());
    definition_event = tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(
        BufferSequencingEvent::Create(this->async_work_runner()));

    gpu::AcquiredCliquesMap acquired_cliques_map;
    for (int i = 0; i < shapes.size(); ++i) {
      absl::StatusOr<PreparedReceive> prepared_receive =
          PrepareReceive(this, gpu_collectives, stream, device, memory_space,
                         src_global_device_ids[i], transfer_keys[i], shapes[i],
                         acquired_cliques_map, definition_event);

      if (!prepared_receive.ok()) {
        SetEventAsError(definition_event->event(), prepared_receive.status());
        return prepared_receive.status();
      }

      buffers.push_back(std::move(prepared_receive->buffer_));
      prepared_receives.push_back(*std::move(prepared_receive));
    }

    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(setup_receives());

  // Form the closure called for each group of receives.
  auto launch_receive_group = [this](
                                  gpu::GpuCommunicator* gpu_communicator,
                                  absl::Span<PreparedReceive> prepared_receives,
                                  se::Stream* stream) -> absl::Status {
    for (PreparedReceive& prepared_receive : prepared_receives) {
      // Wait until the receive buffer is allocated.
      WaitForAllocation(stream, *prepared_receive.raw_buffer_);

      // Launch the receive.
      auto mem = tensorflow::down_cast<PjRtStreamExecutorRawBuffer*>(
                     prepared_receive.raw_buffer_.get())
                     ->device_buffer();
      TF_RETURN_IF_ERROR(gpu_communicator->LaunchRecv(
          /*recv_buffer=*/mem->mem(),
          /*dtype=*/xla::PrimitiveType::U8,
          /*count=*/mem->mem().size(),
          /*peer=*/RankId(0),
          /*executor=*/gpu::GpuCollectives::On(*stream)));
    }
    return absl::OkStatus();
  };

  // Form the closure to schedule on the device's execute thread.
  auto execute_receives_fn =
      [this, local_device_state, stream,
       prepared_receives = std::move(prepared_receives),
       launch_receive_group = std::move(launch_receive_group),
       definition_event = std::move(definition_event)]() mutable {
        // Group transfers by GPU clique.
        absl::flat_hash_map<gpu::GpuCliqueKey, std::vector<PreparedReceive>>
            grouped_receives =
                GroupReceivesByCliqueKey(std::move(prepared_receives));

        // Transfers for a particular clique are executed as a group. This
        // vector holds group futures for each clique_key in grouped_receives.
        std::vector<Future<>> group_futures;
        group_futures.reserve(grouped_receives.size());

        for (auto& [clique_key, curr_receives] : grouped_receives) {
          // Get the communicator on which we will execute this group of
          // transfers. We assume each clique key is associated with a unique
          // communicator, so we just take the communicator of the first
          // transfer_idx of this clique key.
          gpu::GpuCommunicator* gpu_communicator =
              curr_receives[0].clique_and_communicator_.second;

          // Launch the group of transfers.
          group_futures.push_back(gpu_communicator->GroupExecute(
              [&launch_receive_group, &curr_receives = curr_receives,
               stream](gpu::GpuCommunicator* gpu_comm) -> absl::Status {
                return launch_receive_group(
                    gpu_comm, absl::MakeSpan(curr_receives), stream);
              }));
        }

        // On a separate thread pool, await group futures and fulfill buffer
        // sequencing events and promises.
        Future<> all_receives_future = JoinFutures(group_futures);

        all_receives_future.OnReady(
            this->async_work_runner()->AsExecutor(),
            [this, local_device_state, stream,
             grouped_receives = std::move(grouped_receives),
             definition_event = std::move(definition_event)](
                const absl::Status& status) mutable {
              CHECK_OK(FulfillDeviceEvent(this, local_device_state, stream,
                                          definition_event, status));
            });
      };

  // Schedule transfers on the execute thread.
  local_device_state->execute_thread()->Schedule(
      std::move(execute_receives_fn));

  return buffers;
}

// Send functionality for original cross-host transfers API.
void StreamExecutorGpuClient::ScheduleRemoteSend(
    PjRtMemorySpace* memory_space,
    tsl::RCReference<CommonPjRtRawBuffer> raw_buffer,
    std::vector<tsl::RCReference<tsl::AsyncValue>> definition_events,
    tsl::RCReference<PjRtDeviceEventPromise> usage_event_promise,
    Future<std::string> serialized_descriptor,
    PjRtBuffer::RemoteSendCallback on_done) {
  // Get the default GpuCollectives instance.
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Default("gpu");
  if (!collectives.ok()) {
    on_done(collectives.status(), /*sends_were_enqueued=*/false);
  }
  gpu::GpuCollectives* gpu_collectives =
      tsl::down_cast<gpu::GpuCollectives*>(*collectives);
  if (gpu_collectives == nullptr) {
    auto error = absl::InternalError("Failed to get GPU collectives");
    on_done(error, /*sends_were_enqueued=*/false);
    usage_event_promise->SetError(error);
    return;
  }

  BufferSequencingEventRef usage_event =
      BufferSequencingEvent::Create(this->async_work_runner());

  // Keep memory alive until the event is done.
  usage_event.AndThen([raw_buffer]() {});

  serialized_descriptor.OnReady(
      [this, gpu_collectives = std::move(gpu_collectives),
       on_done = std::move(on_done),
       definition_events = std::move(definition_events),
       raw_buffer = std::move(raw_buffer), usage_event = usage_event](
          absl::StatusOr<std::string> serialized_descriptor) mutable {
        if (!serialized_descriptor.ok()) {
          on_done(serialized_descriptor.status(),
                  /*sends_were_enqueued=*/false);
          SetEventAsError(usage_event, serialized_descriptor.status());
        }
        auto events = absl::MakeSpan(definition_events);
        async_work_runner()->ScheduleWhenReady(
            events,
            [this, on_done = std::move(on_done),
             gpu_collectives = std::move(gpu_collectives),
             definition_events = std::move(definition_events),
             raw_buffer = std::move(raw_buffer), usage_event = usage_event,
             serialized_descriptor =
                 *std::move(serialized_descriptor)]() mutable {
              auto status = [&]() {
                for (const auto& event : definition_events) {
                  if (auto* status = event->GetErrorIfPresent()) {
                    return *status;
                  }
                }
                auto* local_device =
                    tensorflow::down_cast<PjRtStreamExecutorRawBuffer*>(
                        raw_buffer.get())
                        ->local_device();
                auto* stream = local_device->GetDeviceToDeviceStream();
                auto mem = tensorflow::down_cast<PjRtStreamExecutorRawBuffer*>(
                               raw_buffer.get())
                               ->device_buffer();
                CliqueId clique_id(serialized_descriptor);

                // Create a communicator.
                TF_ASSIGN_OR_RETURN(
                    std::unique_ptr<Communicator> communicator,
                    CreateTransferCommunicator(local_device, gpu_collectives,
                                               clique_id, /*is_sender=*/true));

                // Send data to the receiver.
                Future<> send_future = communicator->Send(
                    mem->mem(), xla::PrimitiveType::U8, mem->mem().size(),
                    RankId(0), gpu::GpuCollectives::On(*stream));
                TF_RETURN_IF_ERROR(send_future.Await());

                TF_RETURN_IF_ERROR(
                    AllocateAndRecordEvent(usage_event, local_device, stream));

                return absl::OkStatus();
              }();
              std::move(on_done)(status, /*sends_were_enqueued=*/status.ok());
              if (!status.ok()) {
                SetEventAsError(usage_event, status);
              }
            });
      });
  usage_event_promise->Set(
      tsl::MakeRef<PjRtStreamExecutorDeviceEvent>(std::move(usage_event)));
}

// Receive functionality for original cross-host transfers API.
absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
StreamExecutorGpuClient::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PjRtDevice* device,
    PjRtCrossHostRecvNotifier notifier) {
  // Validate arguments.
  if (shapes.empty()) {
    return InvalidArgument(
        "shapes parameter empty in MakeCrossHostReceiveBuffers");
  }
  if (shapes.size() != 1) {
    // TODO(mwhittaker): Support more than one shape.
    return Unimplemented(
        "StreamExecutorGpuClient::MakeCrossHostReceiveBuffers currently only "
        "supports one shape, but got %d",
        shapes.size());
  }
  Shape shape = shapes[0];

  // Get the default GpuCollectives instance.
  TF_ASSIGN_OR_RETURN(Collectives * collectives,
                      CollectivesRegistry::Default("gpu"));
  gpu::GpuCollectives* gpu_collectives =
      tsl::down_cast<gpu::GpuCollectives*>(collectives);
  if (gpu_collectives == nullptr) {
    return absl::InternalError("Failed to get GPU collectives");
  }

  TF_ASSIGN_OR_RETURN(
      StreamExecutorGpuClient::PrepareReceiveBufferResult receive_prep_result,
      PrepareReceiveBuffer(device, shape));

  auto recv = [this, gpu_collectives, notifier = std::move(notifier),
               local_device = receive_prep_result.local_device,
               definition_event = receive_prep_result.definition_event,
               stream = receive_prep_result.stream,
               raw_buffer = std::move(receive_prep_result.raw_buffer),
               shape = shapes[0],
               dtype = receive_prep_result.buffer->element_type()]() mutable {
    WaitForAllocation(stream, *raw_buffer);
    auto f = [&]() -> absl::Status {
      // Create a CliqueId.
      TF_ASSIGN_OR_RETURN(CliqueId clique_id,
                          gpu_collectives->CreateUniqueCliqueId());
      auto mem =
          tensorflow::down_cast<PjRtStreamExecutorRawBuffer*>(raw_buffer.get())
              ->device_buffer();

      // Notify the caller with the CliqueId. They will send the id to the
      // sender.
      //
      // TODO(mwhittaker): Implement cancellation.
      notifier(PjRtCrossHostRecvState{
          /*descriptors=*/{
              PjRtCrossHostRecvDescriptors{{clique_id.ToString()}}},
          /*cancel_notifier=*/nullptr,
      });

      // Create a communicator.
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Communicator> communicator,
          CreateTransferCommunicator(local_device, gpu_collectives, clique_id,
                                     /*is_sender=*/false));

      // Receive data from the sender.
      Future<> recv_future = communicator->Recv(
          mem->mem(), xla::PrimitiveType::U8, mem->mem().size(), RankId(1),
          gpu::GpuCollectives::On(*stream));
      TF_RETURN_IF_ERROR(recv_future.Await());

      // Keep mem alive until the Recv has finished executing. Note that
      // recv_event is fulfilled when the receive is enqueued, but not
      // necessarily executed.
      definition_event.AndThen([mem]() {});

      // Set definition event.
      TF_RETURN_IF_ERROR(
          AllocateAndRecordEvent(definition_event, local_device, stream));

      return absl::OkStatus();
    };

    if (absl::Status s = f(); !s.ok()) {
      SetEventAsError(definition_event, s);
    }
  };
  async_work_runner()->Schedule(recv);

  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.push_back(std::move(receive_prep_result.buffer));
  return buffers;
}

// ==== End cross-host transfer implementations ==== //

absl::StatusOr<const xla::PjRtTopologyDescription*>
StreamExecutorGpuClient::GetTopologyDescription() const {
  if (!topology_.has_value()) {
    return absl::FailedPreconditionError("GPU Topology is missing");
  }
  return &*topology_;
}

absl::StatusOr<Layout> StreamExecutorGpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  if (!topology_.has_value()) {
    return absl::FailedPreconditionError("GPU Topology is missing");
  }
  return topology_->GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::CompileAndLoad(mlir::ModuleOp module,
                                        CompileOptions options) {
  auto executable = PjRtStreamExecutorClient::CompileAndLoad(module, options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
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
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return executable;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::CompileAndLoad(const XlaComputation& computation,
                                        CompileOptions options) {
  auto executable =
      PjRtStreamExecutorClient::CompileAndLoad(computation, options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
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
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return executable;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::LoadSerialized(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  return PjRtStreamExecutorClient::LoadSerializedExecutable(serialized, options,
                                                            load_options);
}

namespace {

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::unique_ptr<se::GpuCudaMallocAsyncAllocator>>
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

  auto allocator = std::make_unique<se::GpuCudaMallocAsyncAllocator>(
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
absl::StatusOr<std::unique_ptr<tsl::Allocator>> CreateCudaAsyncAllocator(
    const LocalDeviceState& device, double memory_fraction, bool reserve_memory,
    bool create_new_pool, bool sync_mode, bool compute_stats = true) {
  return FailedPrecondition("CUDA async allocator requires CUDA >= 11.2");
}

#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

// Builds a LocalDeviceState for each GPU present.
absl::StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
absl::StatusOr<std::unique_ptr<se::DeviceAddressAllocator>>
GetStreamExecutorGpuDeviceAllocator(
    se::Platform* platform, const GpuAllocatorConfig& allocator_config,
    const std::map<int, std::unique_ptr<LocalDeviceState>>&
        addressable_devices) {
  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync: {
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto async_allocator,
            CreateCudaAsyncAllocator(
                *(ordinal_and_device.second), allocator_config.memory_fraction,
                allocator_config.preallocate, false, false, true));
        allocators.emplace_back(
            std::move(async_allocator),
            ordinal_and_device.second->compute_stream(),
            /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kDefault);
      }
      break;
    }

    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto bfc_allocator,
            CreateBFCAllocator(ordinal_and_device.second->executor(),
                               allocator_config.memory_fraction,
                               allocator_config.preallocate,
                               allocator_config.gpu_system_memory_size,
                               allocator_config.sub_allocator_alloc_visitors,
                               allocator_config.sub_allocator_free_visitors));
        allocators.emplace_back(
            std::move(bfc_allocator),
            ordinal_and_device.second->compute_stream(),
            /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kDefault);
      }
      break;
    }

    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      if (allocator_config.collective_memory_size != 0) {
        LOG(WARNING)
            << "collective_memory_size is non-zero, but allocator kind is set "
               "to \"platform\". Collective memory will not be allocated.";
      }
      // Returning null will cause the client to use the default backend
      // allocator.
      return nullptr;
  }

  // Add any additional allocators for alternate memory spaces.
  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto collective_bfc_allocator,
        CreateCollectiveBFCAllocator(
            ordinal_and_device.second->executor(),
            /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
            allocator_config.collective_memory_size));
    allocators.emplace_back(
        std::move(collective_bfc_allocator),
        ordinal_and_device.second->compute_stream(),
        /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kCollective);
  }

  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto host_allocator,
        GetGpuHostAllocator(ordinal_and_device.second->executor()));
    allocators.emplace_back(
        std::move(host_allocator), ordinal_and_device.second->compute_stream(),
        /*memory_space=*/
        static_cast<int>(stream_executor::MemorySpace::kHost));
  }

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
  const auto& debug_options = xla::GetDebugOptionsFromFlags();
  if (debug_options.xla_gpu_temp_buffer_use_separate_color()) {
    // Add memory allocator to allocate memory buffers with persistent temp
    // memory space color.
    for (const auto& ordinal_and_device : addressable_devices) {
      TF_ASSIGN_OR_RETURN(
          auto async_allocator,
          CreateCudaAsyncAllocator(*(ordinal_and_device.second), 1.0, false,
                                   true, true, true));
      allocators.emplace_back(
          std::move(async_allocator),
          ordinal_and_device.second->compute_stream(),
          /*memory_space=*/(int)xla::gpu::MemorySpaceColor::kTempBuffer);
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
  auto suffix = absl::StrFormat(":#global=%d,local=%d,process=%d,partition=%d#",
                                device_proto.global_device_id(),
                                device_proto.local_device_ordinal(),
                                node.node_id(), device_proto.partition_index());
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

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology,
    std::optional<int> partition_index,
    absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  LocalTopologyProto local_topology;
  local_topology.set_node_id(node_id);
  std::string boot_id_str;
  auto boot_id_str_or_status = GetBootIdString();
  if (!boot_id_str_or_status.ok()) {
    LOG(INFO) << boot_id_str_or_status.status();
  } else {
    boot_id_str = boot_id_str_or_status.value();
  }
  local_topology.set_boot_id(boot_id_str);
  if (partition_index.has_value()) {
    local_topology.set_partition_index(*partition_index);
  }
  for (const auto& ordinal_and_device : local_device_states) {
    const se::Platform* platform =
        ordinal_and_device.second->executor()->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(
            ordinal_and_device.second->local_hardware_id().value()));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(ordinal_and_device.first);
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
    auto compute_capability = MakeComputeCapabilityString(desc.get());
    device_proto->set_compute_capability(compute_capability);
    device_proto->set_core_count(desc->core_count());
    device_proto->set_shared_memory_per_block_optin(
        desc->shared_memory_per_block_optin());
    device_proto->set_numa_node(desc->numa_node());

    stream_executor::DeviceInterconnectInfo info =
        desc->device_interconnect_info();
    if (!info.cluster_uuid.empty() && !info.clique_id.empty()) {
      device_proto->set_fabric_uuid(
          absl::StrCat(info.cluster_uuid, "/", info.clique_id));
    }
  }

  GlobalTopologyProto global_topology;
  if (enable_mock_nccl) {
    TopologySizes sizes;
    if (mock_gpu_topology.has_value()) {
      TF_ASSIGN_OR_RETURN(sizes, TopologySizes::FromString(*mock_gpu_topology));
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
        int node_id = i * sizes.num_hosts_per_partition + j;
        local_topologies[node_id].set_node_id(node_id);
        local_topologies[node_id].set_boot_id(absl::StrCat(i));
      }
    }
    TF_ASSIGN_OR_RETURN(global_topology,
                        BuildGlobalTopology(absl::MakeSpan(local_topologies),
                                            /*assign_global_device_ids=*/true));
  } else {
    TF_RETURN_IF_ERROR(ExchangeTopologies(
        platform_name, node_id, num_nodes, get_local_topology_timeout,
        get_global_topology_timeout, kv_store.get(), local_topology,
        &global_topology, /*assign_global_device_ids=*/true));
  }

  absl::btree_map<LocalDeviceId, GlobalDeviceId> gpu_device_ids;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process;
  int curr_partition_index = -1;
  int curr_process_index = -1;
  int curr_process_index_in_partition = 0;
  for (const LocalTopologyProto& node : global_topology.nodes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      // The devices in the global topology are ordered by node_id,
      // partition_index. This is guaranteed by the `BuildGlobalTopology`
      // function and the `ExchangeTopologies` function.
      if (curr_partition_index != device_proto.partition_index()) {
        curr_partition_index = device_proto.partition_index();
        curr_process_index = node.node_id();
        curr_process_index_in_partition = 0;
      }
      if (curr_process_index != node.node_id()) {
        curr_process_index = node.node_id();
        curr_process_index_in_partition++;
      }

      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_process[global_device_id] = node.node_id();
      std::unique_ptr<LocalDeviceState> local_device;
      if (node.node_id() == node_id) {
        auto it = local_device_states.find(device_proto.local_device_ordinal());
        TF_RET_CHECK(it != local_device_states.end())
            << device_proto.local_device_ordinal();
        TF_RET_CHECK(it->second != nullptr);
        local_device = std::move(it->second);
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
          device_proto.shared_memory_per_block_optin(),
          device_proto.local_device_ordinal(), node.node_id(),
          curr_process_index_in_partition, device_proto.partition_index(),
          device_proto.numa_node());
      devices.push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device.second == nullptr);
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));

  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Default("gpu"));
  xla::gpu::GpuCollectives* gpu_collectives =
      tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);

  if (gpu_collectives == nullptr) {
    return absl::InternalError("Failed to get GPU collectives");
  }

  size_t num_processes = global_topology.nodes().size();
  TF_ASSIGN_OR_RETURN(
      auto clique_id_callback,
      gpu_collectives->InitializeTopology({ProcessId(node_id), num_processes,
                                           local_device_states.size(), kv_store,
                                           device_to_process}));
  gpu_executable_run_options->set_clique_id_callback(
      std::move(clique_id_callback));

  TF_ASSIGN_OR_RETURN(GpuTopologyProto gpu_topology,
                      BuildGpuTopology(global_topology));
  return std::make_pair(std::move(devices), gpu_topology);
}

std::string MakeComputeCapabilityString(const se::DeviceDescription* desc) {
  se::GpuComputeCapability cc = desc->gpu_compute_capability();
  if (cc.IsCuda()) {
    auto* nvcc = cc.cuda_compute_capability();
    return absl::StrCat(nvcc->major, ".", nvcc->minor);
  }
  if (cc.IsRocm()) {
    auto* rocmcc = cc.rocm_compute_capability();
    return rocmcc->gfx_version();
  }
  return "unknown";
}

StreamExecutorGpuDevice::StreamExecutorGpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor,
    std::string compute_capability, int core_count,
    int shared_memory_per_block_optin, int local_device_id, int process_index,
    int process_index_in_partition, int partition_index, int numa_node)
    : PjRtStreamExecutorDevice(
          id, std::move(local_device_state), local_device_id, process_index,
          process_index_in_partition, partition_index, std::move(device_kind)),
      device_vendor_(std::move(device_vendor)) {
  StreamExecutorGpuTopologyDescription::SetupDeviceDescription(
      description(), device_vendor_, compute_capability, core_count,
      static_cast<int64_t>(shared_memory_per_block_optin), partition_index,
      numa_node);
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

  TF_ASSIGN_OR_RETURN(auto allocator, allocator_adapter->GetAllocator(
                                          local_device_id().value()));

  auto stats = allocator->GetStats();
  TF_RET_CHECK(stats.has_value());
  return stats.value();
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
  auto pjrt_platform_name = xla::SyclName();
#else   // TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(xla_client));
  EnablePeerAccess(xla_client->backend().stream_executors());
  TF_ASSIGN_OR_RETURN(auto allocator,
                      GetStreamExecutorGpuDeviceAllocator(
                          xla_client->platform(), options.allocator_config,
                          local_device_states));
  std::unique_ptr<HostMemoryAllocator> host_memory_allocator;
  if (options.host_memory_allocator_factory != nullptr) {
    stream_executor::StreamExecutor* const stream_executor =
        local_device_states.begin()->second->compute_stream()->parent();
    HostMemoryAllocator::Options allocator_options;
    allocator_options.alignment = tsl::Allocator::kAllocatorAlignment;
    allocator_options.map_fn = [stream_executor](void* data, size_t size) {
      bool success = stream_executor->HostMemoryRegister(data, size);
      if (!success) {
        return absl::InternalError(absl::StrFormat(
            "Failed to register host memory at address: %ps", data));
      }
      return absl::OkStatus();
    };
    allocator_options.unmap_fn = [stream_executor](void* data) {
      bool success = stream_executor->HostMemoryUnregister(data);
      if (!success) {
        return absl::InternalError(absl::StrFormat(
            "Failed to unregister host memory at address: %ps", data));
      }
      return absl::OkStatus();
    };
    host_memory_allocator =
        options.host_memory_allocator_factory(std::move(allocator_options));
  } else {
    TF_ASSIGN_OR_RETURN(
        auto allocator,
        GetGpuHostAllocator(local_device_states.begin()->second->executor()));
    host_memory_allocator = std::make_unique<BasicHostMemoryAllocator>(
        std::move(allocator), tsl::Allocator::kAllocatorAlignment);
  }

  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_collectives();
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
  TF_ASSIGN_OR_RETURN(
      DeviceTopologyPair device_topology_pair,
      BuildDistributedDevices(
          pjrt_platform_name, std::move(local_device_states), options.node_id,
          options.num_nodes, gpu_run_options.get(), kv_store,
          options.enable_mock_nccl, options.mock_gpu_topology,
          options.partition_index));

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(device_topology_pair.second));

  return std::make_unique<StreamExecutorGpuClient>(
      pjrt_platform_name, xla_client, std::move(device_topology_pair.first),
      options.node_id, std::move(allocator), std::move(host_memory_allocator),
      options.should_stage_host_to_device_transfers, std::move(gpu_run_options),
      std::move(kv_store), options.abort_collectives_on_failure,
      std::move(gpu_topology), options.num_nodes);
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& desc =
        ordinal_and_device.second->executor()->GetDeviceDescription();
    auto device = std::make_unique<StreamExecutorGpuDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second),
        desc.name(), desc.device_vendor(), MakeComputeCapabilityString(&desc),
        desc.core_count(), desc.shared_memory_per_block_optin(),
        ordinal_and_device.second->local_device_id().value(), node_id,
        desc.numa_node());
    devices.push_back(std::move(device));
  }
  return devices;
}

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
static absl::Status CheckAlignment(const BufferAllocation& allocation,
                                   se::DeviceAddressBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return gpu::kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return gpu::kConstantBufferAlignBytes;
    } else {
      return gpu::kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

absl::StatusOr<PjRtStreamExecutorExecutionOutput>
StreamExecutorGpuClient::RunAsync(
    LocalExecutable& exec, PjRtDevice* device,
    std::vector<PjRtStreamExecutorExecutionInput> flat_arguments,
    ExecutableRunOptions run_options_inp, bool parameter_is_tupled_arguments,
    absl::Span<const Shape> executable_parameter_shapes) {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(flat_arguments.size());
  for (const Shape& arg_shape : executable_parameter_shapes) {
    argument_shapes.push_back(&arg_shape);
  }

  TF_ASSIGN_OR_RETURN(auto options_and_stream,
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

    TF_ASSIGN_OR_RETURN(
        globals, gpu_exec->ResolveConstantGlobals(run_options->stream()));
  }

  absl::Span<const BufferAllocation* const> allocations =
      gpu_exec->GetAllocations();

  std::vector<se::DeviceAddressBase> buffers(allocations.size());
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Build buffer allocations"); },
        tsl::profiler::TraceMeLevel::kInfo);
    const int64_t num_buffers = allocations.size();
    for (int64_t i = 0; i < num_buffers; ++i) {
      const BufferAllocation& allocation = *allocations[i];
      se::DeviceAddressBase& buffer = buffers[i];
      if (allocation.is_thread_local()) {
        // buffer = se::DeviceAddressBase{};
      } else if (allocation.is_entry_computation_parameter()) {
        int64_t param_no;
        if (parameter_is_tupled_arguments) {
          // TODO(parkers): Change compiler to not even pretend to read
          // the tuple index tables (also GPU shouldn't tuple ever).
          if (allocation.param_shape_index().empty()) {
            continue;
          }
          param_no = allocation.param_shape_index()[0];
        } else {
          param_no = allocation.parameter_number();
        }
        buffer = flat_arguments[param_no].buf->mem();
        if (buffer.is_null() && buffer.size() > 0) {
          return FailedPrecondition(
              "Cannot run XLA computation because pointer to (sub-)buffer at "
              "index %s of parameter %d was null.  All pointers to "
              "(sub-)buffers must not be null, unless the (sub-)buffer has "
              "zero elements.",
              allocation.param_shape_index().ToString(), param_no);
        }
      } else if (allocation.is_constant()) {
        auto it = globals->find(i);
        if (it != globals->end()) {
          buffer = it->second;
        }
      } else {
        // Allocate each allocation that might escape, or is the temp buffer.
        CHECK(allocation.maybe_live_out() ||
              allocation.IsPreallocatedTempBuffer());
        const int64_t buffer_size = allocation.size();
        if (buffer_size > 0) {
          TF_ASSIGN_OR_RETURN(
              se::ScopedDeviceAddress<uint8_t> owning_buffer,
              memory_allocator->Allocate(device_ordinal, buffer_size,
                                         /*retry_on_failure=*/true,
                                         /*memory_space=*/allocation.color()));
          buffer = owning_buffer.Release();
        }
      }
      TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
    }
  }
  xla::gpu::BufferAllocations buffer_allocations(buffers, device_ordinal,
                                                 memory_allocator);
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Buffer allocations: " << buffer_allocations.ToString();

  std::set<se::DeviceAddressBase> buffers_in_result;

  xla::ShapeTree<tsl::AsyncValueRef<RawSEDeviceMemory>> results(
      gpu_exec->result_shape());

  for (auto& p : results) {
    const ShapeIndex& index = p.first;
    if (!gpu_exec->output_info().contains(index)) {
      continue;
    }
    const gpu::GpuExecutable::OutputInfo& output_info =
        gpu_exec->output_info().at(index);
    const BufferAllocation* allocation =
        allocations[output_info.allocation_index];
    se::DeviceAddressBase result_buffer;

    XLA_VLOG_DEVICE(4, device_ordinal)
        << "Looking at: allocation " << output_info.allocation_index
        << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      PjRtStreamExecutorExecutionInput& input =
          flat_arguments[parameter_is_tupled_arguments
                             ? allocation->param_shape_index()[0]
                             : allocation->parameter_number()];
      if (output_info.alias_config->must_alias() && !input.is_donated) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (input.is_donated) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        buffers_in_result.insert(input.buf->mem());
        p.second = input.buf;
        input.is_donated = false;
        continue;
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(gpu_exec->result_shape(), index)
                      .IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        XLA_VLOG_DEVICE(3, device_ordinal)
            << "Using copy-protection: aliasing is specified, but the "
               "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size = ShapeUtil::ByteSizeOf(
            ShapeUtil::GetSubshape(gpu_exec->result_shape(), index));
        absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size,
                                       /*retry_on_failure=*/true,
                                       /*memory_space=*/allocation->color());
        if (!allocated_buffer.ok()) {
          return gpu_exec->VerboseAllocationError(allocated_buffer.status());
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceAddressBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
            &result_buffer, aliased_buffer, aliased_buffer.size()));
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);
    }
    buffers_in_result.insert(result_buffer);

    p.second = RawSEDeviceMemory::Create(
        result_buffer,
        tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
            ->local_device_state(),
        memory_allocator);
  }

  TF_RETURN_IF_ERROR(gpu_exec->ExecuteThunks(buffer_allocations, run_options));

  TF_RETURN_IF_ERROR(buffer_allocations.TearDown(buffers_in_result,
                                                 gpu_exec->GetAllocations()));

  std::vector<tsl::AsyncValueRef<RawSEDeviceMemory>> to_be_released;

  // Free allocations for arguments.
  for (PjRtStreamExecutorExecutionInput& input : flat_arguments) {
    if (input.is_donated) {
      to_be_released.push_back(std::move(input.buf));
    }
  }

  return PjRtStreamExecutorExecutionOutput(
      {std::move(results), std::move(to_be_released), {}});
#else
  return PjRtStreamExecutorClient::RunAsync(
      exec, device, std::move(flat_arguments), std::move(run_options_inp),
      parameter_is_tupled_arguments, executable_parameter_shapes);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace xla
