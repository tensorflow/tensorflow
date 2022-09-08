/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/tpu_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/tpu_computation_placer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_executable.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_executable_interface.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_executor_interface.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_stream.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tpu/tpu_initializer_helper.h"
#include "tensorflow/tsl/platform/casts.h"

namespace tf_tpu = tensorflow::tpu;

namespace xla {
namespace {

class TpuDeviceState : public LocalDeviceState {
 public:
  TpuDeviceState(se::StreamExecutor* executor, LocalClient* client,
                 int max_inflight_computations);

  Status ThenMemcpyDeviceToDevice(se::Stream* transfer_stream,
                                  se::Stream* dst_stream,
                                  se::DeviceMemoryBase src_buffer,
                                  se::DeviceMemoryBase dst_buffer) override;
};

TpuDeviceState::TpuDeviceState(se::StreamExecutor* executor,
                               LocalClient* client,
                               int max_inflight_computations)
    : LocalDeviceState(executor, client, LocalDeviceState::kAsynchronous,
                       max_inflight_computations,
                       /*allow_event_reuse=*/false,
                       /*use_callback_stream=*/true) {}

Status TpuDeviceState::ThenMemcpyDeviceToDevice(
    se::Stream* transfer_stream, se::Stream* dst_stream,
    se::DeviceMemoryBase src_buffer, se::DeviceMemoryBase dst_buffer) {
  auto* transfer_tpu_stream = tensorflow::down_cast<tf_tpu::TpuStream*>(
      transfer_stream->implementation());
  TF_RETURN_IF_ERROR(transfer_tpu_stream->EnqueueOnTpuDeviceSendRecvLocal(
      src_buffer, dst_buffer));
  return OkStatus();
}

}  // namespace

PjRtTpuClient::PjRtTpuClient(
    LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index)
    : PjRtStreamExecutorClient(TpuName(), client, std::move(devices),
                               process_index,
                               /*allocator=*/nullptr,
                               /*host_memory_allocator=*/nullptr,
                               /*should_stage_host_to_device_transfers=*/false,
                               /*gpu_run_options=*/nullptr),
      platform_version_([]() {
        // Example platform version string:
        //   libtpu version 0.0.1
        //   Built on Mar 4 2021 15:25:57 (1614900357) cl/360760169
        tf_tpu::TpuPlatformInterface* platform =
            tf_tpu::TpuPlatformInterface::GetRegisteredPlatform();
        TpuRuntimeVersion version = platform->version();
        return absl::StrCat(
            "libtpu version ", absl::StrJoin(version.version, "."), "\n",
            absl::string_view(version.metadata, version.metadata_size));
      }()) {
  // We always initialize the tpu client even if libtpu isn't linked in or
  // initialized.
  if (tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_InitFn !=
      nullptr) {
    tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_InitFn();
  }
}

PjRtTpuClient::~PjRtTpuClient() {
  if (tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_ShutdownFn !=
      nullptr) {
    tf_tpu::ExecutorApiFn()->TpuAsyncCollectiveOffloadHelper_ShutdownFn();
  }
}

StatusOr<DeviceAssignment> PjRtTpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  tf_tpu::TpuPlatformInterface* platform =
      tf_tpu::TpuPlatformInterface::GetRegisteredPlatform();
  tf_tpu::TpuHostLocationExternal host = platform->GetTpuHostLocation();
  int num_local_devices = host.Cores(kTensorCore).size();
  if (num_replicas * num_partitions <= num_local_devices) {
    return tf_tpu::TpuComputationPlacer::AssignLocalDevices(host, num_replicas,
                                                            num_partitions);
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

StatusOr<std::optional<std::string>> PjRtTpuClient::ExecutableFingerprint(
    const PjRtLoadedExecutable& executable) const {
  if (executable.client() != this) {
    return InvalidArgument(
        "Passed executable from different client (platform '%s') to "
        "PjRtTpuClient::ExecutableFingerprint",
        executable.client()->platform_name());
  }
  if (executable.num_partitions() > 1) {
    LOG(INFO) << "ExecutableFingerprint not fully implemented for MPMD "
                 "executables, fingerprint may not be unique.";
  }
  xla::TpuExecutableInterface* tpu_executable =
      tensorflow::down_cast<xla::TpuExecutableInterface*>(
          tensorflow::down_cast<const PjRtStreamExecutorExecutable*>(
              &executable)
              ->executables()[0]
              ->executable());
  return std::optional<std::string>(tpu_executable->fingerprint());
}

StatusOr<std::string> PjRtTpuClient::SerializeExecutable(
    const PjRtLoadedExecutable& executable) const {
  const PjRtStreamExecutorExecutable* se_executable =
      tensorflow::down_cast<const PjRtStreamExecutorExecutable*>(&executable);
  if (se_executable->executables().size() > 1) {
    return Unimplemented(
        "PjRtTpuClient::SerializeExecutable unimplemented for MPMD "
        "executables");
  }
  const TpuExecutable* tpu_executable =
      tensorflow::down_cast<const TpuExecutable*>(
          se_executable->executables()[0]->executable());
  return tpu_executable->Serialize();
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtTpuClient::DeserializeExecutable(absl::string_view serialized,
                                     CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TpuExecutable> tpu_executable,
                      TpuExecutable::Deserialize(serialized));

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));

  // TODO(skyewm): can we streamline this? e.g. removing proto serialization
  XlaComputation computation(tpu_executable->module().ToProto());
  TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                      computation.GetProgramShape());
  std::vector<const Shape*> unused_argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = client()](Shape shape) {
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &unused_argument_layout_pointers));

  auto local_executable = std::make_unique<LocalExecutable>(
      std::move(tpu_executable), client_->mutable_backend(),
      options.executable_build_options);
  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.emplace_back(std::move(local_executable));

  auto pjrt_executable = std::make_unique<PjRtStreamExecutorExecutable>(
      std::move(local_executables), options.parameter_is_tupled_arguments,
      std::move(extras.device_assignment),
      std::move(extras.addressable_device_logical_ids),
      std::move(extras.addressable_devices), this);
  TF_RETURN_IF_ERROR(
      pjrt_executable->SetUpDonation(options.parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(pjrt_executable));
}

static StatusOr<std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>>
GetTpuDevices(
    LocalClient* client,
    std::vector<std::unique_ptr<LocalDeviceState>> local_device_states) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  tf_tpu::TpuTopologyExternal topology =
      tf_tpu::TpuPlatformInterface::GetRegisteredPlatform()->topology();

  std::map<int, int> core_id_to_device_ordinal;
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor = client->backend().stream_executor(i).value();
    tf_tpu::TpuExecutorInterface* tpu_executor =
        tensorflow::down_cast<tf_tpu::TpuExecutorInterface*>(
            executor->implementation());
    core_id_to_device_ordinal[tpu_executor->GetCoreLocationExternal().Id()] = i;
  }

  for (const tf_tpu::TpuCoreLocationExternal& core :
       topology.cores(TpuCoreTypeEnum::kTensorCore)) {
    auto it = core_id_to_device_ordinal.find(core.Id());
    int device_ordinal =
        (it != core_id_to_device_ordinal.end()) ? it->second : -1;
    int process_index = topology.IdForHost(core.host_coordinates());
    const tf_tpu::TpuDimensionsExternal coords = core.chip_coordinates();
    std::array<int, 3> coords_array = {coords.x, coords.y, coords.z};
    std::unique_ptr<LocalDeviceState> local_device_state;
    if (device_ordinal >= 0) {
      local_device_state = std::move(local_device_states[device_ordinal]);
    }
    auto device = std::make_unique<PjRtTpuDevice>(
        core, std::move(local_device_state), process_index, coords_array,
        std::string(tf_tpu::TpuVersionEnumToString(topology.version())));
    devices.push_back(std::move(device));
  }
  return devices;
}

StatusOr<std::shared_ptr<PjRtClient>> GetTpuClient(
    int max_inflight_computations, absl::Duration init_retry_timeout) {
#if !defined(PLATFORM_GOOGLE) || defined(LIBTPU_STATIC)
  TF_RETURN_IF_ERROR(tensorflow::tpu::FindAndLoadTpuLibrary());
#endif
  tf_tpu::TpuPlatformInterface* platform =
      tf_tpu::TpuPlatformInterface::GetRegisteredPlatform(
          /*initialize_platform=*/true, /*num_tries=*/1);
  if (platform == nullptr) {
    return InvalidArgument("TpuPlatform is not available.");
  }
  // NOTE: We retry in a loop since some pod failures are transient (e.g. some
  // RPCs may timeout waiting for other hosts to come up, but will succeed
  // at a later point if retried).
  auto start = absl::Now();
  while (true) {
    Status status = platform->Initialize({});
    if (status.ok()) {
      break;
    }
    // TODO(b/165870356): refactor this loop to be
    // while(!platform->Initialized()) once the Initialized() function works
    // correctly, and remove this check. The platform may already be initialized
    // when running internally.
    if (status.code() == tensorflow::error::ALREADY_EXISTS) {
      LOG(INFO) << "TpuPlatform already initialized, continuing...";
      break;
    }
    LOG(INFO) << "TPU platform initialization failed: " << status;
    if ((absl::Now() - start) >= init_retry_timeout) {
      return status;
    }
    absl::SleepFor(absl::Microseconds(10));
  }
  CHECK(platform->Initialized());
  if (platform->VisibleDeviceCount() <= 0) {
    return InvalidArgument("No TPU devices found.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<LocalDeviceState>> local_device_states;
  local_device_states.reserve(client->device_count());
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor = client->backend().stream_executor(i).value();
    local_device_states.push_back(std::make_unique<TpuDeviceState>(
        executor, client, max_inflight_computations));
  }

  TF_ASSIGN_OR_RETURN(auto devices,
                      GetTpuDevices(client, std::move(local_device_states)));
  int process_index = platform->GetTpuHostLocation().Id();

  return std::shared_ptr<PjRtClient>(std::make_unique<PjRtTpuClient>(
      client, std::move(devices), process_index));
}

}  // namespace xla
