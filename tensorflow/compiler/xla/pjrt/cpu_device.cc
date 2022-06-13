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

#include "tensorflow/compiler/xla/pjrt/cpu_device.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

namespace xla {

static const char kCpuPlatformName[] = "cpu";

CpuDevice::CpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               /*device_kind=*/kCpuPlatformName) {}

StatusOr<std::unique_ptr<PjRtClient>> GetCpuClient(bool asynchronous) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("Host"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("CPU platform has no visible devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutorConfig config;
    config.ordinal = i;
    // 8MiB stacks seem to be necessary for running LAPACK/OpenBLAS
    // computations.
    config.device_options.non_portable_tags["host_thread_stack_size_in_bytes"] =
        absl::StrCat(8192 * 1024);
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        platform->GetExecutor(config));
    auto device_state = std::make_unique<LocalDeviceState>(
        executor, client, LocalDeviceState::kSynchronous,
        /*max_inflight_computations=*/32,
        /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
    auto device = std::make_unique<CpuDevice>(i, std::move(device_state));
    devices.push_back(std::move(device));
  }

  return std::unique_ptr<PjRtClient>(std::make_unique<PjRtStreamExecutorClient>(
      CpuName(), client, std::move(devices), /*process_index=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr));
}

}  // namespace xla
