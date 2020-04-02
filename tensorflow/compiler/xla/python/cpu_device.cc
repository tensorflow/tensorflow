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

#include "tensorflow/compiler/xla/python/cpu_device.h"

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

namespace xla {

static const char kCpuPlatformName[] = "cpu";

CpuDevice::CpuDevice(int id,
                     std::unique_ptr<LocalDeviceState> local_device_state)
    : Device(id, std::move(local_device_state), kCpuPlatformName) {}

StatusOr<std::shared_ptr<PyLocalClient>> GetCpuClient(bool asynchronous) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("Host"));
  if (platform->VisibleDeviceCount() <= 0) {
    return FailedPrecondition("CPU platform has no visible devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(LocalClient * client,
                      ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<Device>> devices;
  for (int i = 0; i < client->device_count(); ++i) {
    se::StreamExecutor* executor =
        client->backend().stream_executor(i).ValueOrDie();
    auto device_state = absl::make_unique<LocalDeviceState>(
        executor, client, /*synchronous_deallocation=*/true, asynchronous,
        /*allow_event_reuse=*/false);
    auto device = absl::make_unique<CpuDevice>(i, std::move(device_state));
    devices.push_back(std::move(device));
  }

  return std::make_shared<PyLocalClient>(
      kCpuPlatformName, client, std::move(devices), /*host_id=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*gpu_run_options=*/nullptr);
}

}  // namespace xla
