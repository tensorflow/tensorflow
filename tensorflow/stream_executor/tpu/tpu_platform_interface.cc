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

#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace tpu {

/* static */
TpuPlatformInterface* TpuPlatformInterface::GetRegisteredPlatform() {
  // Prefer TpuPlatform if it's registered.
  auto status_or_tpu_platform =
      stream_executor::MultiPlatformManager::PlatformWithName("TPU");
  if (status_or_tpu_platform.ok()) {
    return static_cast<TpuPlatformInterface*>(
        status_or_tpu_platform.ValueOrDie());
  }
  if (status_or_tpu_platform.status().code() != error::NOT_FOUND) {
    LOG(WARNING) << "Error when getting the TPU platform: "
                 << status_or_tpu_platform.status();
    return nullptr;
  }

  // Use any other registered TPU platform.
  auto status_or_other_tpu_platforms =
      stream_executor::MultiPlatformManager::PlatformsWithFilter(
          [](const stream_executor::Platform* platform) {
            return dynamic_cast<const TpuPlatformInterface*>(platform) !=
                   nullptr;
          });
  if (!status_or_other_tpu_platforms.ok()) {
    LOG(WARNING) << "Error when getting other TPU platforms: "
                 << status_or_tpu_platform.status();
    return nullptr;
  }
  auto other_tpu_platforms = status_or_other_tpu_platforms.ValueOrDie();
  if (!other_tpu_platforms.empty()) {
    LOG(WARNING) << other_tpu_platforms.size()
                 << " TPU platforms registered, selecting "
                 << other_tpu_platforms[0]->Name();
    return static_cast<TpuPlatformInterface*>(other_tpu_platforms[0]);
  }

  LOG(WARNING) << "No TPU platform registered";
  return nullptr;
}

}  // namespace tpu
}  // namespace tensorflow
