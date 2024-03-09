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

#include "xla/stream_executor/tpu/tpu_transfer_manager_interface.h"

#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"

namespace xla {

/*static*/ TpuTransferManagerInterface*
TpuTransferManagerInterface::GetRegisteredTpuTransferManager() {
  auto* platform = tensorflow::tpu::TpuPlatformInterface::GetRegisteredPlatform(
      /*initialize_platform=*/false);
  if (platform == nullptr) {
    LOG(ERROR) << "Unable to retrieve registered TPU platform.";
    return nullptr;
  }
  auto tm = xla::TransferManager::GetForPlatform(platform);
  if (!tm.ok()) {
    LOG(ERROR) << "Unable to retrieve TpuTransferManager. No TPU platform is "
                  "registered for platform "
               << platform->Name() << " and ID " << platform->id();
    return nullptr;
  }
  return static_cast<TpuTransferManagerInterface*>(tm.value());
}

}  // namespace xla
