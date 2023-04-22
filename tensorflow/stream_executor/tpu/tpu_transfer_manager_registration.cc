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

#include <memory>

#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_id.h"
#include "tensorflow/stream_executor/tpu/tpu_transfer_manager.h"

namespace tensorflow {
namespace tpu {

static std::unique_ptr<xla::TransferManager> CreateTpuTransferManager() {
  return std::make_unique<TpuTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(GetTpuPlatformId(),
                                                CreateTpuTransferManager);
  return true;
}
static bool module_initialized = InitModule();

}  // namespace tpu
}  // namespace tensorflow
