/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/interpreter/interpreter_transfer_manager.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/interpreter/platform_id.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"

namespace xla {

InterpreterTransferManager::InterpreterTransferManager()
    : GenericTransferManager(se::interpreter::kXlaInterpreterPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

}  // namespace xla

static std::unique_ptr<xla::TransferManager>
CreateInterpreterTransferManager() {
  return absl::make_unique<xla::InterpreterTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::interpreter::kXlaInterpreterPlatformId,
      &CreateInterpreterTransferManager);
  return true;
}

static bool module_initialized = InitModule();
