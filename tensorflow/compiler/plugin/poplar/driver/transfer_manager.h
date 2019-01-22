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

#ifndef TENSORFLOW_COMPILER_XLA_PLUGIN_POPLAR_DRIVER_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_PLUGIN_POPLAR_DRIVER_TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace poplarplugin {

class PoplarTransferManager : public GenericTransferManager {
 public:
  PoplarTransferManager();

  ~PoplarTransferManager() override = default;

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;

 private:
  Status TransferBufferToInfeed(se::StreamExecutor* executor, int64 size,
                                const void* source);

  StatusOr<cpu::runtime::XfeedBuffer*> TransferBufferToInfeedInternal(
      se::StreamExecutor* executor, int64 size, const void* source);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoplarTransferManager);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
