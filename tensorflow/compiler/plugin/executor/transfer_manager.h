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

#ifndef TENSORFLOW_COMPILER_EXECUTOR_DRIVER_EXECUTOR_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_EXECUTOR_DRIVER_EXECUTOR_TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

#include <vector>

namespace se = ::perftools::gputools;

namespace xla {
namespace executorplugin {

class ExecutorTransferManager : public TransferManager {
 public:
  ExecutorTransferManager();

  ~ExecutorTransferManager() override {}

  se::Platform::Id PlatformId() const override;

  StatusOr<std::vector<se::DeviceMemoryBase>> ShallowCopyTupleFromDevice(
      se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
      const Shape& shape) override;

  Status TransferLiteralFromDevice(se::StreamExecutor* executor,
                                   const se::DeviceMemoryBase& source,
                                   const Shape& device_shape,
                                   const Shape& literal_shape,
                                   Literal* literal) override;

  Status TransferLiteralToDevice(se::StreamExecutor* executor,
                                 const Literal& literal,
                                 se::DeviceMemoryBase* destination) override;

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const Literal& literal) override;

  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    const Shape& literal_shape,
                                    Literal* literal) override;

  Status ResetDevices(
      tensorflow::gtl::ArraySlice<se::StreamExecutor*> executors) override;

  int64 GetByteSizeRequirement(const Shape& shape) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorTransferManager);
};

}  // namespace executorplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_EXECUTOR_DRIVER_EXECUTOR_TRANSFER_MANAGER_H_
