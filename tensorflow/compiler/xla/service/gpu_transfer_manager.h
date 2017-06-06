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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRANSFER_MANAGER_H_

#include <vector>

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// An implementation of the XLA GenericTransferManager that
// handles GPU-specific infeed.
class GpuTransferManager : public GenericTransferManager {
 public:
  GpuTransferManager();
  ~GpuTransferManager() override {}

  Status TransferLiteralToInfeed(perftools::gputools::StreamExecutor* executor,
                                 const Literal& literal) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GpuTransferManager);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRANSFER_MANAGER_H_
