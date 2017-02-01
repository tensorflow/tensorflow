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

#include "tensorflow/compiler/poplar/driver/poplar_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/poplar/poplar_platform_id.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/infeed_manager.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace poplar {

PoplarTransferManager::PoplarTransferManager()
        : GenericTransferManager(se::host::kHostPlatformId) {}

Status
PoplarTransferManager::TransferLiteralToInfeed(se::StreamExecutor *executor,
                                               const Literal &literal) {
  const Shape &shape = literal.shape();
  VLOG(2) << "transferring literal shape to infeed: "
          << ShapeUtil::HumanString(shape);

  return Status::OK();
}

}  // namespace poplar
}  // namespace xla

static xla::TransferManager* CreatePoplarTransferManager() {
  return new xla::poplar::PoplarTransferManager();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(se::poplar::kPoplarPlatformId,
                                                &CreatePoplarTransferManager);
  return true;
}
static bool module_initialized = InitModule();
