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

#include "tensorflow/compiler/xla/service/cpu/cpu_transfer_manager.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

CpuTransferManager::CpuTransferManager()
    : GenericTransferManager(se::host::kHostPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

Status CpuTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  return TransferLiteralToInfeedOnCpu(executor->device_ordinal(), literal);
}

Status CpuTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
  return TransferLiteralFromOutfeedOnCpu(executor->device_ordinal(), literal);
}

Status CpuTransferManager::ReadDynamicShapes(se::Stream* stream,
                                             ShapedBuffer* device_buffer,
                                             Shape* device_shape) {
  if (stream != nullptr) {
    // When a stream is presented, respect the stream dependency.
    return TransferManager::ReadDynamicShapes(stream, device_buffer,
                                              device_shape);
  }
  TF_ASSIGN_OR_RETURN(auto platform,
                      se::MultiPlatformManager::PlatformWithId(PlatformId()));
  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));
  return ReadDynamicShapesOnCpu(device_buffer, device_shape,
                                compiler->ShapeSizeBytesFunction());
}
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateCpuTransferManager() {
  return std::make_unique<xla::CpuTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::host::kHostPlatformId, &CreateCpuTransferManager);
  return true;
}
static bool module_initialized = InitModule();
