/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime_intrinsics.h"

#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

extern const char* const kXlaGpuAssertCustomCallTag = "__xla_gpu_assert";

static Status AssertOnGpu(void* stream_handle, void* buffer,
                          absl::string_view error_msg) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::MultiPlatformManager::PlatformWithName("CUDA"));
  se::StreamExecutorConfig config;
  config.gpu_stream = stream_handle;
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      platform->GetExecutor(config));
  se::Stream* stream = executor->FindAllocatedStream(stream_handle);
  if (!stream) {
    return InternalError("Stream not found for: %p", stream_handle);
  }

  int8_t expected = false;
  int64_t byte_size = sizeof(int8_t);
  CHECK_EQ(byte_size, ShapeUtil::ByteSizeOfPrimitiveType(PrimitiveType::PRED));
  stream->ThenMemcpy(
      &expected, se::DeviceMemoryBase{buffer, static_cast<uint64_t>(byte_size)},
      byte_size);
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  if (!static_cast<bool>(expected)) {
    return InternalError("%s", error_msg);
  }

  return Status::OK();
}

static void AssertionCustomCall(void* stream_handle, void** buffers,
                                const char* opaque, int opaque_len,
                                XlaCustomCallStatus* status) {
  Status s =
      AssertOnGpu(stream_handle, buffers[0],
                  absl::string_view{opaque, static_cast<uint64_t>(opaque_len)});
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(kXlaGpuAssertCustomCallTag,
                                         AssertionCustomCall, "CUDA");

}  // namespace xla
