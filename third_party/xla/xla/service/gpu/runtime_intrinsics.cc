/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime_intrinsics.h"

#include <cstdint>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

std::string GetGpuPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

absl::Status AssertOnGpu(void* stream_handle, void* buffer,
                         absl::string_view error_msg) {
  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(GetGpuPlatformName()));
  se::StreamExecutorConfig config;
  config.gpu_stream = stream_handle;
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      platform->GetExecutor(config));
  se::Stream* stream = executor->FindAllocatedStream(stream_handle);
  if (!stream) {
    return Internal("Stream not found for: %p", stream_handle);
  }

  int8_t expected = false;
  int64_t byte_size = sizeof(int8_t);
  CHECK_EQ(byte_size, ShapeUtil::ByteSizeOfPrimitiveType(PrimitiveType::PRED));
  TF_RETURN_IF_ERROR(stream->Memcpy(
      &expected, se::DeviceMemoryBase{buffer, static_cast<uint64_t>(byte_size)},
      byte_size));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  if (!static_cast<bool>(expected)) {
    return Internal("%s", error_msg);
  }

  return absl::OkStatus();
}

void AssertionCustomCall(void* stream_handle, void** buffers,
                         const char* opaque, int opaque_len,
                         XlaCustomCallStatus* status) {
  absl::Status s =
      AssertOnGpu(stream_handle, buffers[0],
                  absl::string_view{opaque, static_cast<uint64_t>(opaque_len)});
  if (!s.ok()) {
    auto msg = s.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.size());
  }
}

void NopReturnTokenCustomCall(void* stream_handle, void** buffers,
                              const char* opaque, int opaque_len,
                              XlaCustomCallStatus* status) {
  VLOG(1) << "NopReturnTokenCustomCall called.";
}

}  // namespace

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    std::string(kXlaGpuAssertCustomCallTag), AssertionCustomCall,
    GetGpuPlatformName());

// This allows measuring exported HLOs where kOutfeed and kSendDone has been
// replaced with NopReturnToken. In that case the runtime of the original
// kOutfeed and kSendDone operations is not measured.
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    std::string(kNopReturnTokenCustomCallTarget), NopReturnTokenCustomCall,
    GetGpuPlatformName());

}  // namespace xla
