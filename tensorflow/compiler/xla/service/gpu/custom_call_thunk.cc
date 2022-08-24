/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/custom_call_thunk.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/errors.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#endif

namespace xla {
namespace gpu {

CustomCallThunk::CustomCallThunk(ThunkInfo thunk_info,
                                 CustomCallTarget call_target,
                                 std::vector<OptionalSlice> operands,
                                 std::vector<OptionalSlice> results,
                                 const std::string& opaque)
    : Thunk(Thunk::kCustomCall, thunk_info),
      call_target_(std::move(call_target)),
      operands_(std::move(operands)),
      results_(std::move(results)),
      opaque_(opaque) {}

Status CustomCallThunk::ExecuteOnStream(const ExecuteParams& params) {
  // gpu_stream is CUstream or e.g. the equivalent type in ROCm.
  std::vector<void*> buffers;
  buffers.reserve(operands_.size() + results_.size());
  for (const std::vector<OptionalSlice>& slices : {operands_, results_}) {
    for (const OptionalSlice& slice : slices) {
      if (slice) {
        if (!slice->allocation())
          return InternalError("custom call input missing buffer allocation");
        buffers.push_back(
            params.buffer_allocations->GetDeviceAddress(*slice).opaque());
      } else {
        buffers.push_back(nullptr);
      }
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  auto gpu_stream = se::gpu::AsGpuStreamValue(params.stream);
  XlaCustomCallStatus custom_call_status;
  call_target_(gpu_stream, buffers.data(), opaque_.data(), opaque_.size(),
               &custom_call_status);
  auto message = CustomCallStatusGetMessage(&custom_call_status);
  if (message) {
    return InternalError("CustomCall failed: %s", *message);
  } else {
    return OkStatus();
  }
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return Unavailable(
      "Custom calls on GPU are not supported in this configuration. Please "
      "build with --config=cuda or --config=rocm");
#endif  //   GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace gpu
}  // namespace xla
