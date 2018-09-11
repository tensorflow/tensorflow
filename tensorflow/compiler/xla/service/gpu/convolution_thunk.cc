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

#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_runner.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

using se::dnn::AlgorithmDesc;

Status ConvolutionThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  CudnnConvParams params;

  params.input_buf = buffer_allocations.GetDeviceAddress(input_buffer_);
  params.filter_buf = buffer_allocations.GetDeviceAddress(filter_buffer_);
  params.output_buf = buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  TF_RETURN_IF_ERROR(PopulateCudnnConvParams(cudnn_call_, &params));

  auto op_profiler = profiler->MakeScopedInstructionProfiler(hlo_instruction());
  TF_RETURN_IF_ERROR(RunCudnnConvolution(params, scratch, stream));

  // Figure out which of output/input/filter is the result produced by
  // this op, and write the result tuple.
  void* result_ptr = [&] {
    switch (params.kind) {
      case CudnnConvKind::kForward:
        return params.output_buf.opaque();
      case CudnnConvKind::kBackwardInput:
        return params.input_buf.opaque();
      case CudnnConvKind::kBackwardFilter:
        return params.filter_buf.opaque();
    }
  }();
  void* ptrs[] = {result_ptr, scratch.opaque()};
  se::DeviceMemory<void*> tuple_addr(
      buffer_allocations.GetDeviceAddress(tuple_result_buffer_));
  stream->ThenMemcpyH2D<void*>(ptrs, &tuple_addr);

  if (!stream->ok()) {
    return InternalError("ConvolutionThunk::ExecuteOnStream failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
