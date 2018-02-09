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

#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_runner.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

using se::dnn::AlgorithmDesc;
using se::dnn::BatchDescriptor;
using se::dnn::ConvolutionDescriptor;
using se::dnn::DataLayout;
using se::dnn::FilterDescriptor;
using se::dnn::FilterLayout;

ConvolutionThunk::ConvolutionThunk(
    CudnnConvKind convolution_kind, const BufferAllocation::Slice& input_buffer,
    const BufferAllocation::Slice& filter_buffer,
    const BufferAllocation::Slice& output_buffer,
    const BufferAllocation::Slice& tuple_result_buffer,
    const BufferAllocation::Slice& scratch_buffer, const Shape& input_shape,
    const Shape& filter_shape, const Shape& output_shape, const Window& window,
    const ConvolutionDimensionNumbers& dim_nums, int64 algorithm,
    const HloInstruction* hlo)
    : Thunk(Kind::kConvolution, hlo),
      convolution_kind_(convolution_kind),
      input_buffer_(input_buffer),
      filter_buffer_(filter_buffer),
      output_buffer_(output_buffer),
      tuple_result_buffer_(tuple_result_buffer),
      scratch_buffer_(scratch_buffer),
      input_shape_(input_shape),
      filter_shape_(filter_shape),
      output_shape_(output_shape),
      window_(window),
      dim_nums_(dim_nums),
      algorithm_(algorithm) {}

Status ConvolutionThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  se::DeviceMemory<float> input_data(
      buffer_allocations.GetDeviceAddress(input_buffer_));
  se::DeviceMemory<float> filter_data(
      buffer_allocations.GetDeviceAddress(filter_buffer_));
  se::DeviceMemory<float> output_data(
      buffer_allocations.GetDeviceAddress(output_buffer_));
  se::DeviceMemoryBase scratch =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  se::dnn::AlgorithmConfig algorithm_config(
      se::dnn::AlgorithmDesc(algorithm_, /*use_tensor_ops=*/false));

  TF_RETURN_IF_ERROR(RunCudnnConvolution(
      convolution_kind_, input_shape_, filter_shape_, output_shape_, input_data,
      filter_data, output_data, scratch, window_, dim_nums_, algorithm_config,
      stream));

  // Figure out which of output/input/filter is the result produced by this op,
  // and write the result tuple.
  void* result_ptr = [&] {
    switch (convolution_kind_) {
      case CudnnConvKind::kForward:
        return output_data.opaque();
      case CudnnConvKind::kBackwardInput:
        return input_data.opaque();
      case CudnnConvKind::kBackwardFilter:
        return filter_data.opaque();
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
