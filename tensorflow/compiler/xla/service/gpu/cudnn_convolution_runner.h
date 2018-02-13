/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONVOLUTION_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONVOLUTION_RUNNER_H_

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// This file contains low-level routines for running cudnn convolutions.

// Different types of convolutions supported by cudnn.
//
// A way to think about these is that a convolution is defined by three arrays
// -- the "input", the "filter", and the "output" -- and given any two of these,
// we can compute the third.  For example, a backward-input convolution takes as
// input a filter and an "output" and produces an "input" such that if one were
// to do a forward convolution of "input" using filter, the result would be
// something with the same shape as "output".
//
// This way of thinking is not correct if you look at the values produced. For
// example, a backward-input convolution is not actually the mathematical
// inverse of a forward convolution.  But it's right as far as the shapes and
// "connectivity" (i.e. which elements of the input affect which elements of
// the output) are concerned.
enum class CudnnConvKind {
  kForward,         // input  + filter => output
  kBackwardInput,   // filter + output => input
  kBackwardFilter,  // input  + output => filter
};

// Converts a CudnnConvKind value to a string.
string CudnnConvKindToString(CudnnConvKind kind);

// Calls into cudnn to run the specified convolution.
//
// Note that depending on the value of CudnnConvKind, the result of this call
// may be written into input_buf, filter_buf, or output_buf!
//
// At the moment we only support cudnn convolutions over floats.
//
// We provide one overload which takes a scratch buffer, and another which takes
// an allocator which is responsible for allocating the scratch space.  In
// theory the second one shouldn't be necessary -- users of this function could
// just ask cudnn how much scratch space it needs for a particular convolution.
// But in practice, StreamExecutor does not expose such an API, and in the name
// of parsimony, perhaps it's better not to add it.  Instead, the first time you
// call a convolution, you should call the version that takes a scratch
// allocator and take note of how much memory is used.  The next time you call
// the same conv, you can provide an explicitly preallocated scratch buffer of
// that size, if you like.
Status RunCudnnConvolution(
    CudnnConvKind kind, const Shape& input_shape, const Shape& filter_shape,
    const Shape& output_shape,
    perftools::gputools::DeviceMemory<float> input_buf,
    perftools::gputools::DeviceMemory<float> filter_buf,
    perftools::gputools::DeviceMemory<float> output_buf,
    perftools::gputools::DeviceMemoryBase scratch_buf, const Window& window,
    const ConvolutionDimensionNumbers& dnums,
    perftools::gputools::dnn::AlgorithmConfig algorithm,
    perftools::gputools::Stream* stream,
    perftools::gputools::dnn::ProfileResult* profile_result = nullptr);

Status RunCudnnConvolution(
    CudnnConvKind kind, const Shape& input_shape, const Shape& filter_shape,
    const Shape& output_shape,
    perftools::gputools::DeviceMemory<float> input_buf,
    perftools::gputools::DeviceMemory<float> filter_buf,
    perftools::gputools::DeviceMemory<float> output_buf,
    perftools::gputools::ScratchAllocator* scratch_allocator,
    const Window& window, const ConvolutionDimensionNumbers& dnums,
    perftools::gputools::dnn::AlgorithmConfig algorithm,
    perftools::gputools::Stream* stream,
    perftools::gputools::dnn::ProfileResult* profile_result = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONVOLUTION_RUNNER_H_
