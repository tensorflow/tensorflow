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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

struct RunConvOptions {
  // Nullable output-parameter pointer for profiling results.
  se::dnn::ProfileResult* profile_result = nullptr;

  // Use this algorithm, instead of the one from the instruction.
  absl::optional<se::dnn::AlgorithmDesc> algo_override;
};

// Implementation struct exposed for debugging and log analysis.
struct GpuConvParams {
  // Here are the fields related to cuDNN's fused convolution. The result thus
  // is defined as:
  //   activation(conv_result_scale * conv(x, w) +
  //       side_input_scale * side_input + broadcast(bias))
  //
  // The most common fused conv is conv forward + relu/identity, for example.
  //
  // bias_buf is a single-dimensional array, with the length equal to the number
  // of output features. It'll be broadcasted to the output shape in order to be
  // added to the final results.
  //
  // side_input_buf, if valid, must have the same shape as the output buffer.
  struct FusionParams {
    se::dnn::ActivationMode mode;
    double side_input_scale;
    se::DeviceMemoryBase bias_buf;
    se::DeviceMemoryBase side_input_buf;  // nullable
  };

  CudnnConvKind kind;
  se::dnn::BatchDescriptor input_descriptor;
  se::dnn::FilterDescriptor filter_descriptor;
  se::dnn::BatchDescriptor output_descriptor;
  se::DeviceMemoryBase input_buf;
  se::DeviceMemoryBase filter_buf;
  se::DeviceMemoryBase output_buf;
  se::dnn::ConvolutionDescriptor conv_desc;
  se::dnn::AlgorithmConfig algorithm;
  double conv_result_scale;

  absl::optional<FusionParams> fusion;
};

// This file contains low-level routines for running cudnn convolutions.

// Calls into cudnn to run the specified convolution.
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
Status RunGpuConv(const HloCustomCallInstruction* conv,
                  absl::Span<se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  se::DeviceMemoryBase scratch_buf, se::Stream* stream,
                  RunConvOptions = {});

Status RunGpuConv(const HloCustomCallInstruction* conv,
                  absl::Span<se::DeviceMemoryBase> operand_buffers,
                  se::DeviceMemoryBase result_buffer,
                  se::ScratchAllocator* scratch_allocator, se::Stream* stream,
                  RunConvOptions = {});

// Implementation details exposed for debugging and log analysis.
StatusOr<GpuConvParams> GetGpuConvParams(
    const HloCustomCallInstruction* conv,
    absl::Span<se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_CONV_RUNNER_H_
