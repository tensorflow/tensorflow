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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_RUNNER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_RUNNER_H_

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

struct CudnnBatchNormConfig {
  Shape output_shape;
  PrimitiveType output_type;
  float epsilon;
  int64 feature_index;
};

CudnnBatchNormConfig GetCudnnBatchNormConfig(const HloInstruction *instr,
                                             float epsilon,
                                             int64_t feature_index);

Status RunCudnnBatchNormForwardInference(
    const CudnnBatchNormConfig &config, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, se::DeviceMemory<float> mean,
    se::DeviceMemory<float> variance, se::Stream *stream);

Status RunCudnnBatchNormForwardTraining(
    const CudnnBatchNormConfig &config, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_data, se::DeviceMemory<float> output_mean,
    se::DeviceMemory<float> output_inv_stddev, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, se::Stream *stream);

Status RunCudnnBatchNormBackward(
    const CudnnBatchNormConfig &config, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_grad_data, se::DeviceMemoryBase grad_output,
    se::DeviceMemory<float> output_grad_scale,
    se::DeviceMemory<float> output_grad_offset, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> mean, se::DeviceMemory<float> inv_stddev,
    se::Stream *stream);
}  // namespace gpu
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_RUNNER_H_
