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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_ALGORITHM_PICKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_ALGORITHM_PICKER_H_

#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/stream_executor/cuda/redzone_allocator.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace gpu {

// Modifies CustomCalls to cudnn convolutions, choosing the best algorithm for
// each and adding explicit scratch space to the CustomCalls.
class CudnnConvAlgorithmPicker : public GpuConvAlgorithmPicker {
 public:
  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  CudnnConvAlgorithmPicker(se::StreamExecutor* stream_exec,
                           se::DeviceMemoryAllocator* allocator)
      : GpuConvAlgorithmPicker(stream_exec, allocator) {}

  absl::string_view name() const override {
    return "cudnn-conv-algorithm-picker";
  }

 protected:
  StatusOr<tensorflow::AutotuneResult> PickBestAlgorithmNoCache(
      const HloCustomCallInstruction& instr,
      se::DeviceMemoryAllocator* allocator, se::Stream* stream);

  Status AllocateInitializeBuffers(
      const HloCustomCallInstruction& instr,
      se::ScratchAllocator* input_output_allocator, se::Stream* stream,
      std::vector<se::DeviceMemoryBase>* operand_buffers,
      se::DeviceMemoryBase* result_buffer);

  Status ProfileConvCandidates(
      const HloCustomCallInstruction& instr, se::Stream* stream,
      se::cuda::RedzoneAllocator* input_output_allocator,
      se::cuda::RedzoneAllocator* scratch_allocator,
      std::vector<se::DeviceMemoryBase>* operand_buffers,
      se::DeviceMemoryBase* result_buffer,
      std::vector<tensorflow::AutotuneResult>* profile_results,
      bool* crash_on_checking_failure);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_ALGORITHM_PICKER_H_
