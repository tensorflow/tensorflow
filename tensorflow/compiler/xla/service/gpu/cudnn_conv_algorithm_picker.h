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
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_runner.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// Modifies CustomCalls to cudnn convolutions, choosing the best algorithm for
// each and adding explicit scratch space to the CustomCalls.
class CudnnConvAlgorithmPicker : public HloModulePass {
 public:
  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  CudnnConvAlgorithmPicker(se::StreamExecutor* stream_exec,
                           DeviceMemoryAllocator* allocator, Compiler* compiler)
      : stream_exec_(stream_exec), allocator_(allocator), compiler_(compiler) {}

  absl::string_view name() const override {
    return "cudnn-conv-algorithm-picker";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  struct AutotuneResult {
    int64 algorithm;
    bool tensor_ops_enabled;
    int64 scratch_bytes;
    absl::Duration runtime;
  };

  StatusOr<bool> RunOnComputation(HloComputation* computation);
  StatusOr<bool> RunOnInstruction(HloInstruction* instr);
  StatusOr<AutotuneResult> PickBestAlgorithm(
      const HloCustomCallInstruction* instr);

  se::StreamExecutor* stream_exec_;                   // never null
  DeviceMemoryAllocator* allocator_;                  // may be null
  Compiler* compiler_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_ALGORITHM_PICKER_H_
