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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONVOLUTION_ALGORITHM_PICKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONVOLUTION_ALGORITHM_PICKER_H_

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_runner.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// Modifies CustomCalls to cudnn convolutions, choosing the best algorithm for
// each and adding explicit scratch space to the CustomCalls.
class CudnnConvolutionAlgorithmPicker : public HloPassInterface {
 public:
  // If the `allocator` parameter is not null, we will use it to allocate temp
  // memory while timing the various convolution algorithms.  If it's null,
  // we'll use the default allocator on the StreamExecutor.
  CudnnConvolutionAlgorithmPicker(se::StreamExecutor* stream_exec,
                                  DeviceMemoryAllocator* allocator)
      : stream_exec_(stream_exec), allocator_(allocator) {}

  tensorflow::StringPiece name() const override {
    return "cudnn-convolution-algorithm-picker";
  }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  StatusOr<bool> RunOnComputation(HloComputation* computation);
  StatusOr<bool> RunOnInstruction(HloInstruction* instr);
  tensorflow::gtl::optional<std::tuple<int64, bool, int64>> PickBestAlgorithm(
      CudnnConvKind kind, const Shape& input_shape, const Shape& filter_shape,
      const Shape& output_shape, const Window& window,
      const ConvolutionDimensionNumbers& dnums, HloInstruction* instr);

  se::StreamExecutor* stream_exec_;                   // never null
  DeviceMemoryAllocator* allocator_;                  // may be null
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONVOLUTION_ALGORITHM_PICKER_H_
