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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_PAD_FOR_TENSOR_CORES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_PAD_FOR_TENSOR_CORES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Adds padding to cudnn convolutions to make them run faster on GPUs with
// tensor cores.
//
//  - f16 convolutions are padded to have input/output channel dimensions that
//    are multiples of 8, so that we can use tensor cores.
//
//  - f16 convolutions with 3 input channels and 32 or 64 output channels are
//    padded to 4 input channels.  There's a special-cased cudnn algorithm just
//    for this.
//
// Don't run this pass on GPUs without tensor cores -- it will make them slower!
//
// TODO(jlebar): Also pad dots.
class CudnnConvPadForTensorCores : public HloModulePass {
 public:
  absl::string_view name() const override { return "cudnn-conv-pad-for-speed"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_CONV_PAD_FOR_TENSOR_CORES_H_
