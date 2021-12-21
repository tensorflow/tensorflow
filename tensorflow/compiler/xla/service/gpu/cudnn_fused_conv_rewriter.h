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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_

#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrite the custom call targeting cudnnConvolutionForward to
// cudnnConvolutionBiasActivationForward by fusing applicable point-wise
// operations following forward convolution.  This transform must run after
// cudnn_conv_rewriter.
// It is straightforward for floating point convolutions:
// transforming
//   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias))
// to
//   cudnnConvolutionBiasActivationForward(x, w, bias, alpha1, alpha2, side)
//
// Integer convolution requires additional patterns to match CuDNN semantics:
//   #1 from
//   cast<int8_t>(clamp<-128, 127>(conv(int8_x, int8_w)))
//   to
//   cudnnConvolutionForward<int8_t>(int8_x, int8_w)
// or #2 from
//   cast<float>(conv(int8_x, int8_w))
//   to
//   cudnnConvolutionForward<float>(int8_x, int8_w)
// or #3 from
//   cast<int8_t>(clamp<-128, 127>(max(0, alpha1 *
//                           cast<float>(conv(int8_x, int8_w)) +
//                           alpha2 * cast<float>(int8_side) +
//                           broadcast(bias)))
//   to
//   cudnnConvolutionBiasActivationForward<int8_t>(int8_x, int8_w, bias, alpha1,
//   alpha2, int8_side)
// or #4 from
//   max(0, alpha1 * cast<float>(conv(int8_x, int8_w)) +
//          alpha2 * float_side + broadcast(bias))
//   to
//   cudnnConvolutionBiasActivationForward<float>(int8_x, int8_w, bias, alpha1,
//   alpha2, float_side)

class CudnnFusedConvRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "cudnn-fused-convolution-rewriter";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
