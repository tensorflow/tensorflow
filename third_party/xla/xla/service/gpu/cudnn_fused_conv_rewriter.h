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

#ifndef XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
#define XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites custom-calls targeting cudnnConvolutionForward to
// cudnnConvolutionBiasActivationForward by fusing operations following forward
// convolution.  This transform must run after cudnn_conv_rewriter.
//
// Semantics of underlying cudnn ops:
//
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBiasActivationForward
// https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters
//
// ## Floating-point convs
//
// A "complete" fused floating-point conv has the form
//
//   max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias)),
//
// which we fuse to
//
//   cudnnConvolutionBiasActivationForward(x, w, bias, side_input).
//
// You can leave out side_input, bias, alpha1, alpha2, and max(x, 0) and still
// get a fused convolution.  alpha1/2 must be broadcasts of scalar constants.
//
// f16 convs accumulate in f32.  We represent this in HLO as an f32 convolution
// whose inputs can be converted to f16 without loss of precision and whose
// output is immediately converted to f16.  A fused f16 conv must follow one of
// the following idioms.
//
//   1. convert_f16(conv_f32(x_f32, w_f32)) +
//      side_input_f16 + broadcast(bias_f16)
//
//   2. convert_f16(conv_f32(x_f32, w_f32) +
//                  side_input_f32 + broadcast(bias_f32))
//
// (These are not strictly mathematically equivalent, but cudnn doesn't tell us
// which one it does, and we deem them "close enough".)
//
// The foo_f32 HLOs must all be losslessly-convertible to f16.  Some valid
// examples:
//
//   - foo_f32 = convert_f32(foo_f16)
//   - foo_f32 = an f32 constant whose values all fit within f16
//   - foo_f32 = broadcast/transpose/reshape(one of the above)
//
// If you have a relu, it can appear before or after the convert_f16.
//
// Note that here `bias` must be losslessly-convertible to f16; this is
// different than for s8 convolutions, where bias is f32.
//
// ## Integer convs
//
// In pure HLO, a "complete" integer conv is spelled as one of the following
// `result`s.
//
//   base = alpha1_f32 * convert_f32(conv_s32(input_s32, filter_s32)) +
//          alpha2_f32 * side_input +
//          bias_f32
//
//   result_f32        = max(result_f32, 0)
//   result_s8_option1 = max(convert_s8(clamp(-128, base, 127)), 0)
//   result_s8_option2 = convert_s8(clamp(-128, max(base, 0), 127))
//
// The foo_s32 HLOs must be losslessly-convertible to s8.  If the `result_s8`
// case, side_input should be an f32 HLO that's losslessly-convertible to s8;
// otherwise, it should be losslessly-convertible to f32.
//
// In the `result_s8` case where there's no bias, side-input, or alpha1, you can
// skip the convert_f32 on conv.
//
// If you have an integer convolution that doesn't fit one of these idioms, this
// pass returns an error -- cudnn will not be able to run it.
class CudnnFusedConvRewriter : public HloModulePass {
 public:
  explicit CudnnFusedConvRewriter(se::CudaComputeCapability cc)
      : compute_capability_(cc) {}

  absl::string_view name() const override {
    return "cudnn-fused-convolution-rewriter";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::CudaComputeCapability compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUDNN_FUSED_CONV_REWRITER_H_
