#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_REWRITER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_REWRITER_H_

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

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites BatchNorm HLOs into calls into cudnn where possible.
//
// A call into cudnn for performing a batchnorm op is represented as a
// CustomCall HLO with custom_call_target equal to one of
//
//   - kCudnnBatchNormForwardInferenceCallTarget
//   - kCudnnBatchNormForwardTrainingCallTarget, or
//   - kCudnnBatchNormBackwardCallTarget.
//
// A CustomCall created by this pass has the same operands corresponding
// batchnorm HLO, except the epsilon() and feature_index() properties of the
// batchnorm HLO are converted into proper operands, added to the end of the
// CustomCall's operands list.
//
// The inputs/outputs of the cudnn calls for BatchNormTraining and BatchNormGrad
// do not correspond exactly to the HLOs.  In particular, the training cudnn
// call returns 1/sqrt(variance + epsilon), while the HLO returns plain
// variance.  Similarly, the grad cudnn call expects 1/sqrt(variance + epsilon)
// as input, whereas the HLO expects plain variance.
//
// This pass adds HLOs in front of / behind the CustomCalls to fix up the
// inputs/outputs as appropriate, and we rely on the AlgebraicSimplifier to
// remove these where possible.
//
// Currently batchnorm ops over F32s are converted into cudnn calls, so long as
// epsilon is not too small.  This pass leaves other batchnorm ops unmodified.
//
// The GPU backend does not implement a lowering for the batchnorm HLOs -- it
// expects them to be lowered to cudnn calls via this pass or to HLO soup via
// BatchNormRewriter.
class CudnnBatchNormRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "cudnn_batchnorm_rewriter"; }
  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUDNN_BATCHNORM_REWRITER_H_
