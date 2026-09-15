/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

namespace tensorflow {
namespace metal {


void RegisterAllMetalKernels() {
  RegisterMetalActivationKernels();
  RegisterMetalBatchSpaceKernels();
  RegisterMetalAliasKernels();
  RegisterMetalArrayKernels();
  RegisterMetalBatchNormKernels();
  RegisterMetalCompareKernels();
  RegisterMetalConvKernels();
  RegisterMetalConv3DKernels();
  RegisterMetalDepthwiseKernels();
  RegisterMetalDilationKernels();
  RegisterMetalBincountKernels();
  RegisterMetalBatchNormGlobalKernels();
  RegisterMetalResizeGradKernels();
  RegisterMetalVolumePatchKernels();
  RegisterMetalSparseKernels();
  RegisterMetalDebugKernels();
  RegisterMetalCollectiveKernels();
  RegisterMetalCudnnRnnKernels();
  RegisterMetalGatherNdKernels();
  if (ResourceVariableApiAvailable()) {
    RegisterMetalRefVariableKernels();
  }
  RegisterMetalMisc2Kernels();
  RegisterMetalSparseManipKernels();
  RegisterMetalFusedKernels();
  RegisterMetalFftKernels();
  RegisterMetalSparseSegmentKernels();
  RegisterMetalBoxProposalKernels();
  RegisterMetalCtcKernels();
  RegisterMetalGenericConvKernels();
  RegisterMetalNmsKernels();
  RegisterMetalLinalgKernels();
  if (ResourceVariableApiAvailable()) {
    RegisterMetalInplaceKernels();
  }
  RegisterMetalDynamicKernels();
  RegisterMetalRnnKernels();
  RegisterMetalRandomDistKernels();
  RegisterMetalCropResizeKernels();
  RegisterMetalTransformKernels();
  RegisterMetalQuantizeDequantizeKernels();
  RegisterMetalMaxPoolArgmaxKernels();
  RegisterMetalImageKernels();
  RegisterMetalImage2Kernels();
  RegisterMetalIndexKernels();
  RegisterMetalMatrixKernels();
  RegisterMetalSearchKernels();
  RegisterMetalSliceKernels();
  RegisterMetalStridedKernels();
  RegisterMetalReductionKernels();
  RegisterMetalElementwiseKernels();
  RegisterMetalFillKernels();
  RegisterMetalIdentityKernels();
  RegisterMetalMatMulKernels();
  RegisterMetalNnKernels();
  RegisterMetalPoolingKernels();
  RegisterMetalPoolVariantKernels();
  RegisterMetalExtraKernels();
  RegisterMetalMiscKernels();
  RegisterMetalQuantKernels();
  RegisterMetalRandomKernels();
  // The resource variable ops and the optimisers reach a variable through
  // kernel C API entry points a released TensorFlow no longer exports. Without
  // them these would be registered and take the process down at the first
  // call, so they are left to the host instead.
  if (ResourceVariableApiAvailable()) {
    RegisterMetalResourceKernels();
    RegisterMetalTrainingKernels();
  }
}

}  // namespace metal
}  // namespace tensorflow
