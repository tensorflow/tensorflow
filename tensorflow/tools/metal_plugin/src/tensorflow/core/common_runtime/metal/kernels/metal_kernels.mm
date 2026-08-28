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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#if defined(TF_METAL_OUT_OF_TREE)
#include <dlfcn.h>

#include "absl/log/log.h"
#endif  // TF_METAL_OUT_OF_TREE

namespace tensorflow {
namespace metal {

#if defined(TF_METAL_OUT_OF_TREE)
namespace {

// Four kernel families reach resource and reference variables through
// tensorflow/c/kernels_experimental.h. Those entry points are declared in the
// headers a released TensorFlow ships but are not exported by any binary in
// it, so a plugin loaded into a released TensorFlow cannot call them: the
// reference would resolve to nothing at the first call and take the process
// with it.
//
// An in-tree build links them directly and never asks this question. Out of
// tree the answer decides whether those families are registered at all, which
// is better than registering kernels that crash when a graph reaches them.
bool ResourceVariableApiAvailable() {
  static const bool available = [] {
    static constexpr const char* kRequired[] = {
        "TF_AssignRefVariable",
        "TF_GetInputTensorFromVariable",
        "TF_MaybeLockVariableInputMutexesInOrder",
        "TF_ReleaseVariableInputLockHolder",
        "TF_OpKernelConstruction_GetAttrTensorShape",
        "TF_OpKernelContext_ForwardRefInputToRefOutput",
    };
    for (const char* name : kRequired) {
      if (dlsym(RTLD_DEFAULT, name) == nullptr) {
        LOG(WARNING) << "Metal: this TensorFlow does not export " << name
                     << ", so the optimisers, the resource gather and scatter "
                        "ops, the reference assignments and ParallelConcat are "
                        "left to the host. Training on the GPU needs a "
                        "TensorFlow that exports the experimental kernel API.";
        return false;
      }
    }
    return true;
  }();
  return available;
}

}  // namespace
#endif  // TF_METAL_OUT_OF_TREE

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
#if defined(TF_METAL_OUT_OF_TREE)
  if (ResourceVariableApiAvailable()) {
    RegisterMetalRefVariableKernels();
  }
#else
  RegisterMetalRefVariableKernels();
#endif
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
#if defined(TF_METAL_OUT_OF_TREE)
  if (ResourceVariableApiAvailable()) {
    RegisterMetalInplaceKernels();
  }
#else
  RegisterMetalInplaceKernels();
#endif
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
#if defined(TF_METAL_OUT_OF_TREE)
  if (ResourceVariableApiAvailable()) {
    RegisterMetalResourceKernels();
  }
#else
  RegisterMetalResourceKernels();
#endif
#if defined(TF_METAL_OUT_OF_TREE)
  if (ResourceVariableApiAvailable()) {
    RegisterMetalTrainingKernels();
  }
#else
  RegisterMetalTrainingKernels();
#endif
}

}  // namespace metal
}  // namespace tensorflow
