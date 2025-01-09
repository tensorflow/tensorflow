/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_PASSES_TD_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_PASSES_TD_

include "mlir/Pass/PassBase.td"

def GeneralizeKernelSignaturePass
    : Pass<"generalize-kernel-signature"> {
  let summary = "Rewrite kernels to use generic data pointer arguments.";
  let description = [{
    Rewrite signatures of kernel functions from global pointers to generic
    pointers and cast them to global ones within the kernel.
  }];
  let constructor = "CreateGeneralizeKernelSignaturePass()";
}

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_PASSES_TD_
