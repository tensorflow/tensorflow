/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_PASSES_TD_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

def FusionOpRemoverPass : FunctionPass<"mlir-gpu-fusion-op-remover"> {
  let summary = "Removes lhlo fusion ops by inlining their regions.";
  let constructor = "createFusionOpRemoverPass()";
  let description = [{
    Replaces a FusionOp by the operations contained in its region.
  }];
}

def StoreForwardingPass : FunctionPass<"mlir-gpu-store-forwarding"> {
  let summary = "Limited pass to forward stores to loads.";
  let constructor = "createStoreForwardingPass()";
  let description = [{
    Replaces a load that immediately follows a store to the same address with
    the stored value.
  }];
}

def DeadTempBufferRemovalPass
    : FunctionPass<"mlir-gpu-dead-temp-buffer-removal"> {
  let summary = "Removal of dead temp buffers.";
  let constructor = "createDeadTempBufferRemovalPass()";
  let description = [{
    Removes temporary buffers that are only written to but never read from or
    that are read but the read value is not used. Needs an analysis that proves
    that loads and stores are side-effect free (in bounds, no aliasing, etc.).
  }];
}

def RewriteKernelSignaturePass
    : FunctionPass<"mlir-gpu-rewrite-signatures"> {
  let summary = "Rewrite kernel signatures to be deterministic.";
  let constructor = "createRewriteKernelSignaturePass()";
  let description = [{
    Sorts the operands to the kernel for a deterministic order. First operands
    that are defined by function arguments, followed by operands that are
    returned from the function. This only works for simple functions without
    control flow and can be used in cases where the kernel is extracted and used
    independently of the host-side code.
  }];
}

def MapParallelLoopsPass
    : FunctionPass<"mlir-gpu-map-parallel-loops"> {
  let summary = "Greedily maps loops to GPU hardware dimensions.";
  let constructor = "createMapParallelLoopsPass()";
  let description = [{
    Greedily maps loops to GPU hardware dimensions.
  }];
}

def FuseInnerParallelLoopsPass
    : FunctionPass<"mlir-gpu-fuse-inner-parallel-loops"> {
  let summary = "Limited pass to forward stores to loads.";
  let constructor = "createFuseInnerParallelLoopsPass()";
  let description = [{
    Directs parallel loop fusion to the inner loops. This cannot be done with
    a passmanager alone ATM, as nested pass managers require operations to
    be closed from above.
  }];
}

def ParallelLoopCollapsingToFirstDimPass
    : FunctionPass<"mlir-gpu-collapse-parallel-loops"> {
  let summary = "Collaps n-dimensional loops into one-dimensional ones.";
  let constructor = "createParallelLoopCollapsingToFirstDimPass()";
  let description = [{
    Collapses all loop dimension of a parallel loop into the first one.
  }];
}

#endif // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_PASSES_TD_
