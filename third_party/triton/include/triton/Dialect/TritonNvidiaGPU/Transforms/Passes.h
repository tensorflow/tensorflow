/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

// Used by Triton runtime
struct ClusterInfo {
  ClusterInfo() : clusterDimX(1), clusterDimY(1), clusterDimZ(1) {}
  int clusterDimX;
  int clusterDimY;
  int clusterDimZ;
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

namespace mlir {

std::unique_ptr<Pass> createTritonNvidiaGPUPlanCTAPass(
    mlir::triton::nvidia_gpu::ClusterInfo *clusterInfo = nullptr);

std::unique_ptr<Pass>
createTritonNvidiaGPUFenceInsertionPass(int computeCapability = 90);

std::unique_ptr<Pass> createTritonNvidiaGPUTMALoweringPass();

std::unique_ptr<Pass> createTensorMemoryAllocationPass();

std::unique_ptr<Pass> createTritonNvidiaGPUMMALoweringPass();

std::unique_ptr<Pass> createTritonNvidiaGPUKeepAccInTMemPass();

std::unique_ptr<Pass> createTritonNvidiaGPUPromoteLHSToTMemPass();

std::unique_ptr<Pass> createTritonNvidiaGPUOptimizeDescriptorEncodingPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_TRITONNVIDIAGPULEGALIZETMALAYOUTS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif // TRITON_DIALECT_TRITONNVIDIAGPU_TRANSFORMS_PASSES_H_
