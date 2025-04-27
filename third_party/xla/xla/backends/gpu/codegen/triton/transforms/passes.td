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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_

include "mlir/Pass/PassBase.td"

def TritonXLAExtractInsertToTritonPass : Pass<"triton-xla-extract-insert-to-triton", "mlir::ModuleOp"> {
  let summary = "Convert Triton XLA extract and insert ops to Triton ops.";
  let description = [{
    This pass converts `triton_xla.extract` and `triton_xla.insert` ops to
    Triton ops. It also rewrites `func` args to `tt.ptr` types and removes
    function return args.
  }];
  let dependentDialects = [
    "triton::TritonDialect",
    "::xla::XlaDialect"
  ];
  let options = [
    Option<"gpu_device_info_", "gpu_device_info", "std::string", /*default=*/"",
           "Serialized stream_executor::GPUDeviceInfo proto">,
    Option<"tma_enabled_", "tma_enabled", "bool", /*default=*/"false",
           "Flag to enable/disable TMA">,
  ];
  let constructor = "CreateTritonXLAExtractInsertToTritonPass()";
}

def GeneralizeKernelSignaturePass
    : Pass<"generalize-kernel-signature"> {
  let summary = "Rewrite kernels to use generic data pointer arguments.";
  let description = [{
    Rewrite signatures of kernel functions from global pointers to generic
    pointers and cast them to global ones within the kernel.
  }];
  let constructor = "CreateGeneralizeKernelSignaturePass()";
}

def PreventMmaV3LoopUnrollingPass
    : Pass<"prevent-mmav3-loop-unrolling", "mlir::ModuleOp"> {
  let summary = "Prevent MMAv3 loop unrolling.";
  let description = [{
    This pass is a result of b/344841434:
    PTX sometimes unrolls wgmma loops that can cause a 1000x slow down in
    compilation time. Most unrolling has already been done before PTX,
    this pragma prevents ptxas from doing more.
  }];
  let constructor = "CreatePreventMmaV3LoopUnrollingPass()";
}

def LoadInt4RewritePass
    : Pass<"int4-to-packed-int4-rewrite", "mlir::ModuleOp"> {
  let summary = "Converts ops with int4 tensors to the ops with int4 packed to int8 tensors.";
  let description = [{
    This pass replaces the int4 tensors with the int4 packed to int8 tensor of
    the twice smaller size. It also replaces the plain ExtSIOp upcast to the
    int8 tensor with the unpack sequence.
  }];
  let constructor = "CreateInt4ToPackedInt4RewritePass()";
}

def RoundF32ToTF32ForTf32DotRewritePass
    : Pass<"round-f32-to-tf32-for-tf32-dot-rewrite", "mlir::ModuleOp"> {
  let summary = "dot with tf32 algorithm requires explicit rounding.";
  let description = [{
    This pass adds explicit rounding from f32 to tf32 for the dot with tf32 algorithm.
    This is required because mma instruction does not have explicit rounding and
    by default does truncation. As a result, the dot with tf32 algorithm has too
    small precision. It is even less than for the dot with BF16 arguments.
  }];
  let constructor = "CreateRoundF32ToTF32ForTf32DotRewritePass()";
}

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_PASSES_TD_
