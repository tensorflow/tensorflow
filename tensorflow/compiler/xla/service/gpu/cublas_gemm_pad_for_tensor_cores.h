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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_GEMM_PAD_FOR_TENSOR_CORES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_GEMM_PAD_FOR_TENSOR_CORES_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Adds padding to dot operations to make them run faster on GPUs with
// tensor cores (https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/).
//
// f16 dots are padded to have input/output shapes with dimensions that
// are multiples of 8, so that we can use tensor cores.
//
// Don't run this pass on GPUs without tensor cores -- it will make them slower!
//
// This pass depends on xla::DotDecomposer pass,
// so it should go strictly later.
class CublasGemmPadForTensorCores : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "cublas-gemm-pad-for-speed";
  }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_GEMM_PAD_FOR_TENSOR_CORES_H_
