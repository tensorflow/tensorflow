/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_PREPARE_TF_PASS_OPTIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_PREPARE_TF_PASS_OPTIONS_H_

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassOptions.h"  // from @llvm-project

namespace mlir {
namespace TFL {

////////////////////////////////////////////////////////////////////////////////
// Pass Options
////////////////////////////////////////////////////////////////////////////////

struct PrepareTFPassOptions : public mlir::detail::PassOptions {
  mlir::detail::PassOptions::Option<bool> unfold_batch_matmul{
      *this, "unfold_batchmatmul",
      ::llvm::cl::desc("Unfold BatchMatMul into individual MatMul ops."),
      ::llvm::cl::init(true)};
  mlir::detail::PassOptions::Option<bool> allow_bf16_and_f16_type_legalization{
      *this, "allow-bf16-and-f16-type-legalization",
      ::llvm::cl::desc("Allow bf16 type legalization."),
      ::llvm::cl::init(false)};
  mlir::detail::PassOptions::Option<bool> use_fake_quant_num_bits{
      *this, "use-fake-quant-num-bits",
      ::llvm::cl::desc(
          "Use quantization calculated from fake quant attributes."),
      ::llvm::cl::init(false)};
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TF_LEGALIZATIONS_PREPARE_TF_PASS_OPTIONS_H_
