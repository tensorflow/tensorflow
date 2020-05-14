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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_HLO_TO_LHLO_WITH_XLA_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_HLO_TO_LHLO_WITH_XLA_H_

#include "mlir/IR/Module.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace mlir {

// Populate the MLIR `module` with the computation from the `hlo_module` using
// the provided buffer `assignment`. The returned `Status` indicates success
// or failure in the conversion.
tensorflow::Status HloToLhloModule(const xla::BufferAssignment& assignment,
                                   const xla::HloModule& hlo_module,
                                   ModuleOp module);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_HLO_TO_LHLO_WITH_XLA_H_
