/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_EVAL_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_EVAL_UTIL_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/c/eager/c_api.h"

namespace tensorflow {

// Attempts to evaluates an MLIR Operation in TensorFlow eager mode with the
// specified operands. The op is always executed on the local host CPU
// irrespective of the device attribute of the given op. If there is a CPU
// kernel registered for the op and is executed successfully, this fills in the
// results vector.  If not, results vector is unspecified.
//
mlir::LogicalResult EvaluateOperation(
    mlir::Operation* inst, llvm::ArrayRef<mlir::ElementsAttr> operands,
    TFE_Context* context, llvm::SmallVectorImpl<mlir::Attribute>* results);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_EVAL_UTIL_H_
