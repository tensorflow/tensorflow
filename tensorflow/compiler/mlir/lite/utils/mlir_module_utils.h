/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_MLIR_MODULE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_MLIR_MODULE_UTILS_H_

#include <stdlib.h>

#include <cstdint>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/const_tensor_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
namespace TFL {

// This function estimates the size of the module in mega bytes. It does so by
// iterating through all the constant-like attributes and tensors in the module
// and summing up their sizes.
//
// This function is used to reserve space in the buffer before serializing the
// module to avoid reallocating the buffer during serialization.
//
// This function may need to be improved to give more accurate size of the
// module if the current estimate is not good enough and causes huge
// reallocations during serialization.
inline uint64_t GetApproximateModuleSize(mlir::ModuleOp module) {
  uint64_t module_size_estimate = 0;
  mlir::DenseSet<mlir::Attribute> unique_tensors;

  for (auto global_tensor_op :
       module.getOps<mlir::tf_saved_model::GlobalTensorOp>()) {
    mlir::ElementsAttr elements_attr = global_tensor_op.getValueAttr();
    uint64_t tensor_size =
        mlir::TFL::GetSizeInBytes(global_tensor_op.getType());
    unique_tensors.insert(elements_attr);
    module_size_estimate += tensor_size;
  }

  module.walk([&](Operation* op) {
    mlir::ElementsAttr attr;
    if (mlir::detail::constant_op_binder<mlir::ElementsAttr>(&attr).match(op)) {
      // If the tensor hasn't been seen before
      if (!unique_tensors.contains(attr)) {
        uint64_t tensor_size =
            mlir::TFL::GetSizeInBytes(op->getResult(0).getType());
        unique_tensors.insert(attr);  // Store the size in the map
        module_size_estimate += tensor_size;
      }
    }
  });
  return module_size_estimate;
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_MLIR_MODULE_UTILS_H_
