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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_

#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"

namespace tensorflow {

extern const char* const kXlaShardingAttrName;
extern const char* const kInputShardingAttr;
extern const char* const kOutputShardingAttr;

// Parse "_XlaSharding" attribute from operation, if it exists.
llvm::Optional<mlir::StringRef> ParseShardingAttribute(
    mlir::Operation* operation);

// Parses "input_sharding_configuration" attribute and returns a list where
// i-th element is a list of mlir::Value's which represent inputs for the
// TPU computation correponding to i-th logical device. If the attribute
// does not exist, the all inputs are placed on logical core 0.
llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4>
ExtractInputsForLogicalDevices(int num_logical_cores,
                               mlir::tf_device::LaunchFuncOp launch_func);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_XLA_SHARDING_UTIL_H_
