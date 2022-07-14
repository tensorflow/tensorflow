/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_UTILS_H_

#include <string>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

namespace mlir {
namespace kernel_gen {
namespace transforms {

// Attempts to find function symbol in the module, adds it if not found.
FlatSymbolRefAttr GetOrInsertLLVMFunction(StringRef func_name, Type func_type,
                                          Operation* op, OpBuilder* b);

// Attemts to find a global string constant in the module, adds it if not found.
Value CreateOrFindGlobalStringConstant(Location loc, StringRef global_name,
                                       StringRef content, OpBuilder* builder);

// Generates a global name with the format "base_hash(content)".
std::string GetGlobalName(StringRef base, StringRef content);

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_UTILS_H_
