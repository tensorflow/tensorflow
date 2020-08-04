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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T>
class OperationPass;
class Pass;

namespace kernel_gen {
namespace tf_framework {

// Test pass for applying TF Framework -> LLVM patterns.
std::unique_ptr<OperationPass<ModuleOp> >
createTestTFFrameworkLegalizeToLLVMPass();

// Pass to replace some of the Standard ops with TF Framework ops.
// * adds tf_framework::OpKernelContextType argument to the function
// * std.alloc becomes tf_framework.alloc_raw
// * std.dealloc becomes tf_framework.dealloc_raw
std::unique_ptr<OperationPass<ModuleOp> > createEmbedTFFrameworkPass();

}  // namespace tf_framework

namespace transforms {

// Pass to tranform shape computations in shape dialect to standard and scf
// using memref descriptors.
std::unique_ptr<Pass> CreateShapeToDescriptorsPass();

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TRANSFORMS_PASSES_H_
