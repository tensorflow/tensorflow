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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_RESOURCE_OP_LIFTING_CLEANUP_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_RESOURCE_OP_LIFTING_CLEANUP_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

// Performs IR cleanup and canonicalization in preparation for Resource Op
// Lifting pass. It does several things:
// - Eliminate identity nodes to remove (most) of resource aliasing
// - Canonicalize functional control flow. For functional control flow we
//   expect that any resource output of these ops matches the corresponding
//   input, and then forward that input to the output. Fails if this is not the
//   case. If successful, the following invariants will hold true:
//   (a) For if/case, any resource type results will be deleted.
//   (b) For while, any resource type results will be unused.
// - Canonicalize region based control flow. Again, any resource outputs are
//   expected to be resolved to be one of the captured resource inputs. Fails
//   if this is not the case. If successful, the following invariants will hold
//   true:
//   (a) For if/case, any resource type results will be deleted.
//   (b) For while, any resource type results will be unused.
namespace mlir {
namespace TF {
LogicalResult CleanupAndCanonicalizeForResourceOpLifting(ModuleOp module);
LogicalResult CleanupAndCanonicalizeForResourceOpLifting(FuncOp func);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_RESOURCE_OP_LIFTING_CLEANUP_H_
