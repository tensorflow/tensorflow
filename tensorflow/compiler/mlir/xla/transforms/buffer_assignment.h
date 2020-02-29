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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_

#include "mlir/IR/Builders.h"   // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace xla {

namespace detail {
class BufferAssignmentAliasAnalysis {
 public:
  using ValueSetT = SmallPtrSet<Value, 16>;

 public:
  /// Constructs a new alias analysis using the op provided.
  BufferAssignmentAliasAnalysis(Operation* op);

  /// Finds all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const;

 private:
  /// Recursively determines alias information for the given value. It stores
  /// all newly found potential aliases in the given result set.
  void resolveRecursive(Value value, ValueSetT& result) const;

  /// Initializes the internal mappings.
  void build(MutableArrayRef<Region> regions);

 private:
  /// Maps values to all immediate aliases this value can have.
  llvm::DenseMap<Value, ValueSetT> aliases;
};
}  // namespace detail
}  // namespace xla
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_BUFFER_ASSIGNMENT_H_
