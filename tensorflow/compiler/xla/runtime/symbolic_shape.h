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

#ifndef XLA_RUNTIME_SYMBOLIC_SHAPE_H_
#define XLA_RUNTIME_SYMBOLIC_SHAPE_H_

#include <optional>

#include "absl/status/statusor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/xla/runtime/arguments.h"
#include "tensorflow/compiler/xla/runtime/constraints.h"

namespace xla {
namespace runtime {

// Symbolic shapes resolver computes the symbolic shapes of the arguments based
// on the function signature, and concrete shapes of the arguments at runtime.
//
// Example: dimensions that have the same symbolic shape at runtime.
//
//   signature: func @compute(%arg0: tensor<?xf32>, %arg1: tensor<?xf32)
//                            ^                     ^
//   arguments:               memref<123xf32>       memref<123xf32>
//                            ^                     ^
//   symbolic shapes:         [-2xf32]              [-2xf32]
//
// Each unknown dimension in the function signature will be assigned a symbolic
// dimension. If multiple shaped arguments have unknown dimensions that are the
// same at runtime, they will be assigned the same symbolic dimensions value
// (e.g. `-2` in the example above).
//
// If an unknown dimension at runtime is equal to some statically known
// dimension in the function signature (of any shaped argument), it will be
// resolved to that statically known constant value:
//
// Example: in this example unknown dimension of `arg0` replaced with a `32`.
//
//  signature:  func @compute(%arg0: tensor<?xf32>, %arg1: tensor<32xf32>)
//                            ^                     ^
//  arguments:                memref<32xf32>        memref<32xf32>
//                            ^                     ^
//  symbolic shapes:          [32xf32]              [32xf32]
//
// Unknown dimensions that are `1` at runtime are always materialized as a
// statically known `1` in the symbolic shape.
class SymbolicShapesResolver {
 public:
  // Dimension size can be symbolic (<= -2) or static.
  using SymbolicShape = llvm::SmallVector<int64_t>;
  // Dimension size can be dynamic (ShapedType::kDynamicSize) or static.
  using StaticShape = llvm::SmallVector<int64_t>;

  SymbolicShapesResolver(const FunctionType& signature,
                         absl::Span<const ArgumentConstraint> constraints);

  // Resolves symbolic shapes from the runtime arguments. Returns failure if
  // runtime dimensions do not match the statically known dimensions.
  absl::StatusOr<llvm::SmallVector<SymbolicShape>> Resolve(
      ArgumentsRef arguments) const;

  // Resolves symbolic shapes and computes the hash value from the runtime
  // arguments. Returns failure if runtime dimensions do not match the
  // statically known dimensions.
  //
  // This function might not return the same hash value as calling `Resolve` and
  // then `Hash`, because it might use more efficient hashing algorithm.
  absl::StatusOr<llvm::hash_code> ResolveHash(ArgumentsRef arguments) const;

  // Replaces all symbolic dimensions with dynamic dimension.
  static llvm::SmallVector<int64_t> Normalize(const SymbolicShape& shape);

  // Computes a hash value of the symbolic shapes.
  static llvm::hash_code Hash(absl::Span<const SymbolicShape> symbolic_shapes);

  ArgumentConstraint constraint(size_t index) const;
  size_t num_arguments() const;
  bool has_argument_sizes(size_t index) const;
  const StaticShape& argument_sizes(size_t index) const;
  bool seen_static_size(size_t dim) const;

 private:
  // Constraints on the function arguments.
  llvm::SmallVector<ArgumentConstraint> constraints_;

  // Statically known sizes of shaped arguments from the function signature. For
  // non-shaped arguments (e.g. opaque pointers) we keep empty shape value.
  llvm::SmallVector<std::optional<StaticShape>> arguments_sizes_;

  // Values of statically known dimensions sizes in the function signature.
  llvm::DenseSet<int64_t> seen_static_sizes_;

  // The iteration order for the arguments when resolving symbolic shapes.
  llvm::SmallVector<size_t> iteration_order_;

  // The iteration order for the arguments when resolving symbolic shapes hash.
  llvm::SmallVector<size_t> hash_iteration_order_;
};

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_SYMBOLIC_SHAPE_H_
