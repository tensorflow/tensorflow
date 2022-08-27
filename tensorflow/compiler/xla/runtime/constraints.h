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

#ifndef XLA_RUNTIME_CONSTRAINTS_H_
#define XLA_RUNTIME_CONSTRAINTS_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {
namespace runtime {

// Constraints on the function argument can be specified with the function
// argument attributes.
//
// Example:
//
//   func @compute(
//     // Rank of the `%arg` must be known at compile time.
//     %arg: tensor<*xf32> { rt.constraint = "rank" }
//   ) -> tensor<?xf32> { ... }
//
// TODO(b/187114012): Add attribute verifier to `rt` dialect.
constexpr const char* kArgumentConstraintAttrName = "rt.constraint";

// Constraint on what argument information must be available at compile time in
// order to successfully compile the executable:
//
//   `rank`  : argument must have statically known rank.
//   `shape` : argument must have statically known shape.
//   `value` : argument must have statically known value, and such arguments
//             replaced with constants inside the compiled function body and
//             and all value constrained argument uses replaced with the sunk
//             constant value.
//
// For now these constraints are supported by arguments of shaped types (tensors
// or memrefs), but potentially can be extended to support open type hierarchy
// of user-defined types.
//
// XLA program example:
//
//   func @main(
//     %input0: memref<*xf32>   { rt.constraint = "rank"  },
//     %input1: memref<?x?xf32> { rt.constraint = "shape" },
//     %perm: memref<4xi32>     { rt.constraint = "value" }
//   ) attributes { rt.entrypoint } { ... }
//
// Entrypoint function can define constraints on its arguments, that must be
// resolved before the function can be compiled. If constraints can't be
// resolved statically from the function signature (e.g. rank is unknown), then
// the runtime will specialize generic function to concrete operands at runtime
// (concrete operands rank, shape or value).
//
// If function arguments do not have unresolved constraints, compiler can
// instantiate the default executable, that can take all compatible inputs
// without recompilation.
//
// (a) Rank constraint:
//
//     %arg : tensor<*xf32> { rt.constraint = "rank" }
//
//     Before compiling the function, unranked input type will be updated to the
//     corresponding ranked input type (e.g. unranked tensor -> ranked tensor).
//
// (b) Shape constraint:
//
//     %arg : tensor<?x?xf32> { rt.constraint = "shape" }
//
//     Shape of the runtime argument will be used to specialize the compiled
//     function, if this shape seen the first time, it will trigger function
//     recompilation.
//
// (c) Value constraint:
//
//     %reduction_dimension : tensor<i32> { rt.constraint = "value" }
//
//     Runtime value will be sunk into the body of a function as a constant,
//     and the function will be recompiled. For example this can be used to sink
//     reduction dimensions to generate more efficient code.
//
//     Value constraint is only supported for the integer data type, in practice
//     it should be reduction dimension, dimension permutation, or any similar
//     value that does not change often, and is required for generating
//     efficient code.
//
//  Shape and value specialization example:
//
//    // Computes `%arg0` mean value over the axis specified by the `%arg1`.
//    // See: https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
//    func @mean(%arg0: tensor<?x?xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
//      %0 = "tf.Mean(%arg0, %arg1)
//             : (tensor<?x?xf32>, tensor<i32>) -> tensor<?xf32>
//      return %0: tensor<?xf32>
//    }
//
//  Shape specialization to input shapes: [tensor<4x8xf32>, tensor<f32>]
//
//    func @mean(%arg0: tensor<4x8xf32>, %arg1: tensor<i32>) -> tensor<?xf32> {
//      %0 = "tf.Mean(%arg0, %arg1)
//             : (tensor<4x8xf32>, tensor<i32>) -> tensor<?xf32>
//      return %0: tensor<?xf32>
//    }
//
//    Shape specialization in this particular case doesn't bring much
//    improvement, because without knowing the reduction axis we can't infer
//    any new information from the input shape alone.
//
//  Value specialization to input values: [ <do-not-specialize>, dense<1 : i32>]
//
//    func @mean(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
//      %0 = "tf.Constant" { value = dense<1 : i32>} -> tensor<i32>
//      %1 = "tf.Mean(%arg0, %0)
//             : (tensor<4x8xf32>, tensor<i32>) -> tensor<4xf32>
//      return %1 : tensor<4xf32>
//    }
//
//    By specializing function to the concrete value of the second argument, by
//    sinking it into the function body we can infer the output shape. Also this
//    information allows to statically choose reduction implementation optimized
//    for reducing along the inner most dimension.
//
//    Furthermore static information about reduction axis allows to lower mean
//    operation to Linalg generic operation. Dynamic reduction axis is not
//    representable in Linalg, and would require multi-versioning and dynamic
//    dispatch at runtime.
//
enum class ArgumentConstraint {
  // Constraint was resolved based on the static information in the function
  // signature type or it was never specified by the argument attribute.
  kResolved = 0,
  kRank = 1,
  kShape = 2,
  kValue = 3
};

// Converts argument constraint string to the corresponding enum class.
absl::StatusOr<ArgumentConstraint> ParseArgumentConstraint(
    std::string_view str);

std::string ArgumentConstraintToString(ArgumentConstraint constraint);

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_CONSTRAINTS_H_
