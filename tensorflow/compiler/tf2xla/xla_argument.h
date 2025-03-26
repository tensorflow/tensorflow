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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_ARGUMENT_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_ARGUMENT_H_

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

// Describes how to derive the value of each _Arg node in the graph/function
// being compiled. There must be one Argument for each _Arg index.
struct XlaArgument {
  enum Kind {
    // Default value; not a valid kind.
    kInvalid,

    // Argument is a compile-time constant. No associated runtime parameter.
    kConstant,

    // Argument is a Variable, TensorArray, or Stack resource. Has an
    // associated runtime parameter iff `initialized` is true.
    kResource,

    // A resource variable with a constant value known at compile time.
    kConstantResource,

    // Argument is a run-time parameter.
    kParameter,

    // Argument is an XLA token.
    kToken,

    // Argument is a TensorList.
    kTensorList,
  };

  Kind kind = kInvalid;

  // The type of the argument. If the argument is a resource, this
  // is the type of the variable's value, not DT_RESOURCE.
  DataType type = DT_INVALID;

  // The shape of the argument. For:
  // * a parameter: the shape of the parameter. We allow setting the xla shape
  //   if known. This helps avoid conversions to and from TensorShape.
  // * a constant: ignored; the shape given by constant_value is used
  //     instead.
  // * an uninitialized resource: ignored. We don't yet know the shape of an
  //     uninitialized resource (otherwise we would have initialized it!)
  // * an initialized variable: the shape of the variable's value.
  // * an initialized TensorArray or Stack resource: the shape of an entry in
  //   the TensorArray/Stack. Note this is the size of a single entry, not the
  //   XLA data structure that represents the complete stack/array.
  absl::variant<TensorShape, xla::Shape> shape;

  // The value of the argument, if it is a compile-time constant. Must be a
  // host-memory tensor.
  Tensor constant_value;

  // The upper bounds of the value.
  std::optional<Tensor> value_bound;

  // Indicates whether each value is dynamic or constant.
  std::optional<Tensor> value_dynamism;

  // The name of this argument, used for debugging.
  string name;

  // The name of TensorFlow _Arg node, used for debugging.
  string node_name;

  // For a kResource, what kind of resource is it?
  XlaResource::Kind resource_kind = XlaResource::kInvalid;

  // For a kResource, has this resource been initialized?
  bool initialized = false;

  // For a kResource, is this resource on Fast Memory.
  bool fast_mem = false;

  // For a TensorArray or Stack resource, what is the array's declared size?
  // (Used for lazy initialization.)
  int64_t max_array_size = -1;

  // TensorArray resource parameters are passed as (array, gradient array 0,
  // ..., gradient array k), where the gradient arrays are in the same order
  // as `tensor_array_gradients`.
  std::set<string> tensor_array_gradients;

  // Whether this argument will receive the same data across all replicas.
  bool is_same_data_across_replicas = false;

  bool operator==(const XlaArgument& other) const;

  // Returns a human-readable summary of the argument.
  string HumanString() const;

  // Returns the dimension sizes for either TensorShape or xla::Shape.
  std::vector<int64_t> DimensionSizes() const;
  absl::InlinedVector<int64_t, 4> DimensionSizesAsInlinedVector() const;

  // Returns the human-readable string for either TensorShape or xla::Shape.
  string ShapeHumanString() const;

  // Whether to broadcast this parameter to all replicas before use.
  // When true, xla_compiler should input/output alias this arg to prevent
  // unnecessary HBM usage.
  bool requires_broadcast = false;
  std::optional<ManagedStackTrace> definition_stack_trace;
};

// Returns true if any of `args` is an uninitialized resource variable.
bool AnyUninitializedResourceArg(absl::Span<const XlaArgument> args);

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_ARGUMENT_H_
