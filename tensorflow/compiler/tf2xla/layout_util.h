/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Utilities for working with XLA layout and shapes.

#ifndef TENSORFLOW_COMPILER_TF2XLA_LAYOUT_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_LAYOUT_UTIL_H_

#include <vector>

#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

class XlaShapeLayoutHelpers {
 public:
  // The following defines the layout preference of an xla tensor.
  // The return value of LayoutPreferenceFn can be used in
  // XlaHelper::ShapeRepresentationFn.
  typedef std::function<XlaLayoutPreference(const TensorShape&, DataType,
                                            std::optional<XlaArgument::Kind>)>
      LayoutPreferenceFn;

  // A bundle of LayoutPreferenceFn and ShapeRepresentationFn.
  struct ShapeDeterminationFns {
    // Use no preference function, and identity shape representation function,
    // as default value.
    ShapeDeterminationFns();

    ShapeDeterminationFns(
        LayoutPreferenceFn layout_preference_fn,
        XlaHelpers::ShapeRepresentationFn shape_representation_fn)
        : layout_preference_fn(layout_preference_fn),
          shape_representation_fn(shape_representation_fn) {}

    LayoutPreferenceFn layout_preference_fn;
    XlaHelpers::ShapeRepresentationFn shape_representation_fn;
  };
};

// Return a LayoutPreferenceFn that always uses kNoPreference layout.
XlaShapeLayoutHelpers::LayoutPreferenceFn UseNoPreferenceLayoutFn();

// Rewrites the layout of xla_shape if there is tiled sharding.
Status RewriteLayoutWithShardedShape(
    const std::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    xla::Shape* xla_shape);

// Adds reshapes to fix the layout of an output, if a shape_representation_fn or
// sharding is present.
StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    std::optional<xla::OpSharding> sharding, bool fast_mem);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_SHAPE_UTIL_H_
