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

#ifndef XLA_TRANSLATE_MHLO_TO_HLO_LAYOUT_UTIL_H_
#define XLA_TRANSLATE_MHLO_TO_HLO_LAYOUT_UTIL_H_

#include <functional>
#include <vector>

#include "xla/client/xla_builder.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/xla_data.pb.h"

namespace mlir {

// XLA Layout preferences. Currently, when it comes to TPU, there are two
// primary layout choices for any XLA argumetns (parameter or resource): (1)
// CompactChunkPadded and (2) Linear. CompactChunkPadded is the native TPU
// layout while Linear is native host (CPU) layout.
// This enum allows the caller of XLA to progogate layout preference to the XLA
// compiler.
//   kNoPreference: the generic layout where the XLA compiler has the freedom
//                  to assign any layout.
//   kTpuPreferCompactChunkPaddedLayout: use native TPU layout on TPU.
//   kTpuPreferLinearLayout: use native CPU layout on TPU. The compiler may
//                           insert transformation TPU kernels.
// As the layout of any argument will change from a native host layout to a
// native TPU layout either on host or on device, XLA compiler and TPU runtime
// must be in coordination to transform the parameters in a consistent way.
enum class XlaLayoutPreference {
  kNoPreference = 0,
  kTpuPreferCompactChunkPaddedLayout = 1,
  kTpuPreferLinearLayout = 2
};

// The following defines the layout preference of an xla tensor.
// The return value of LayoutPreferenceFn can be used in
// ShapeRepresentationFn.
typedef std::function<xla::StatusOr<XlaLayoutPreference>(
    const xla::Shape& shape)>
    LayoutPreferenceFn;

typedef std::function<xla::StatusOr<xla::Shape>(
    const xla::Shape& shape, bool fast_mem,
    XlaLayoutPreference layout_preference)>
    ShapeRepresentationFn;

// Return a LayoutPreferenceFn that always uses kNoPreference layout.
LayoutPreferenceFn UseNoPreferenceLayoutFn();

// Rewrites the layout of xla_shape if there is tiled sharding.
xla::Status RewriteLayoutWithShardedShape(
    const std::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    const LayoutPreferenceFn& layout_preference_fn,
    const ShapeRepresentationFn& shape_representation_fn,
    xla::Shape* xla_shape);

// Adds reshapes to fix the layout of an output, if a shape_representation_fn or
// sharding is present.
xla::StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    const LayoutPreferenceFn& layout_preference_fn,
    const ShapeRepresentationFn& shape_representation_fn,
    std::optional<xla::OpSharding> sharding, bool fast_mem);

}  // namespace mlir

#endif  // XLA_TRANSLATE_MHLO_TO_HLO_LAYOUT_UTIL_H_
