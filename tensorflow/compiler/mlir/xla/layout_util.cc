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

#include "tensorflow/compiler/mlir/xla/layout_util.h"

namespace mlir {

// Rewrites the layout of xla_shape if there is tiled sharding.
xla::Status RewriteLayoutWithShardedShape(
    const std::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    const LayoutPreferenceFn& layout_preference_fn,
    const ShapeRepresentationFn& shape_representation_fn,
    xla::Shape* xla_shape) {
  if (sharding && !sharding->IsTileMaximal() && !sharding->IsManual()) {
    // After sharding, per core shape might have different layout. For example,
    // before sharding, a shape [128, 128] will be assigned default
    // minor-to-major {1, 0}. But after we shard this shape to [128, 64] * 2,
    // the sharded shapes will have minor-to-major {0, 1}.
    //
    // As a result, for sharded shapes, we set their layout to per core shape's
    // layout.
    //
    // TODO(endlessroad): for variable input & update, we might have
    // different layouts which will prevent input output aliasing and
    // increase memory usage. Investigate such cases.
    int64_t device = *sharding->tile_assignment().begin();
    std::vector<int64_t> offset =
        sharding->TileOffsetForDevice(*xla_shape, device);
    std::vector<int64_t> limit =
        sharding->TileLimitForDevice(*xla_shape, device);
    std::vector<int64_t> dimensions(xla_shape->rank());
    for (int64_t i = 0; i < xla_shape->rank(); ++i) {
      dimensions[i] = limit[i] - offset[i];
    }
    xla::Shape per_device_xla_shape =
        xla::ShapeUtil::MakeShape(xla_shape->element_type(), dimensions);
    TF_ASSIGN_OR_RETURN(auto layout_preference,
                        layout_preference_fn
                            ? layout_preference_fn(per_device_xla_shape)
                            : XlaLayoutPreference::kNoPreference);
    TF_ASSIGN_OR_RETURN(
        per_device_xla_shape,
        shape_representation_fn
            ? shape_representation_fn(per_device_xla_shape, use_fast_memory,
                                      layout_preference)
            : per_device_xla_shape);
    *xla_shape->mutable_layout() = per_device_xla_shape.layout();
  }
  return xla::OkStatus();
}

// There is a shape_representation_fn or sharding for an output, this function
// uses a reshape to fix the layout.
xla::StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    const LayoutPreferenceFn& layout_preference_fn,
    const ShapeRepresentationFn& shape_representation_fn,
    std::optional<xla::OpSharding> sharding, bool fast_mem) {
  if (original_shape.IsTuple()) {
    std::vector<xla::XlaOp> elements;
    for (int i = 0; i < original_shape.tuple_shapes_size(); ++i) {
      auto subsharding = sharding ? sharding->tuple_shardings(i) : sharding;
      TF_ASSIGN_OR_RETURN(
          auto element,
          ReshapeWithCorrectRepresentationAndSharding(
              builder, xla::GetTupleElement(original, i),
              original_shape.tuple_shapes(i), layout_preference_fn,
              shape_representation_fn, subsharding, fast_mem));
      elements.push_back(element);
    }
    return xla::Tuple(builder, elements);
  }
  if (!original_shape.IsArray()) return original;
  TF_ASSIGN_OR_RETURN(auto layout_preference,
                      layout_preference_fn
                          ? layout_preference_fn(original_shape)
                          : XlaLayoutPreference::kNoPreference);
  TF_ASSIGN_OR_RETURN(
      auto to_shape,
      shape_representation_fn
          ? shape_representation_fn(original_shape, fast_mem, layout_preference)
          : original_shape);
  if (sharding) {
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        xla::HloSharding::FromProto(*sharding));

    TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
        hlo_sharding, fast_mem, layout_preference_fn, shape_representation_fn,
        &to_shape));
  }
  if (xla::ShapeUtil::Compatible(original_shape, to_shape)) {
    for (int64_t i = 0; i < original_shape.rank(); ++i) {
      to_shape.set_dynamic_dimension(i, original_shape.is_dynamic_dimension(i));
    }
  }
  return xla::Reshape(to_shape, original);
}

}  // namespace mlir
