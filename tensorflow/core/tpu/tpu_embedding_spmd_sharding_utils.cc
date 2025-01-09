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

#include "tensorflow/core/tpu/tpu_embedding_spmd_sharding_utils.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace tensorflow {
namespace tpu {

absl::StatusOr<xla::OpSharding> SpmdShardingAnnotationOnFirstDim(
    const xla::Shape& shape, int core_count_per_replica,
    xla::XlaBuilder* builder) {
  if (!shape.IsArray()) {
    LOG(ERROR) << "Input shape is not ArrayType";
  }
  if (!shape.is_static()) {
    LOG(ERROR) << "Input shape is not static shape.";
  }

  xla::OpSharding op_sharding;
  if (shape.rank() == 0) {
    // Replicate scalar tensor (used for handling dynamic learning rates).
    op_sharding.set_type(xla::OpSharding::REPLICATED);
  } else {
    // Split tensors with rank >= 1 (used for embedding activations, gradients,
    // and deduplication data).
    if (shape.dimensions(0) % core_count_per_replica != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Number of elements %d in the split dimension must be a multiple of "
          "the number of cores per replica %d",
          shape.dimensions(0), core_count_per_replica));
    }

    std::vector<int> tile_assignment_dimensions(shape.dimensions_size(), 1);
    tile_assignment_dimensions[0] = core_count_per_replica;

    op_sharding.set_type(xla::OpSharding::OTHER);
    for (const int tile_assignment : tile_assignment_dimensions) {
      op_sharding.add_tile_assignment_dimensions(tile_assignment);
    }
    for (int i = 0; i < core_count_per_replica; ++i) {
      op_sharding.add_tile_assignment_devices(i);
    }
  }
  return op_sharding;
}

}  // namespace tpu
}  // namespace tensorflow
