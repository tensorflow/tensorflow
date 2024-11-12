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

#ifndef TENSORFLOW_CORE_TPU_TPU_EMBEDDING_SPMD_SHARDING_UTILS_H_
#define TENSORFLOW_CORE_TPU_TPU_EMBEDDING_SPMD_SHARDING_UTILS_H_

#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {
namespace tpu {

// Gets SPMD manual sharding annotation from the input shape. If the shape is a
// scalar (rank = 0), the tensor is replicated across all the cores within the
// replica. If the shape is a non-scalar (rank >= 1), the tensor is sharded on
// dimension `0' across all the cores within the same replica.
absl::StatusOr<xla::OpSharding> SpmdShardingAnnotationOnFirstDim(
    const xla::Shape& shape, int core_count_per_replica,
    xla::XlaBuilder* builder);

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_TPU_EMBEDDING_SPMD_SHARDING_UTILS_H_
