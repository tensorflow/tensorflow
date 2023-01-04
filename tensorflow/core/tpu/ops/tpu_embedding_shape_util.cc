/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/ops/tpu_embedding_shape_util.h"

#include <array>
#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {
namespace tpu {

using tensorflow::tpu::TPUEmbeddingConfiguration;

/* static */ Status TpuEmbeddingShapeUtil::ComputeOneTableShape(
    int64 vocabulary_size, int table_dimension, int shard_id, int num_shards,
    TensorShapeProto* shape) {
  if (num_shards <= 0) {
    return errors::InvalidArgument(
        "The number of shards for the embedding layer must be > 0. Currently "
        "set to: ",
        num_shards);
  }
  if (shard_id < 0 || shard_id >= num_shards) {
    return errors::InvalidArgument("The value of shard_id must be >= 0 and < ",
                                   num_shards,
                                   ". Currently set to: ", shard_id);
  }
  *shape = TensorShapeProto();
  auto* dim0 = shape->add_dim();
  TF_ASSIGN_OR_RETURN(
      int64 num_sharded_ids,
      ComputeNumEmbeddingIdsPerShard(vocabulary_size, shard_id, num_shards));
  dim0->set_size(num_sharded_ids);
  auto* dim1 = shape->add_dim();
  dim1->set_size(table_dimension);
  return OkStatus();
}

/* static */ Status TpuEmbeddingShapeUtil::ComputeTableShapes(
    const absl::Span<const int64> vocabulary_sizes,
    const absl::Span<const int> table_dimensions, int shard_id, int num_shards,
    std::vector<TensorShapeProto>* shapes) {
  shapes->resize(vocabulary_sizes.size());
  for (int i = 0; i < vocabulary_sizes.size(); ++i) {
    TF_RETURN_IF_ERROR(TpuEmbeddingShapeUtil::ComputeOneTableShape(
        vocabulary_sizes[i], table_dimensions[i], shard_id, num_shards,
        &(*shapes)[i]));
  }
  return OkStatus();
}

/* static */ Status TpuEmbeddingShapeUtil::ComputeTableShapes(
    const TPUEmbeddingConfiguration& config, int shard_id, int num_shards,
    std::vector<TensorShapeProto>* shapes) {
  std::vector<int64> vocabulary_sizes;
  std::vector<int> table_dimensions;
  for (auto& table_descriptor : config.table_descriptor()) {
    vocabulary_sizes.push_back(table_descriptor.vocabulary_size());
    table_dimensions.push_back(table_descriptor.dimension());
  }
  return ComputeTableShapes(vocabulary_sizes, table_dimensions, shard_id,
                            num_shards, shapes);
}

TensorShapeProto TpuEmbeddingShapeUtil::MakeEmpty2DShape() {
  TensorShapeProto shape;
  shape.add_dim()->set_size(0);
  shape.add_dim()->set_size(0);
  return shape;
}

/* static */ xla::StatusOr<int64>
TpuEmbeddingShapeUtil::ComputeNumEmbeddingIdsPerShard(int64 vocabulary_size,
                                                      int shard_id,
                                                      int num_shards) {
  // If the number of IDs does not evenly divide the number of shards, the first
  // `vocabulary_size % num_shards` partitions are assigned one more ID.
  int64 vocabulary_size_per_shard =
      xla::FloorOfRatio<int64>(vocabulary_size, num_shards);
  if (shard_id < (vocabulary_size % num_shards)) {
    ++vocabulary_size_per_shard;
  }
  if (vocabulary_size_per_shard == 0) {
    return errors::InvalidArgument(
        "All embedding shards must be non-empty, shard ID: ", shard_id,
        " is empty, vocabulary size: ", vocabulary_size,
        ", number of shards: ", num_shards);
  }
  return vocabulary_size_per_shard;
}

}  // namespace tpu
}  // namespace tensorflow
