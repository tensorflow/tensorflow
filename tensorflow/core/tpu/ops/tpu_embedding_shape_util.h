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
#ifndef TENSORFLOW_CORE_TPU_OPS_TPU_EMBEDDING_SHAPE_UTIL_H_
#define TENSORFLOW_CORE_TPU_OPS_TPU_EMBEDDING_SHAPE_UTIL_H_

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"

namespace tensorflow {
namespace tpu {

// Utility class for inferring TpuEmbedding shape information.
class TpuEmbeddingShapeUtil {
 public:
  // Compute the shape of one embedding table stored on the
  // TpuEmbeddingEngine. The table descriptor from the TpuEmbedding
  // configuration is supplied in config. On success, shape is populated with
  // the shape of the embedding table that will be loaded or retrieved using
  // Ops such as {Load,Retrieve}TpuEmbedding*Parameters.
  static Status ComputeOneTableShape(int64 vocabulary_size, int table_dimension,
                                     int shard_id, int num_shards,
                                     TensorShapeProto* shape);

  // Compute the shapes of the embedding tables stored on the
  // TpuEmbeddingEngine. The TpuEmbedding configuration is supplied in
  // config. On success, shapes is populated with the shape of each embedding
  // table that will be loaded or retrieved using Ops such as
  // {Load,Retrieve}AllTpuEmbeddingParameters.
  static Status ComputeTableShapes(absl::Span<const int64> vocabulary_sizes,
                                   absl::Span<const int> table_dimensions,
                                   int shard_id, int num_shards,
                                   std::vector<TensorShapeProto>* shapes);

  static Status ComputeTableShapes(
      const tensorflow::tpu::TPUEmbeddingConfiguration& config, int shard_id,
      int num_shards, std::vector<TensorShapeProto>* shapes);

  static TensorShapeProto MakeEmpty2DShape();

 private:
  // Compute the number of embedding IDs per embedding table shard.
  // There are as many shards as the number of hosts in the job.
  static xla::StatusOr<int64> ComputeNumEmbeddingIdsPerShard(
      int64 vocabulary_size, int shard_id, int num_shards);
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_OPS_TPU_EMBEDDING_SHAPE_UTIL_H_
