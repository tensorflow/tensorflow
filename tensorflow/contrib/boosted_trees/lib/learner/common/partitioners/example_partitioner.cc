// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/learner/common/partitioners/example_partitioner.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/parallel_for.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {

void ExamplePartitioner::UpdatePartitions(
    const boosted_trees::trees::DecisionTreeConfig& tree_config,
    const boosted_trees::utils::BatchFeatures& features,
    const int desired_parallelism, thread::ThreadPool* const thread_pool,
    int32* example_partition_ids) {
  // Get batch size.
  const int64 batch_size = features.batch_size();
  if (batch_size <= 0) {
    return;
  }

  // Lambda for doing a block of work.
  auto partition_examples_subset = [&tree_config, &features,
                                    &example_partition_ids](const int64 start,
                                                            const int64 end) {
    if (TF_PREDICT_TRUE(tree_config.nodes_size() > 0)) {
      auto examples_iterable = features.examples_iterable(start, end);
      for (const auto& example : examples_iterable) {
        int32& example_partition = example_partition_ids[example.example_idx];
        example_partition = boosted_trees::trees::DecisionTree::Traverse(
            tree_config, example_partition, example);
        DCHECK_GE(example_partition, 0);
      }
    } else {
      std::fill(example_partition_ids + start, example_partition_ids + end, 0);
    }
  };

  // Parallelize partitioning over the batch.
  boosted_trees::utils::ParallelFor(batch_size, desired_parallelism,
                                    thread_pool, partition_examples_subset);
}

void ExamplePartitioner::PartitionExamples(
    const boosted_trees::trees::DecisionTreeConfig& tree_config,
    const boosted_trees::utils::BatchFeatures& features,
    const int desired_parallelism, thread::ThreadPool* const thread_pool,
    int32* example_partition_ids) {
  // Get batch size.
  const int64 batch_size = features.batch_size();
  if (batch_size <= 0) {
    return;
  }

  // Lambda for doing a block of work.
  auto partition_examples_subset = [&tree_config, &features,
                                    &example_partition_ids](const int64 start,
                                                            const int64 end) {
    if (TF_PREDICT_TRUE(tree_config.nodes_size() > 0)) {
      auto examples_iterable = features.examples_iterable(start, end);
      for (const auto& example : examples_iterable) {
        uint32 partition = boosted_trees::trees::DecisionTree::Traverse(
            tree_config, 0, example);
        example_partition_ids[example.example_idx] = partition;
        DCHECK_GE(partition, 0);
      }
    } else {
      std::fill(example_partition_ids + start, example_partition_ids + end, 0);
    }
  };

  // Parallelize partitioning over the batch.
  boosted_trees::utils::ParallelFor(batch_size, desired_parallelism,
                                    thread_pool, partition_examples_subset);
}

}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow
