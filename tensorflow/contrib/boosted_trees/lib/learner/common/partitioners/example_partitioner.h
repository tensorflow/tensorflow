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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_PARTITIONERS_EXAMPLE_PARTITIONER_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_PARTITIONERS_EXAMPLE_PARTITIONER_H_

#include <vector>
#include "tensorflow/contrib/boosted_trees/lib/trees/decision_tree.h"
#include "tensorflow/contrib/boosted_trees/lib/utils/batch_features.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {

// Partitions examples based on the path through the current tree.
class ExamplePartitioner {
 public:
  // Updates partitions from previous set using the current tree structure by
  // traversing sub-roots for each example. This method can be optionally
  // parallelized using the passed thread pool.
  static void UpdatePartitions(const trees::DecisionTreeConfig& tree_config,
                               const utils::BatchFeatures& features,
                               int desired_parallelism,
                               thread::ThreadPool* const thread_pool,
                               int32* example_partition_ids);

  // Partitions examples using the current tree structure by traversing from
  // root for each example. This method can be optionally parallelized using
  // the passed thread pool.
  static void PartitionExamples(const trees::DecisionTreeConfig& tree_config,
                                const utils::BatchFeatures& features,
                                int desired_parallelism,
                                thread::ThreadPool* const thread_pool,
                                int32* example_partition_ids);
};

}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_PARTITIONERS_EXAMPLE_PARTITIONER_H_
