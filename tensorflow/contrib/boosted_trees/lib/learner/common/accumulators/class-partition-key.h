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
#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_ACCUMULATORS_CLASS_PARTITION_KEY_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_ACCUMULATORS_CLASS_PARTITION_KEY_H_

#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {

// Key into a specific class and partition to accumulate stats
// for the specified feature id. A feature id can be the quantile
// for a float feature or the hash/dictionary entry for a string feature.
struct ClassPartitionKey {
  ClassPartitionKey() : class_id(-1), partition_id(-1), feature_id(-1) {}

  ClassPartitionKey(uint32 c, uint32 p, uint64 f)
      : class_id(c), partition_id(p), feature_id(f) {}

  bool operator==(const ClassPartitionKey& other) const {
    return (feature_id == other.feature_id) &&
           (partition_id == other.partition_id) && (class_id == other.class_id);
  }

  // Hasher for ClassPartitionKey.
  struct Hash {
    size_t operator()(const ClassPartitionKey& key) const {
      uint64 class_partition =
          (static_cast<uint64>(key.partition_id) << 32) | (key.class_id);
      return Hash64Combine(class_partition, key.feature_id);
    }
  };

  // Class to predict for, this is constant for binary tasks.
  uint32 class_id;

  // Tree partition defined by traversing the tree to the leaf.
  uint32 partition_id;

  // Feature Id within the feature column.
  uint64 feature_id;
};

}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_ACCUMULATORS_CLASS_PARTITION_KEY_H_
