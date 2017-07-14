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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_ACCUMULATORS_FEATURE_STATS_ACCUMULATOR_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_ACCUMULATORS_FEATURE_STATS_ACCUMULATOR_H_

#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/boosted_trees/lib/learner/common/accumulators/class-partition-key.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {

// Feature stats accumulator to aggregate stats across various
// feature columns. This class is thread compatible not safe, the user
// must ensure proper synchronization if many threads update overlapping
// feature columns.
template <typename StatsType, typename Accumulator>
class FeatureStatsAccumulator {
 public:
  using FeatureStats =
      std::unordered_map<ClassPartitionKey, StatsType, ClassPartitionKey::Hash>;

  explicit FeatureStatsAccumulator(size_t num_feature_columns,
                                   Accumulator accumulator = Accumulator())
      : accumulator_(accumulator), feature_column_stats_(num_feature_columns) {}

  // Delete copy and assign.
  FeatureStatsAccumulator(const FeatureStatsAccumulator& other) = delete;
  FeatureStatsAccumulator& operator=(const FeatureStatsAccumulator& other) =
      delete;

  // Adds stats for the specified class, partition and feature within
  // the desired slot.
  void AddStats(uint32 slot_id, uint32 class_id, uint32 partition_id,
                uint64 feature_id, const StatsType& stats) {
    accumulator_(stats, &feature_column_stats_[slot_id][ClassPartitionKey(
                            class_id, partition_id, feature_id)]);
  }

  // Retrieves stats for the specified class, partition and feature
  // within the desired feature column. Default stats are returned if no match
  // can be found. Note that the feature column index must be valid.
  StatsType GetStats(uint32 slot_id, uint32 class_id, uint32 partition_id,
                     uint64 feature_id) const {
    auto it = feature_column_stats_[slot_id].find(
        ClassPartitionKey(class_id, partition_id, feature_id));
    return it != feature_column_stats_[slot_id].end() ? it->second
                                                      : StatsType();
  }

  // Returns feature stats for a given slot.
  FeatureStats GetFeatureStats(uint32 slot_id) const {
    return feature_column_stats_[slot_id];
  }

 private:
  // Accumulator method to use.
  const Accumulator accumulator_;

  // Vector of stats indexed by the feature column.
  std::vector<FeatureStats> feature_column_stats_;
};

}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_COMMON_ACCUMULATORS_FEATURE_STATS_ACCUMULATOR_H_
