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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_HANDLERS_FEATURE_COLUMN_HANDLER_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_HANDLERS_FEATURE_COLUMN_HANDLER_H_

#include <vector>
#include "tensorflow/contrib/boosted_trees/lib/learner/common/accumulators/feature-stats-accumulator.h"
#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/stats/feature-split-candidate.h"
#include "tensorflow/contrib/boosted_trees/proto/learner.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

// Handler interface for feature columns. Each feature column type may
// have its own handler which encapsulates the logic of aggregating gradient
// stats as well as generating split candidates for each partition.
// Handlers can be stateful and must be thread compatible.
class FeatureColumnHandler {
 public:
  FeatureColumnHandler(const int32 class_id, const int32 slot_id,
                       const int64 batch_size)
      : class_id_(class_id), slot_id_(slot_id), batch_size_(batch_size) {}

  virtual ~FeatureColumnHandler() {}
  FeatureColumnHandler(const FeatureColumnHandler& other) = delete;
  FeatureColumnHandler& operator=(const FeatureColumnHandler& other) = delete;

  // Aggregates example gradient stats for the feature column.
  virtual void AggregateGradientStats(
      const std::vector<int32>& example_partition_ids,
      const Tensor& example_first_order_gradients,
      const Tensor& example_second_order_gradients,
      FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>*
          gradient_stats_accumulator) const = 0;

  // Generates feature column split candidates for the specified roots.
  virtual void GenerateFeatureSplitCandidates(
      const LearnerConfig& learner_config, const std::vector<int32>& roots,
      const std::vector<NodeStats>& root_stats,
      const FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>&
          gradient_stats_accumulator,
      std::vector<FeatureSplitCandidate>* split_candidates) const = 0;

  // Accessors.
  int32 class_id() const { return class_id_; }
  int32 slot_id() const { return slot_id_; }
  int64 batch_size() const { return batch_size_; }

 protected:
  // The class Id.
  const int32 class_id_;

  // The slod Id for use as a unique Id across all feature columns.
  const int32 slot_id_;

  // Size of the batch of examples.
  const int64 batch_size_;
};

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  //  THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_HANDLERS_FEATURE_COLUMN_HANDLER_H_
