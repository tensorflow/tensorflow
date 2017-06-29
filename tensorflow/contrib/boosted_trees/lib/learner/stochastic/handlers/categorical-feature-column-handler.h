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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_HANDLERS_CATEGORICAL_FEATURE_COLUMN_HANDLER_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_HANDLERS_CATEGORICAL_FEATURE_COLUMN_HANDLER_H_

#include "tensorflow/contrib/boosted_trees/lib/learner/stochastic/handlers/feature-column-handler.h"

namespace tensorflow {
namespace boosted_trees {
namespace learner {
namespace stochastic {

// Handler for a categorical feature column in the single class case.
class CategoricalFeatureColumnHandler : public FeatureColumnHandler {
 public:
  CategoricalFeatureColumnHandler(const int32 class_id, const int32 slot_id,
                                  const int64 batch_size,
                                  const int32 feature_column,
                                  TTypes<int64>::ConstMatrix indices,
                                  TTypes<int64>::ConstVec values)
      : FeatureColumnHandler(class_id, slot_id, batch_size),
        feature_column_(feature_column),
        indices_(indices),
        values_(values) {}

  void AggregateGradientStats(
      const std::vector<int32>& example_partition_ids,
      const Tensor& example_first_order_gradients,
      const Tensor& example_second_order_gradients,
      FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>*
          gradient_stats_accumulator) const override;

  void GenerateFeatureSplitCandidates(
      const LearnerConfig& learner_config, const std::vector<int32>& roots,
      const std::vector<NodeStats>& root_stats,
      const FeatureStatsAccumulator<GradientStats, GradientStatsAccumulator>&
          gradient_stats_accumulator,
      std::vector<FeatureSplitCandidate>* split_candidates) const override;

 protected:
  const int32 feature_column_;
  TTypes<int64>::ConstMatrix indices_;
  TTypes<int64>::ConstVec values_;
};

}  // namespace stochastic
}  // namespace learner
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_LEARNER_STOCHASTIC_HANDLERS_CATEGORICAL_FEATURE_COLUMN_HANDLER_H_
