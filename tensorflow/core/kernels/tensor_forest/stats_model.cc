/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_RESOURCES_H_

#include <cfloat>
#include <queue>
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/stat_utils.h"
#include "tensorflow/core/lib/random/distribution_sampler.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {
// When creating evaluators for the split candidates, use these
// for the left and right return values.
static const int32 LEFT_INDEX = 0;
static const int32 RIGHT_INDEX = 1;

GrowStats::GrowStats(const int32& split_after_samples,
                     const int32& num_splits_to_consider)
    : weight_sum_(0),
      split_after_samples_(split_after_samples),
      num_splits_to_consider_(num_splits_to_consider),
      num_outputs_(params.num_outputs()) {}

void GrowStats::AddSplit(const decision_trees::BinaryNode& split,
                         const std::unique_ptr<TensorDataSet>& input_data,
                         const InputTarget* target, int example) {
  // It's possible that the split collection calls AddSplit, but we actually
  // have all the splits we need and are just waiting for them to be fully
  // initialized.
  if (splits_.size() < num_splits_to_consider_) {
    splits_.push_back(split);
    evaluators_.emplace_back(
        CreateBinaryDecisionNodeEvaluator(split, LEFT_INDEX, RIGHT_INDEX));
    AddSplitStats(target, example);
  }

  if (input_data != nullptr && target != nullptr &&
      params_.initialize_average_splits()) {
    AdditionalInitializationExample(input_data, target, example);
  }
}

void GrowStats::RemoveSplit(int split_num) {
  splits_.erase(splits_.begin() + split_num);
  evaluators_.erase(evaluators_.begin() + split_num);
  RemoveSplitStats(split_num);
}

// ------------------------ Classification --------------------------- //

ClassificationStats::ClassificationStats(const TensorForestParams& params,
                                         int32 depth)
    : GrowStats(params, depth) {
  // Early splitting params.
  min_split_samples_ = split_after_samples_;
  finish_sample_epoch_ = 1;
  finish_check_every_ = split_after_samples_ * 2;

  // Pruning params.
  prune_check_every_ = split_after_samples_ * 2;
  prune_sample_epoch_ = 1;

  single_rand_ = std::unique_ptr<random::PhiloxRandom>(
      new random::PhiloxRandom(random::New64()));
  rng_ = std::unique_ptr<random::SimplePhilox>(
      new random::SimplePhilox(single_rand_.get()));
}

void ClassificationStats::AdditionalInitializationExample(
    const std::unique_ptr<TensorDataSet>& input_data, const InputTarget* target,
    int example) {
  const int32 new_target = target->GetTargetAsClassIndex(example, 0);
  std::unordered_set<int> to_erase;
  for (auto it = half_initialized_splits_.begin();
       it != half_initialized_splits_.end(); ++it) {
    if (it->second != new_target) {
      auto& split = splits_[it->first];
      if (split.has_inequality_left_child_test()) {
        auto& test = split.inequality_left_child_test();
        auto* thresh =
            split.mutable_inequality_left_child_test()->mutable_threshold();
        if (test.has_feature_id()) {
          const float val =
              input_data->GetExampleValue(example, test.feature_id());
          thresh->set_float_value((thresh->float_value() + val) / 2);
        }
      }
      to_erase.insert(it->first);
    }
  }

  for (const int split_id : to_erase) {
    half_initialized_splits_.erase(split_id);
  }
}

bool ClassificationStats::IsFinished() const {
  bool basic = (weight_sum_ >= split_after_samples_) && !is_pure();
  return basic;
}

float ClassificationStats::MaybeCachedGiniScore(int split, float* left_sum,
                                                float* right_sum) const {
  if (left_gini_ == nullptr) {
    return GiniScore(split, left_sum, right_sum);
  } else {
    *left_sum = left_gini_->sum(split);
    const float left = WeightedSmoothedGini(
        *left_sum, left_gini_->square(split), num_outputs_);

    *right_sum = right_gini_->sum(split);
    const float right = WeightedSmoothedGini(
        *right_sum, right_gini_->square(split), num_outputs_);

    return left + right;
  }
}

void ClassificationStats::AddExample(
    const std::unique_ptr<TensorDataSet>& input_data, const InputTarget* target,
    int example) {
  const int64 int_label = target->GetTargetAsClassIndex(example, 0);
  const float weight = target->GetTargetWeight(example);

  for (int i = 0; i < num_splits(); ++i) {
    auto& eval = evaluators_[i];
    if (eval->Decide(input_data, example) == LEFT_INDEX) {
      if (left_gini_ != nullptr) {
        left_gini_->update(i, left_count(i, int_label), weight);
      }
      ClassificationAddLeftExample(i, int_label, weight);
    } else {
      if (right_gini_ != nullptr) {
        right_gini_->update(i, right_count(i, int_label), weight);
      }
      ClassificationAddRightExample(i, int_label, weight);
    }
  }

  ClassificationAddTotalExample(int_label, weight);

  weight_sum_ += weight;

  CheckFinishEarly();
  CheckPrune();
}

void ClassificationStats::CheckPrune() {
  if (params_.pruning_type().type() == SPLIT_PRUNE_NONE || IsFinished() ||
      weight_sum_ < prune_sample_epoch_ * prune_check_every_) {
    return;
  }
  ++prune_sample_epoch_;

  const int to_remove = num_splits() * prune_fraction_;
  if (to_remove <= 0) {
    return;
  }

  // pair ordering is first-then-second by default, no need for custom
  // comparison.  Use std::greater to make it a min-heap.
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      worst;

  // Track indices that are in the heap so we can iterate over them
  // by largest-first later.
  std::set<int> indices;

  for (int i = 0; i < num_splits(); ++i) {
    float left, right;
    const float split_score = MaybeCachedGiniScore(i, &left, &right);
    if (worst.size() < to_remove) {
      worst.push(std::pair<float, int>(split_score, i));
      indices.insert(i);
    } else if (worst.top().first < split_score) {
      indices.erase(worst.top().second);
      worst.pop();
      worst.push(std::pair<float, int>(split_score, i));
      indices.insert(i);
    }
  }

  // traverse indices from the back so that they are removed correctly.
  for (auto it = indices.rbegin(); it != indices.rend(); ++it) {
    RemoveSplit(*it);
  }
}

void ClassificationStats::CheckFinishEarly() {
  if (weight_sum_ < min_split_samples_ ||
      weight_sum_ < finish_sample_epoch_ * finish_check_every_) {
    return;
  }
  ++finish_sample_epoch_;
}

bool ClassificationStats::BestSplit(SplitCandidate* best) const {
  float min_score = FLT_MAX;
  int best_index = -1;
  float best_left_sum, best_right_sum;

  // Calculate sums.
  for (int i = 0; i < num_splits(); ++i) {
    float left_sum, right_sum;
    const float split_score = MaybeCachedGiniScore(i, &left_sum, &right_sum);
    // Find the lowest gini.
    if (left_sum > 0 && right_sum > 0 &&
        split_score < min_score) {  // useless check
      min_score = split_score;
      best_index = i;
      best_left_sum = left_sum;
      best_right_sum = right_sum;
    }
  }

  // This could happen if all the splits are useless.
  if (best_index < 0) {
    return false;
  }

  // Fill in stats to be used for leaf model.
  *best->mutable_split() = splits_[best_index];
  auto* left = best->mutable_left_stats();
  left->set_weight_sum(best_left_sum);
  auto* right = best->mutable_right_stats();
  right->set_weight_sum(best_right_sum);
  InitLeafClassStats(best_index, left, right);

  return true;
}

std::unique_ptr<GrowStats> SplitCollectionOperatorFactory::CreateGrowStats(
    LeafModelType& model_type, int32 depth) const {
  switch (model_type) {
    case CLASSIFICATION:
      return std::unique_ptr<GrowStats>(
          new DenseClassificationGrowStats(depth));

    case REGRESSION:
      return std::unique_ptr<GrowStats>(
          new LeastSquaresRegressionGrowStats(depth));

    default:
      LOG(ERROR) << "Unknown grow stats type: " << params_.stats_type();
      return nullptr;
  }
};
}  // namespace tensorflow
