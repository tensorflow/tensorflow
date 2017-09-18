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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_GROW_STATS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_GROW_STATS_H_
#include <unordered_map>
#include <vector>

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/decision_node_evaluator.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"

namespace tensorflow {
namespace tensorforest {

// Base class for tracking stats necessary to split a leaf.
// Holds and tracks stats for every candidate split.
class GrowStats {
 public:
  virtual ~GrowStats() {}
  // Perform any initialization.
  virtual void Initialize() = 0;

  // Add an example to any stats being collected.
  virtual void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                          const InputTarget* target, int example) = 0;

  // Fill in the best split, return false if none were valid.
  virtual bool BestSplit(SplitCandidate* best) const = 0;

  // Return true if this leaf is finished splitting.
  virtual bool IsFinished() const = 0;

  // Get the split_num BinaryNode.
  const decision_trees::BinaryNode& Split(int split_num) const {
    return splits_[split_num];
  }

  // Clear all state.
  virtual void Clear() {
    weight_sum_ = 0;
    splits_.clear();
    evaluators_.clear();
    ClearInternal();
  }

  virtual void ExtractFromProto(const FertileSlot& slot) = 0;
  virtual void PackToProto(FertileSlot* slot) const = 0;

  // Add split to the list of candidate splits.
  void AddSplit(const decision_trees::BinaryNode& split,
                const std::unique_ptr<TensorDataSet>& input_data,
                const InputTarget* target, int example);
  virtual void AdditionalInitializationExample(
      const std::unique_ptr<TensorDataSet>& input_data,
      const InputTarget* target, int example) {}
  void RemoveSplit(int split_num);

  int num_splits() const {
    return splits_.size();
  }

  float weight_sum() const {
    return weight_sum_;
  }

  virtual bool IsInitialized() const {
    return weight_sum_ > 0 || splits_.size() == num_splits_to_consider_;
  }

  int32 depth() const {
    return depth_;
  }

 protected:
  GrowStats(const TensorForestParams& params, int32 depth);

  // Function called by AddSplit for subclasses to initialize stats for a split.
  virtual void AddSplitStats(const InputTarget* target, int example) = 0;

  virtual void RemoveSplitStats(int split_num) = 0;

  // Function called by Clear for subclasses to clear their state.
  virtual void ClearInternal() = 0;

  std::vector<decision_trees::BinaryNode> splits_;
  std::vector<std::unique_ptr<DecisionNodeEvaluator>> evaluators_;

  float weight_sum_;

  const int32 depth_;

  const TensorForestParams& params_;

  // We cache these because they're used often.
  const int split_after_samples_;
  const int num_splits_to_consider_;

  const int32 num_outputs_;
};

// Don't track anything, useful for systems that want to track split
// candidates but train the model in some other way.
class SimpleStats : public GrowStats {
 public:
  SimpleStats(const TensorForestParams& params, int32 depth)
      : GrowStats(params, depth) {}
  void Initialize() override {}

  void ExtractFromProto(const FertileSlot& slot) override {}
  void PackToProto(FertileSlot* slot) const override {}

  void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                  const InputTarget* target, int example) override {
    weight_sum_ += target->GetTargetWeight(example);
  }

  bool BestSplit(SplitCandidate* best) const override { return false; }

  bool IsFinished() const override {
    return weight_sum_ >= split_after_samples_;
  }

 protected:
  void AddSplitStats(const InputTarget* target, int example) override {}
  void RemoveSplitStats(int split_num) override {}
  void ClearInternal() override {}
};

// Tracks the sum and square of one side of a split for each Gini calculation.
class RunningGiniScores {
 public:
  float sum(int split) const { return sum_[split]; }
  float square(int split) const { return square_[split]; }

  void update(int split, float old_val, float weight) {
    sum_[split] += weight;
    const float new_val = old_val + weight;
    square_[split] = square_[split] - old_val * old_val + new_val * new_val;
  }

  void add_split() {
    sum_.push_back(0);
    square_.push_back(0);
  }

  void remove_split(int i) {
    sum_.erase(sum_.begin() + i);
    square_.erase(square_.begin() + i);
  }

 private:
  std::vector<float> sum_;
  std::vector<float> square_;
};

class ClassificationStats : public GrowStats {
 public:
  ClassificationStats(const TensorForestParams& params, int32 depth);

  bool IsFinished() const override;

  void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                  const InputTarget* target, int example) override;

  void AdditionalInitializationExample(
      const std::unique_ptr<TensorDataSet>& input_data,
      const InputTarget* target, int example) override;

  bool IsInitialized() const override {
    return weight_sum_ > 0 || (splits_.size() == num_splits_to_consider_ &&
                               half_initialized_splits_.empty());
  }

 protected:
  virtual float GiniScore(int split, float* left_sum,
                          float* right_sum) const = 0;
  virtual int num_outputs_seen() const = 0;
  virtual float left_count(int split, int class_num) const = 0;
  virtual float right_count(int split, int class_num) const = 0;

  virtual void ClassificationAddLeftExample(
      int split, int64 int_label, float weight) = 0;
  virtual void ClassificationAddTotalExample(int64 int_label, float weight) = 0;

  virtual void ClassificationAddSplitStats() = 0;
  virtual void ClassificationRemoveSplitStats(int split) = 0;

  void AddSplitStats(const InputTarget* target, int example) override {
    if (left_gini_ != nullptr) {
      left_gini_->add_split();
      right_gini_->add_split();
    }
    if (params_.initialize_average_splits()) {
      if (splits_[splits_.size() - 1].has_inequality_left_child_test()) {
        half_initialized_splits_[splits_.size() - 1] =
            target->GetTargetAsClassIndex(example, 0);
      }
    }
    ClassificationAddSplitStats();
  }
  void RemoveSplitStats(int split) override {
    if (left_gini_ != nullptr) {
      left_gini_->remove_split(split);
      right_gini_->remove_split(split);
    }
    ClassificationRemoveSplitStats(split);
  }

  // Virtual so we can override these to test.
  virtual void CheckFinishEarly();
  virtual void CheckFinishEarlyHoeffding();
  virtual void CheckFinishEarlyBootstrap();

  virtual void CheckPrune();

  // Implement SplitPruningStrategyType::SPLIT_PRUNE_HOEFFDING.
  void CheckPruneHoeffding();

  // Return the gini score, possibly being calculated from sums and squares
  // saved in left_gini_ and right_gini_, otherwise calculated from raw counts.
  float MaybeCachedGiniScore(int split, float* left_sum,
                             float* right_sum) const;

  // Initialize the sum and squares of left_gini_ and right_gini_ for given
  // split and value (being extracted from a proto), if left_gini_ isn't null.
  void MaybeInitializeRunningCount(int split, float val) {
    if (left_gini_ != nullptr) {
      left_gini_->update(split, 0, val);
      right_gini_->update(split, 0, val);
    }
  }

  int NumBootstrapSamples() const;

  // Populate *weights with the smoothed per-class frequencies needed to
  // initialize a DistributionSampler.
  void MakeBootstrapWeights(int index, std::vector<float>* weights);

  // Accessors for RunningGiniScores objects, for testing.
  virtual const std::unique_ptr<RunningGiniScores>& get_left_gini() const {
    return left_gini_;
  }
  virtual const std::unique_ptr<RunningGiniScores>& get_right_gini() const {
    return right_gini_;
  }

 private:
  // Tracks how many check_every_samples epochs we've seen go by in weight_sum.
  int32 finish_sample_epoch_;
  int32 finish_check_every_;
  int32 prune_sample_epoch_;
  int32 prune_check_every_;
  bool finish_early_;
  int32 min_split_samples_;
  float dominate_fraction_;
  float prune_fraction_;

  // When using SPLIT_PRUNE_HOEFFDING, we precompute and store
  // 0.5 * ln(1 / (1.0 - dominate_fraction_)).
  float half_ln_dominate_frac_;

  std::unique_ptr<random::PhiloxRandom> single_rand_;
  std::unique_ptr<random::SimplePhilox> rng_;

  std::unique_ptr<RunningGiniScores> left_gini_;
  std::unique_ptr<RunningGiniScores> right_gini_;

  // Stores split number -> class that was first seen.
  std::unordered_map<int, int32> half_initialized_splits_;
};

// Tracks classification stats by storing class counts densely.
class DenseClassificationGrowStats : public ClassificationStats {
 public:
  DenseClassificationGrowStats(const TensorForestParams& params, int32 depth)
      : ClassificationStats(params, depth) {}

  void Initialize() override {
    Clear();
    total_counts_.resize(num_outputs_);
  }

  void ExtractFromProto(const FertileSlot& slot) override;
  void PackToProto(FertileSlot* slot) const override;

  bool BestSplit(SplitCandidate* best) const override;

 protected:
  void ClassificationAddSplitStats() override {
    left_counts_.resize(num_outputs_ * num_splits());
  }
  void ClassificationRemoveSplitStats(int split_num) override {
    left_counts_.erase(left_counts_.begin() + num_outputs_ * split_num,
                       left_counts_.begin() + num_outputs_ * (split_num + 1));
  }
  void ClearInternal() override {
    total_counts_.clear();
    left_counts_.clear();
    num_outputs_seen_ = 0;
  }

  int num_outputs_seen() const override {
    return num_outputs_seen_;
  }

  void ClassificationAddLeftExample(int split, int64 int_label,
                                    float weight) override {
    mutable_left_count(split, int_label) += weight;
  }
  void ClassificationAddTotalExample(int64 int_label, float weight) override {
    num_outputs_seen_ += total_counts_[int_label] == 0 && weight > 0;
    total_counts_[int_label] += weight;
  }

  float GiniScore(int split, float* left_sum, float* right_sum) const override;

  float left_count(int split, int class_num) const override {
    return left_counts_[split * num_outputs_ + class_num];
  }
  float right_count(int split, int class_num) const override {
    return total_counts_[class_num] -
           left_counts_[split * num_outputs_ + class_num];
  }

 private:
  inline float& mutable_left_count(int split, int class_num) {
    return left_counts_[split * num_outputs_ + class_num];
  }
  // Total class counts seen at this leaf
  std::vector<float> total_counts_;

  // Also track the number of classes seen for not splitting pure leaves.
  int num_outputs_seen_;

  // Left-branch taken class counts at this leaf for each split.
  // This is a flat vector for memory-performance reasons.
  // left_counts_[i * num_outputs_ + j] has the j-th class count for split i.
  std::vector<float> left_counts_;
};

// Tracks classification stats by storing class counts sparsely.
class SparseClassificationGrowStats : public ClassificationStats {
 public:
  SparseClassificationGrowStats(const TensorForestParams& params, int32 depth)
      : ClassificationStats(params, depth) {}

  void Initialize() override {
    Clear();
  }

  void ExtractFromProto(const FertileSlot& slot) override;
  void PackToProto(FertileSlot* slot) const override;

  bool BestSplit(SplitCandidate* best) const override;

 protected:
  void ClassificationAddSplitStats() override {
    left_counts_.resize(num_splits());
  }
  void ClassificationRemoveSplitStats(int split_num) override {
    left_counts_.erase(left_counts_.begin() + split_num,
                       left_counts_.begin() + (split_num + 1));
  }
  void ClearInternal() override {
    total_counts_.clear();
    left_counts_.clear();
  }

  int num_outputs_seen() const override { return total_counts_.size(); }

  void ClassificationAddLeftExample(int split, int64 int_label,
                                    float weight) override {
    left_counts_[split][int_label] += weight;
  }
  void ClassificationAddTotalExample(int64 int_label, float weight) override {
    total_counts_[int_label] += weight;
  }

  float GiniScore(int split, float* left_sum, float* right_sum) const override;

  float left_count(int split, int class_num) const override {
    return left_counts_[split].at(class_num);
  }
  float right_count(int split, int class_num) const override {
    return total_counts_.at(class_num) - left_counts_[split].at(class_num);
  }

 private:
  // Total class counts seen at this leaf
  std::unordered_map<int, float> total_counts_;

  // Left-branch taken class counts at this leaf for each split.
  // left_counts_[i][j] has the j-th class count for split i.
  std::vector<std::unordered_map<int, float>> left_counts_;
};

// Tracks regression stats using least-squares minimization.
class LeastSquaresRegressionGrowStats : public GrowStats {
 public:
  LeastSquaresRegressionGrowStats(const TensorForestParams& params, int32 depth)
      : GrowStats(params, depth) {}

  void Initialize() override {
    Clear();
    total_sum_.resize(num_outputs_);
    total_sum_squares_.resize(num_outputs_);
  }

  void ExtractFromProto(const FertileSlot& slot) override;
  void PackToProto(FertileSlot* slot) const override;

  void AddExample(const std::unique_ptr<TensorDataSet>& input_data,
                  const InputTarget* target, int example) override;
  bool BestSplit(SplitCandidate* best) const override;
  bool IsFinished() const override;

 protected:
  // Returns the variance of split.
  float SplitVariance(int split) const;

  void AddSplitStats(const InputTarget* target, int example) override {
    left_sums_.resize(num_outputs_ * num_splits());
    left_squares_.resize(num_outputs_ * num_splits());
    left_counts_.push_back(0);
  }
  void RemoveSplitStats(int split_num) override {
    left_sums_.erase(left_sums_.begin() + num_outputs_ * split_num,
                       left_sums_.begin() + num_outputs_ * (split_num + 1));
    left_squares_.erase(left_squares_.begin() + num_outputs_ * split_num,
                       left_squares_.begin() + num_outputs_ * (split_num + 1));
    left_counts_.erase(left_counts_.begin() + split_num,
                       left_counts_.begin() + (split_num + 1));
  }

  void ClearInternal() override {
    total_sum_.clear();
    total_sum_squares_.clear();
    left_sums_.clear();
    left_squares_.clear();
  }

 private:
  // Convenience methods for accessing the flat count vectors.
  inline const float& left_sum(int split, int output_num) const {
    return left_sums_[split * num_outputs_ + output_num];
  }
  inline float& left_sum(int split, int output_num) {
    return left_sums_[split * num_outputs_ + output_num];
  }
  inline const float& left_square(int split, int output_num) const {
    return left_squares_[split * num_outputs_ + output_num];
  }
  inline float& left_square(int split, int output_num) {
    return left_squares_[split * num_outputs_ + output_num];
  }

  // Total sums and squares seen at this leaf.
  // sum[i] is the sum of the i-th output.
  std::vector<float> total_sum_;
  std::vector<float> total_sum_squares_;

  // Per-split sums and squares, stored flat for performance.
  // left_sums_[i * num_outputs_ + j] has the j-th sum for split i.
  std::vector<float> left_sums_;
  std::vector<float> left_squares_;

  // The number of example seen at each split.
  std::vector<int64> left_counts_;
};


}  // namespace tensorforest
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_GROW_STATS_H_
