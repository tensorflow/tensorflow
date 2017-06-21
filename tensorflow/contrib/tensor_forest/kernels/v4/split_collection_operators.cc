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
#include "tensorflow/contrib/tensor_forest/kernels/v4/split_collection_operators.h"

#include <cfloat>

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/tree_utils.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/stat_utils.h"

namespace tensorflow {
namespace tensorforest {

std::unique_ptr<SplitCollectionOperator>
SplitCollectionOperatorFactory::CreateSplitCollectionOperator(
    const TensorForestParams& params) {
  switch (params.collection_type()) {
    case COLLECTION_BASIC:
      return std::unique_ptr<SplitCollectionOperator>(
          new SplitCollectionOperator(params));

    case GRAPH_RUNNER_COLLECTION:
      return std::unique_ptr<SplitCollectionOperator>(
          new GraphRunnerSplitCollectionOperator(params));

    default:
      LOG(ERROR) << "Unknown split collection operator: "
                 << params.collection_type();
      return nullptr;
  }
}

std::unique_ptr<GrowStats> SplitCollectionOperator::CreateGrowStats(
    int32 node_id, int32 depth) const {
  switch (params_.stats_type()) {
    case STATS_DENSE_GINI:
      return std::unique_ptr<GrowStats>(
          new DenseClassificationGrowStats(params_, depth));

    case STATS_SPARSE_GINI:
      return std::unique_ptr<GrowStats>(
          new SparseClassificationGrowStats(params_, depth));

    case STATS_LEAST_SQUARES_REGRESSION:
      return std::unique_ptr<GrowStats>(new LeastSquaresRegressionGrowStats(
          params_, depth));

    default:
      LOG(ERROR) << "Unknown grow stats type: " << params_.stats_type();
      return nullptr;
  }
}

void SplitCollectionOperator::ExtractFromProto(
    const FertileStats& stats_proto) {
  for (int i = 0; i < stats_proto.node_to_slot_size(); ++i) {
    const auto& slot = stats_proto.node_to_slot(i);
    stats_[slot.node_id()] = CreateGrowStats(slot.node_id(), slot.depth());
    stats_[slot.node_id()]->ExtractFromProto(slot);
  }
}

void SplitCollectionOperator::PackToProto(FertileStats* stats_proto) const {
  for (int i = 0; i < stats_proto->node_to_slot_size(); ++i) {
    auto* new_slot = stats_proto->mutable_node_to_slot(i);
    const auto& stats = stats_.at(new_slot->node_id());
    if (params_.checkpoint_stats()) {
      stats->PackToProto(new_slot);
    }
    new_slot->set_depth(stats->depth());
  }
}

void SplitCollectionOperator::InitializeSlot(int32 node_id, int32 depth) {
  stats_[node_id] = std::unique_ptr<GrowStats>(CreateGrowStats(node_id, depth));
  stats_[node_id]->Initialize();
}

void SplitCollectionOperator::AddExample(
    const std::unique_ptr<TensorDataSet>& input_data, const InputTarget* target,
    const std::vector<int>& examples, int32 node_id) const {
  auto* slot = stats_.at(node_id).get();
  for (int example : examples) {
    slot->AddExample(input_data, target, example);
  }
}

bool SplitCollectionOperator::IsInitialized(int32 node_id) const {
  return stats_.at(node_id)->IsInitialized();
}

void SplitCollectionOperator::CreateAndInitializeCandidateWithExample(
    const std::unique_ptr<TensorDataSet>& input_data, int example,
    int32 node_id) const {
  // Assumes split_initializations_per_input == 1.
  decision_trees::BinaryNode split;
  float bias;
  int type;
  decision_trees::FeatureId feature_id;
  input_data->RandomSample(example, &feature_id, &bias, &type);

  if (type == kDataFloat) {
    decision_trees::InequalityTest* test =
        split.mutable_inequality_left_child_test();
    *test->mutable_feature_id() = feature_id;
    test->mutable_threshold()->set_float_value(bias);
    test->set_type(params_.inequality_test_type());
  } else if (type == kDataCategorical) {
    decision_trees::MatchingValuesTest test;
    *test.mutable_feature_id() = feature_id;
    test.add_value()->set_float_value(bias);
    split.mutable_custom_left_child_test()->PackFrom(test);
  } else {
    LOG(ERROR) << "Unknown feature type " << type << ", not sure which "
               << "node type to use.";
  }
  stats_.at(node_id)->AddSplit(split);
}

bool SplitCollectionOperator::BestSplit(int32 node_id,
                                        SplitCandidate* best,
                                        int32* depth) const {
  auto* slot = stats_.at(node_id).get();
  *depth = slot->depth();
  return slot->BestSplit(best);
}

// -------------------------------- GraphRunner ------------------ //

std::unique_ptr<GrowStats> GraphRunnerSplitCollectionOperator::CreateGrowStats(
    int32 node_id, int32 depth) const {
  return std::unique_ptr<GrowStats>(new SimpleStats(params_, depth));
}

int64 GraphRunnerSplitCollectionOperator::UniqueId(int32 node_id,
                                                   int32 split_id) const {
  return node_id * num_splits_to_consider_ + split_id;
}

bool GraphRunnerSplitCollectionOperator::BestSplit(int32 node_id,
                                                   SplitCandidate* best,
                                                   int32* depth) const {
  float min_score = FLT_MAX;
  int best_index = -1;
  auto* slot = stats_.at(node_id).get();
  *depth = slot->depth();
  for (int i = 0; i < slot->num_splits(); ++i) {
    // TODO(gilberth): Support uselessness.
    auto& runner = runners_[UniqueId(node_id, i)];
    const float split_score = runner->SplitScore();
    if (split_score < min_score) {
      min_score = split_score;
      best_index = i;
    }
  }

  // This could happen if all the splits are useless.
  if (best_index < 0) {
    return false;
  }

  // Fill in split info and left/right stats to initialize models with.
  *best = SplitCandidate();
  auto& runner = runners_[UniqueId(node_id, best_index)];
  runner->GetLeftStats(best->mutable_left_stats());
  runner->GetRightStats(best->mutable_right_stats());
  runner->GetSplit(best->mutable_split());
  return true;
}

void GraphRunnerSplitCollectionOperator::AddExample(
    const std::unique_ptr<TensorDataSet>& input_data, const InputTarget* target,
    const std::vector<int>& examples, int32 node_id) const {
  // Build input Tensors.
  int size = examples.size();
  Tensor examples_t(tensorflow::DT_INT32, TensorShape({size}));
  auto ex_data = examples_t.flat<int32>();
  std::copy(examples.begin(), examples.end(), ex_data.data());

  const TensorInputTarget* tensor_target =
      dynamic_cast<const TensorInputTarget*>(target);
  CHECK_NOTNULL(tensor_target);

  const Tensor& data_t = input_data->original_tensor();
  const Tensor& target_t = tensor_target->original_tensor();

  // Add to candidates.
  auto* slot = stats_.at(node_id).get();
  for (int i = 0; i < slot->num_splits(); ++i) {
    auto& runner = runners_[UniqueId(node_id, i)];
    runner->AddExample(data_t, target_t, examples_t);
  }

  // Update simple weight sums so we know when we're done.
  for (int example : examples) {
    slot->AddExample(input_data, target, example);
  }
}

void GraphRunnerSplitCollectionOperator::
    CreateAndInitializeCandidateWithExample(
        const std::unique_ptr<TensorDataSet>& input_data, int example,
        int32 node_id) const {
  auto* slot = stats_.at(node_id).get();
  int cand_num = slot->num_splits();
  const int64 unique_id = UniqueId(node_id, cand_num);

  decision_trees::BinaryNode split;

  decision_trees::InequalityTest* test =
      split.mutable_inequality_left_child_test();
  auto* oblique = test->mutable_oblique();
  for (int i = 0; i < features_per_node_; ++i) {
    float bias;
    int type;
    // This is really just a way to select a list of random features.
    // Also a way to warn the user that categoricals don't make sense here.
    input_data->RandomSample(example, oblique->add_features(), &bias, &type);

    if (type == kDataFloat) {
      test->set_type(decision_trees::InequalityTest::LESS_OR_EQUAL);

      // The comparison bias is assumed to be zero.
      test->mutable_threshold()->set_float_value(0);
    } else {
      LOG(ERROR) << "Categorical features not supported with this system.";
      return;
    }
  }

  slot->AddSplit(split);

  runners_[unique_id].reset(new CandidateGraphRunner(graph_dir_, split));
  runners_[unique_id]->Init();
}

void GraphRunnerSplitCollectionOperator::ClearSlot(int32 node_id) {
  SplitCollectionOperator::ClearSlot(node_id);
  for (int i = 0; i < num_splits_to_consider_; ++i) {
    runners_.erase(UniqueId(node_id, i));
  }
}

}  // namespace tensorforest
}  // namespace tensorflow
