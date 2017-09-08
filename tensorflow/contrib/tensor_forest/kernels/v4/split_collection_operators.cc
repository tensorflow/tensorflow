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

std::unordered_map<int, CollectionCreator*>
    SplitCollectionOperatorFactory::factories_;  // NOLINT
REGISTER_SPLIT_COLLECTION(COLLECTION_BASIC, SplitCollectionOperator);

std::unique_ptr<SplitCollectionOperator>
SplitCollectionOperatorFactory::CreateSplitCollectionOperator(
    const TensorForestParams& params) {
  auto it = factories_.find(params.collection_type());
  if (it == factories_.end()) {
    LOG(ERROR) << "Unknown split collection operator: "
               << params.collection_type();
    return nullptr;
  } else {
    return it->second->Create(params);
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
  for (const auto& pair : stats_) {
    auto* new_slot = stats_proto->add_node_to_slot();
    new_slot->set_node_id(pair.first);
    if (params_.checkpoint_stats()) {
      pair.second->PackToProto(new_slot);
    }
    new_slot->set_depth(pair.second->depth());
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
  auto it = stats_.find(node_id);
  if (it == stats_.end()) {
    LOG(WARNING) << "IsInitialized called with unknown node_id = " << node_id;
    return false;
  }
  return it->second->IsInitialized();
}

void SplitCollectionOperator::CreateAndInitializeCandidateWithExample(
    const std::unique_ptr<TensorDataSet>& input_data, const InputTarget* target,
    int example, int32 node_id) const {
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
  stats_.at(node_id)->AddSplit(split, input_data, target, example);
}

bool SplitCollectionOperator::BestSplit(int32 node_id,
                                        SplitCandidate* best,
                                        int32* depth) const {
  auto* slot = stats_.at(node_id).get();
  *depth = slot->depth();
  return slot->BestSplit(best);
}
}  // namespace tensorforest
}  // namespace tensorflow
