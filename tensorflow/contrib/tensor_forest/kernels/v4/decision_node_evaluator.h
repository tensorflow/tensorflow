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
#ifndef TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_DECISION_NODE_EVALUATOR_H_
#define TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_DECISION_NODE_EVALUATOR_H_

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/decision_trees/proto/generic_tree_model_extensions.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_data.h"

namespace tensorflow {
namespace tensorforest {


// Base class for evaluators of decision nodes that effectively copy proto
// contents into C++ structures for faster execution.
class DecisionNodeEvaluator {
 public:
  virtual ~DecisionNodeEvaluator() {}

  // Returns the index of the child node.
  virtual int32 Decide(const std::unique_ptr<TensorDataSet>& dataset,
                       int example) const = 0;
};

// An evaluator for binary decisions with left and right children.
class BinaryDecisionNodeEvaluator : public DecisionNodeEvaluator {
 protected:
  BinaryDecisionNodeEvaluator(int32 left, int32 right)
      : left_child_id_(left), right_child_id_(right) {}

  int32 left_child_id_;
  int32 right_child_id_;
};

// Evaluator for basic inequality decisions (f[x] <= T).
class InequalityDecisionNodeEvaluator : public BinaryDecisionNodeEvaluator {
 public:
  InequalityDecisionNodeEvaluator(const decision_trees::InequalityTest& test,
                                  int32 left, int32 right);

  int32 Decide(const std::unique_ptr<TensorDataSet>& dataset,
               int example) const override;

 protected:
  int32 feature_num_;
  float threshold_;

  // If decision is '<=' as opposed to '<'.
  bool include_equals_;
};

// Evalutor for splits with multiple weighted features.
class ObliqueInequalityDecisionNodeEvaluator
    : public BinaryDecisionNodeEvaluator {
 public:
  ObliqueInequalityDecisionNodeEvaluator(
      const decision_trees::InequalityTest& test, int32 left, int32 right);

  int32 Decide(const std::unique_ptr<TensorDataSet>& dataset,
               int example) const override;

 protected:
  std::vector<int32> feature_num_;
  std::vector<float> feature_weights_;
  float threshold_;
};

// Evaluator for contains-in-set decisions.  Also supports inverse (not-in-set).
class MatchingValuesDecisionNodeEvaluator : public BinaryDecisionNodeEvaluator {
 public:
  MatchingValuesDecisionNodeEvaluator(
      const decision_trees::MatchingValuesTest& test, int32 left, int32 right);

  int32 Decide(const std::unique_ptr<TensorDataSet>& dataset,
               int example) const override;

 protected:
  int32 feature_num_;
  std::vector<float> values_;
  bool inverse_;
};

std::unique_ptr<DecisionNodeEvaluator> CreateDecisionNodeEvaluator(
    const decision_trees::TreeNode& node);
std::unique_ptr<DecisionNodeEvaluator> CreateBinaryDecisionNodeEvaluator(
    const decision_trees::BinaryNode& node, int32 left, int32 right);

struct CandidateEvalatorCollection {
  std::vector<std::unique_ptr<DecisionNodeEvaluator>> splits;
};

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_DECISION_NODE_EVALUATOR_H_
