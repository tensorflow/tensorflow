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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_EVALUATOR_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_EVALUATOR_H_

#include "tensorflow/core/kernels/tensor_forest/leaf_model.h"
#include "tensorflow/core/kernels/tensor_forest/tensor_forest.pb.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

using tensorflow::strings::safe_strto32;
using tensorforest::InequalityTest;

// Class for evaluators of decision nodes that effectively copy proto
// contents into C++ structures for faster execution.
// An evaluator for binary decisions with left and right children.
// Evaluator for basic inequality decisions (f[x] <= T).
class BinaryDecisionNodeEvaluator {
 public:
  BinaryDecisionNodeEvaluator(const InequalityTest& test, int32 left,
                              int32 right)
      : BinaryDecisionNodeEvaluator(left, right) {
    CHECK(safe_strto32(test.feature_id().id().value(), &feature_num_))
        << "Invalid feature ID: [" << test.feature_id().id().value() << "]";
    threshold_ = test.threshold().float_value();
  }
  // Returns the index of the child node.
  int32 Decide(const std::unique_ptr<DenseTensorType>& dataset,
               int example) const {
    const float val = (*dataset)(example, feature_num_);
    if (val <= threshold_) {
      return left_child_id_;
    } else {
      return right_child_id_;
    }
  };

 protected:
  BinaryDecisionNodeEvaluator(int32 left, int32 right)
      : left_child_id_(left), right_child_id_(right) {}

  int32 left_child_id_;
  int32 right_child_id_;
  int32 feature_num_;
  float threshold_;
};

std::unique_ptr<BinaryDecisionNodeEvaluator> CreateBinaryDecisionNodeEvaluator(
    const tensorforest::TreeNode& node) {
  const tensorforest::BinaryNode& bnode = node.binary_node();
  int32 left = bnode.left_child_id().value();
  int32 right = bnode.right_child_id().value();
  const auto& test = bnode.inequality_left_child_test();
  return std::unique_ptr<BinaryDecisionNodeEvaluator>(
      new BinaryDecisionNodeEvaluator(test, left, right));
}
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_EVALUATOR_H_
