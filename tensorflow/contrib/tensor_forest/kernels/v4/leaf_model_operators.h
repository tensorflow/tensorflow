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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_LEAF_MODEL_OPERATORS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_LEAF_MODEL_OPERATORS_H_

#include "tensorflow/contrib/decision_trees/proto/generic_tree_model.pb.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/input_target.h"
#include "tensorflow/contrib/tensor_forest/kernels/v4/params.h"
#include "tensorflow/contrib/tensor_forest/proto/fertile_stats.pb.h"
#include "tensorflow/contrib/tensor_forest/proto/tensor_forest_params.pb.h"

namespace tensorflow {
namespace tensorforest {

// Abstract base class for classes that can initialize, get, and update leaf
// models.
class LeafModelOperator {
 public:
  // Number of outputs is interpreted differently for classification and
  // regression.  For classification, it's the number of possible classes.
  // For regression, it's the target dimensions.
  explicit LeafModelOperator(const TensorForestParams& params)
      : params_(params) {}
  virtual ~LeafModelOperator() {}

  // Returns the value of the requested output, which should be
  // in [0, num_outputs_).  For classification, it's the class count (weighted
  // number of instances seen).  For regression, it's e.g. the average value.
  virtual float GetOutputValue(const decision_trees::Leaf& leaf,
                               int32 o) const = 0;

  // Update the given Leaf's model with the given example.
  virtual void UpdateModel(decision_trees::Leaf* leaf,
                           const InputTarget* target, int example) const = 0;

  // Initialize an empty Leaf model.
  virtual void InitModel(decision_trees::Leaf* leaf) const = 0;

  virtual void ExportModel(const LeafStat& stat,
                           decision_trees::Leaf* leaf) const = 0;

 protected:
  const TensorForestParams& params_;
};

// LeafModelOperator that stores class counts in a dense vector.
class DenseClassificationLeafModelOperator : public LeafModelOperator {
 public:
  explicit DenseClassificationLeafModelOperator(
      const TensorForestParams& params)
      : LeafModelOperator(params) {}
  float GetOutputValue(const decision_trees::Leaf& leaf,
                       int32 o) const override;

  void UpdateModel(decision_trees::Leaf* leaf, const InputTarget* target,
                   int example) const override;

  void InitModel(decision_trees::Leaf* leaf) const override;

  void ExportModel(const LeafStat& stat,
                   decision_trees::Leaf* leaf) const override;
};

// LeafModelOperator that stores class counts sparsely in a map. Assumes default
// value for yet-unseen classes is 0.
class SparseClassificationLeafModelOperator : public LeafModelOperator {
 public:
  explicit SparseClassificationLeafModelOperator(
      const TensorForestParams& params)
      : LeafModelOperator(params) {}
  float GetOutputValue(const decision_trees::Leaf& leaf,
                       int32 o) const override;

  void UpdateModel(decision_trees::Leaf* leaf, const InputTarget* target,
                   int example) const override;

  void InitModel(decision_trees::Leaf* leaf) const override {}

  void ExportModel(const LeafStat& stat,
                   decision_trees::Leaf* leaf) const override;
};

class SparseOrDenseClassificationLeafModelOperator : public LeafModelOperator {
 public:
  explicit SparseOrDenseClassificationLeafModelOperator(
      const TensorForestParams& params)
      : LeafModelOperator(params),
        dense_(new DenseClassificationLeafModelOperator(params)),
        sparse_(new SparseClassificationLeafModelOperator(params)) {}
  float GetOutputValue(const decision_trees::Leaf& leaf,
                       int32 o) const override;

  void UpdateModel(decision_trees::Leaf* leaf, const InputTarget* target,
                   int example) const override;

  void InitModel(decision_trees::Leaf* leaf) const override {}

  void ExportModel(const LeafStat& stat,
                   decision_trees::Leaf* leaf) const override;

 protected:
  std::unique_ptr<DenseClassificationLeafModelOperator> dense_;
  std::unique_ptr<SparseClassificationLeafModelOperator> sparse_;
};

// LeafModelOperator that stores regression leaf models with constant-value
// prediction.
class RegressionLeafModelOperator : public LeafModelOperator {
 public:
  explicit RegressionLeafModelOperator(const TensorForestParams& params)
      : LeafModelOperator(params) {}
  float GetOutputValue(const decision_trees::Leaf& leaf,
                       int32 o) const override;

  // TODO(gilberth): Quick experimentation suggests it's not even worth
  // updating model and just using the seeded values.  Can add this in
  // with additional_data, though protobuf::Any is slow.  Maybe make it
  // optional.  Maybe make any update optional.
  void UpdateModel(decision_trees::Leaf* leaf, const InputTarget* target,
                   int example) const override {}

  void InitModel(decision_trees::Leaf* leaf) const override;

  void ExportModel(const LeafStat& stat,
                   decision_trees::Leaf* leaf) const override;
};

class LeafModelOperatorFactory {
 public:
  static std::unique_ptr<LeafModelOperator> CreateLeafModelOperator(
      const TensorForestParams& params);
};

}  // namespace tensorforest
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_LEAF_MODEL_OPERATORS_H_
