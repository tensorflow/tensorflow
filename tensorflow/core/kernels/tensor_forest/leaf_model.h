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
#ifndef TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_RESOURCES_H_
#define TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_RESOURCES_H_

namespace tensorflow {

namespace {
enum LeafModelType { CLASSIFICATION = 0, REGRESSION = 1 };
}

class LeafModelOperator {
 public:
  // Number of outputs is interpreted differently for classification and
  // regression.  For classification, it's the number of possible classes.
  // For regression, it's the target dimensions.
  explicit LeafModelOperator(const int32& num_output)
      : num_output_(num_output) {}
  virtual ~LeafModelOperator() {}

  // Returns the value of the requested output, which should be
  // in [0, num_outputs_).  For classification, it's the class count (weighted
  // number of instances seen).  For regression, it's e.g. the average value.
  float GetOutputValue(const decision_trees::Leaf& leaf, int32 o) const;

  // Update the given Leaf's model with the given example.
  virtual void UpdateModel(decision_trees::Leaf* leaf,
                           const InputTarget* target, int example) const = 0;

  // Initialize an empty Leaf model.
  void InitModel(decision_trees::Leaf* leaf) const;

  virtual void ExportModel(const LeafStat& stat,
                           decision_trees::Leaf* leaf) const = 0;

 protected:
  const int32& num_output_;
};

// LeafModelOperator that stores class counts in a dense vector.
class ClassificationLeafModelOperator : public LeafModelOperator {
 public:
  void UpdateModel(decision_trees::Leaf* leaf, const InputTarget* target,
                   int example) const override;

  void ExportModel(const LeafStat& stat,
                   decision_trees::Leaf* leaf) const override;
};

// LeafModelOperator that stores regression leaf models with constant-value
// prediction.
class RegressionLeafModelOperator : public LeafModelOperator {
 public:
  // TODO(gilberth): Quick experimentation suggests it's not even worth
  // updating model and just using the seeded values.  Can add this in
  // with additional_data, though protobuf::Any is slow.  Maybe make it
  // optional.  Maybe make any update optional.
  void UpdateModel(decision_trees::Leaf* leaf, const InputTarget* target,
                   int example) const override {}

  void ExportModel(const LeafStat& stat,
                   decision_trees::Leaf* leaf) const override;
};

class LeafModelOperatorFactory {
 public:
  static std::unique_ptr<LeafModelOperator> CreateLeafModelOperator(
      const LeafModelType& model_type, const int32& num_output);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_TENSOR_FOREST_KERNELS_V4_LEAF_MODEL_OPERATORS_H_
