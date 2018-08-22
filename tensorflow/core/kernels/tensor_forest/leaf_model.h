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
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_LEAF_MODEL_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_LEAF_MODEL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/tensor_forest/tensor_forest.pb.h"

namespace tensorflow {

using tensorforest::Leaf;
using tensorforest::LeafStat;

typedef TTypes<const float, 2>::ConstTensor DenseTensorType;

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
  float GetOutputValue(const Leaf& leaf, int32 o) const;

  // Initialize an empty Leaf model.
  void InitModel(Leaf* leaf) const;

  // Update the given Leaf's model with the given example.
  virtual void UpdateModel(Leaf* leaf,
                           const std::unique_ptr<DenseTensorType>& target,
                           int example) const = 0;

  virtual void ExportModel(const LeafStat& stat, Leaf* leaf) const = 0;

 protected:
  const int32& num_output_;
};

// LeafModelOperator that stores class counts in a dense vector.
class ClassificationLeafModelOperator : public LeafModelOperator {
 public:
  explicit ClassificationLeafModelOperator(const int32& num_output)
      : LeafModelOperator(num_output) {}

  void UpdateModel(Leaf* leaf, const std::unique_ptr<DenseTensorType>& target,
                   int example) const override;

  void ExportModel(const LeafStat& stat, Leaf* leaf) const override;
};

class LeafModelFactory {
 public:
  static std::unique_ptr<LeafModelOperator> CreateLeafModelOperator(
      const int32& model_type, const int32& num_output);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FOREST_LEAF_MODEL_H_
