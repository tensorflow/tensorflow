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
#include "tensorflow/contrib/tensor_forest/kernels/v4/leaf_model_operators.h"

namespace tensorflow {
namespace tensorforest {

std::unique_ptr<LeafModelOperator>
LeafModelOperatorFactory::CreateLeafModelOperator(
    const TensorForestParams& params) {
  switch (params.leaf_type()) {
    case MODEL_DENSE_CLASSIFICATION:
      return std::unique_ptr<LeafModelOperator>(
          new DenseClassificationLeafModelOperator(params));

    case MODEL_SPARSE_CLASSIFICATION:
      return std::unique_ptr<LeafModelOperator>(
          new SparseClassificationLeafModelOperator(params));

    case MODEL_SPARSE_OR_DENSE_CLASSIFICATION:
      return std::unique_ptr<LeafModelOperator>(
          new SparseOrDenseClassificationLeafModelOperator(params));

    case MODEL_REGRESSION:
      return std::unique_ptr<LeafModelOperator>(
          new RegressionLeafModelOperator(params));

    default:
      LOG(ERROR) << "Unknown model operator: " << params.leaf_type();
      return nullptr;
  }
}

// ------------------------ Dense ----------------------------- //
float DenseClassificationLeafModelOperator::GetOutputValue(
    const decision_trees::Leaf& leaf, int32 o) const {
  return leaf.vector().value(o).float_value();
}

void DenseClassificationLeafModelOperator::UpdateModel(
    LeafStat* leaf, const InputTarget* target,
    int example) const {
  const int32 int_label = target->GetTargetAsClassIndex(example, 0);
  auto* val = leaf->mutable_classification()->mutable_dense_counts()
      ->mutable_value(int_label);
  float weight = target->GetTargetWeight(example);
  val->set_float_value(val->float_value() + weight);
  leaf->set_weight_sum(leaf->weight_sum() + weight);
}

void DenseClassificationLeafModelOperator::InitModel(
    LeafStat* leaf) const {
  for (int i = 0; i < params_.num_outputs(); ++i) {
    leaf->mutable_classification()->mutable_dense_counts()->add_value();
  }
}

void DenseClassificationLeafModelOperator::ExportModel(
    const LeafStat& stat, decision_trees::Leaf* leaf) const {
  *leaf->mutable_vector() = stat.classification().dense_counts();
}

// ------------------------- Sparse -------------------------- //
float SparseClassificationLeafModelOperator::GetOutputValue(
    const decision_trees::Leaf& leaf, int32 o) const {
  const auto it = leaf.sparse_vector().sparse_value().find(o);
  if (it == leaf.sparse_vector().sparse_value().end()) {
    return 0;  // default value
  } else {
    return it->second.float_value();
  }
}

void SparseClassificationLeafModelOperator::UpdateModel(
    LeafStat* leaf, const InputTarget* target,
    int example) const {
  const int32 int_label = target->GetTargetAsClassIndex(example, 0);
  const float weight = target->GetTargetWeight(example);
  leaf->set_weight_sum(leaf->weight_sum() + weight);
  auto value_map = leaf->mutable_classification()->mutable_sparse_counts()
      ->mutable_sparse_value();
  auto it = value_map->find(int_label);
  if (it == value_map->end()) {
    (*value_map)[int_label].set_float_value(weight);
  } else {
    it->second.set_float_value(it->second.float_value() + weight);
  }
}

void SparseClassificationLeafModelOperator::ExportModel(
    const LeafStat& stat, decision_trees::Leaf* leaf) const {
  *leaf->mutable_sparse_vector() = stat.classification().sparse_counts();
}

// ------------------------- SparseOrDense -------------------------- //
float SparseOrDenseClassificationLeafModelOperator::GetOutputValue(
    const decision_trees::Leaf& leaf, int32 o) const {
  if (leaf.has_vector()) {
    return dense_->GetOutputValue(leaf, o);
  } else {
    return sparse_->GetOutputValue(leaf, o);
  }
}

void SparseOrDenseClassificationLeafModelOperator::UpdateModel(
    LeafStat* leaf, const InputTarget* target, int example) const {
  if (leaf->classification().has_dense_counts()) {
    return dense_->UpdateModel(leaf, target, example);
  } else {
    return sparse_->UpdateModel(leaf, target, example);
  }
}

void SparseOrDenseClassificationLeafModelOperator::ExportModel(
    const LeafStat& stat, decision_trees::Leaf* leaf) const {
  if (stat.classification().has_dense_counts()) {
    return dense_->ExportModel(stat, leaf);
  } else {
    return sparse_->ExportModel(stat, leaf);
  }
}

// ------------------------ Regression ----------------------------- //
float RegressionLeafModelOperator::GetOutputValue(
    const decision_trees::Leaf& leaf, int32 o) const {
  return leaf.vector().value(o).float_value();
}

void RegressionLeafModelOperator::InitModel(
    LeafStat* leaf) const {
  for (int i = 0; i < params_.num_outputs(); ++i) {
    leaf->mutable_regression()->mutable_mean_output()->add_value();
  }
}

void RegressionLeafModelOperator::ExportModel(
    const LeafStat& stat, decision_trees::Leaf* leaf) const {
  for (int i = 0; i < params_.num_outputs(); ++i) {
    const float new_val =
        stat.regression().mean_output().value(i).float_value() /
        stat.weight_sum();
    leaf->mutable_vector()->add_value()->set_float_value(new_val);
  }
}


}  // namespace tensorforest
}  // namespace tensorflow
