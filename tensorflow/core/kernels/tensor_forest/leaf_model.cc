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

namespace tensorflow {

using decision_trees::Leaf;

std::unique_ptr<LeafModelOperator>
LeafModelOperatorFactory::CreateLeafModelOperator(
    const LeafModelType& model_type, const int32& num_output) {
  switch (model_type) {
    case CLASSIFICATION:
      return std::unique_ptr<LeafModelOperator>(
          new ClassificationLeafModelOperator(num_output));

    case REGRESSION:
      return std::unique_ptr<LeafModelOperator>(
          new RegressionLeafModelOperator(num_output));

    default:
      LOG(ERROR) << "Unknown model operator: " << model_type;
      return nullptr;
  }
}

float LeafModelOperator::GetOutputValue(const decision_trees::Leaf& leaf,
                                        int32 o) const {
  return leaf.vector().value(o).float_value();
}

void LeafModelOperator::InitModel(Leaf* leaf) const {
  for (int i = 0; i < num_output_; ++i) {
    leaf->mutable_vector()->add_value();
  }
}

// ------------------------ classification ----------------------------- //

void ClassificationLeafModelOperator::UpdateModel(Leaf* leaf,
                                                  const DenseTensorType& target,
                                                  int example) const {
  const int32 int_label =static_cast<int32>((*target)(example, 0));
  QCHECK_LT(int_label, num_output_)
      << "Got label greater than indicated number of classes. Is "
         "num_output set correctly?";
  QCHECK_GE(int_label, 0);
  auto* val = leaf->mutable_vector()->mutable_value(int_label);
  val->set_float_value(val->float_value() + 1.0);
}

void ClassificationLeafModelOperator::ExportModel(
    const tensor_forest::LeafStat& stat, tensor_forest::Leaf* leaf) const {
  *leaf->mutable_vector() = stat.classification().dense_counts();
}

// ------------------------ Regression ----------------------------- //

void RegressionLeafModelOperator::ExportModel(
    const tensor_forest::LeafStat& stat, tensor_forest::Leaf* leaf) const {
  leaf->clear_vector();
  for (int i = 0; i < num_output_; ++i) {
    const float new_val =
        stat.regression().mean_output().value(i).float_value() /
        stat.weight_sum();
    leaf->mutable_vector()->add_value()->set_float_value(new_val);
  }
}

}  // namespace tensorflow
}  // namespace tensorflow
