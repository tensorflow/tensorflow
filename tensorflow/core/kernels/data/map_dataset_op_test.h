/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_MAP_DATASET_OP_TEST_H
#define TENSORFLOW_CORE_KERNELS_DATA_MAP_DATASET_OP_TEST_H

#include "tensorflow/core/kernels/data/dataset_test_params.h"
#include "tensorflow/core/kernels/data/map_dataset_op.h"

namespace tensorflow {
namespace data {

class MapDatasetParams : public DatasetParams {
 public:
  template <typename T>
  MapDatasetParams(T input_dataset_params, std::vector<Tensor> other_arguments,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   DataTypeVector type_arguments, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   bool use_inter_op_parallelism, bool preserve_cardinality,
                   string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name), DatasetParamsType::Map),
        other_arguments_(std::move(other_arguments)),
        func_(std::move(func)),
        func_lib_(std::move(func_lib)),
        type_arguments_(std::move(type_arguments)),
        use_inter_op_parallelism_(use_inter_op_parallelism),
        preserve_cardinality_(preserve_cardinality) {
    auto input_dataset_params_ptr =
        std::make_shared<T>(std::move(input_dataset_params));
    input_dataset_params_group_.emplace_back(
        std::make_pair(std::move(input_dataset_params_ptr), Tensor()));
  }

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override {
    inputs->reserve(input_dataset_params_group_.size());
    for (auto& pair : input_dataset_params_group_) {
      if (!IsDatasetTensor(pair.second)) {
        inputs->clear();
        return errors::Internal(
            "The input dataset is not populated as the dataset tensor yet.");
      } else {
        inputs->emplace_back(TensorValue(&pair.second));
      }
    }
    for (auto& argument : other_arguments_) {
      inputs->emplace_back(TensorValue(&argument));
    }
    return Status::OK();
  }

  Status MakeInputPlaceholder(
      std::vector<string>* input_placeholder) const override {
    input_placeholder->emplace_back(MapDatasetOp::kInputDataset);
    for (int i = 0; i < other_arguments_.size(); ++i) {
      input_placeholder->emplace_back(
          absl::StrCat(MapDatasetOp::kOtherArguments, "_", i));
    }
    return Status::OK();
  }

  Status MakeAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {
        {MapDatasetOp::kFunc, func_},
        {MapDatasetOp::kTarguments, type_arguments_},
        {MapDatasetOp::kOutputShapes, output_shapes_},
        {MapDatasetOp::kOutputTypes, output_dtypes_},
        {MapDatasetOp::kUseInterOpParallelism, use_inter_op_parallelism_},
        {MapDatasetOp::kPreserveCardinality, preserve_cardinality_}};
    return Status::OK();
  }

  std::vector<FunctionDef> func_lib() const override { return func_lib_; }

  int num_of_other_arguments() const { return other_arguments_.size(); }

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
  bool use_inter_op_parallelism_;
  bool preserve_cardinality_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_MAP_DATASET_OP_TEST_H
