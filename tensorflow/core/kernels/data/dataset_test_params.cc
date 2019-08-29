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

#include "tensorflow/core/kernels/data/dataset_test_params.h"

namespace tensorflow {
namespace data {

string ToString(DatasetParamsType type) {
  switch (type) {
    case DatasetParamsType::Range:
      return "RangeDatasetParams";
    case DatasetParamsType::Batch:
      return "BatchDatasetParams";
    case DatasetParamsType::Map:
      return "MapDatasetParams";
  }
}

DatasetParams::DatasetParams(DataTypeVector output_dtypes,
                             std::vector<PartialTensorShape> output_shapes,
                             string node_name, DatasetParamsType type)
    : output_dtypes_(std::move(output_dtypes)),
      output_shapes_(std::move(output_shapes)),
      node_name_(std::move(node_name)),
      type_(type) {}

bool DatasetParams::IsDatasetTensor(const Tensor& tensor) {
  return tensor.dtype() == DT_VARIANT &&
         TensorShapeUtils::IsScalar(tensor.shape());
}

Status BatchDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
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
  inputs->emplace_back(TensorValue(&batch_size_));
  inputs->emplace_back(TensorValue(&drop_remainder_));
  return Status::OK();
}

Status BatchDatasetParams::MakeAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{BatchDatasetOp::kParallelCopy, parallel_copy_},
                  {BatchDatasetOp::kOutputTypes, output_dtypes_},
                  {BatchDatasetOp::kOutputShapes, output_shapes_}};
  return Status::OK();
}

Status MapDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
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

Status MapDatasetParams::MakeAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {
      {MapDatasetOp::kFunc, func_},
      {MapDatasetOp::kTarguments, type_arguments_},
      {MapDatasetOp::kOutputShapes, output_shapes_},
      {MapDatasetOp::kOutputTypes, output_dtypes_},
      {MapDatasetOp::kUseInterOpParallelism, use_inter_op_parallelism_},
      {MapDatasetOp::kPreserveCardinality, preserve_cardinality_}};
  return Status::OK();
}

int MapDatasetParams::num_of_other_arguments() const {
  return other_arguments_.size();
}

std::vector<FunctionDef> MapDatasetParams::func_lib() { return func_lib_; }

RangeDatasetParams::RangeDatasetParams(
    int64 start, int64 stop, int64 step, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), DatasetParamsType::Range),
      start_(CreateTensor<int64>(TensorShape({}), {start})),
      stop_(CreateTensor<int64>(TensorShape({}), {stop})),
      step_(CreateTensor<int64>(TensorShape({}), {step})) {}

RangeDatasetParams::RangeDatasetParams(int64 start, int64 stop, int64 step)
    : DatasetParams({DT_INT64}, {PartialTensorShape({})}, "range_dataset",
                    DatasetParamsType::Range),
      start_(CreateTensor<int64>(TensorShape({}), {start})),
      stop_(CreateTensor<int64>(TensorShape({}), {stop})),
      step_(CreateTensor<int64>(TensorShape({}), {step})) {}

Status RangeDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  *inputs = {TensorValue(&start_), TensorValue(&stop_), TensorValue(&step_)};
  return Status::OK();
}

Status RangeDatasetParams::MakeAttributes(AttributeVector* attr_vector) const {
  *attr_vector = {{RangeDatasetOp::kOutputTypes, output_dtypes_},
                  {RangeDatasetOp::kOutputShapes, output_shapes_}};
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
