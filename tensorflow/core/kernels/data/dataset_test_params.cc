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
    : output_dtypes(std::move(output_dtypes)),
      output_shapes(std::move(output_shapes)),
      node_name(std::move(node_name)),
      type(type) {}

bool DatasetParams::IsDatasetTensor(const Tensor& tensor) {
  return tensor.dtype() == DT_VARIANT &&
         TensorShapeUtils::IsScalar(tensor.shape());
}

BatchDatasetParams::BatchDatasetParams(
    std::shared_ptr<DatasetParams> input_dataset_params, int64 batch_size,
    bool drop_remainder, bool parallel_copy, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), DatasetParamsType::Batch),
      batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
      drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
      parallel_copy(parallel_copy) {
  input_dataset_params_group.emplace_back(
      std::make_pair(std::move(input_dataset_params), Tensor()));
}

Status BatchDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  inputs->reserve(input_dataset_params_group.size());
  for (auto& pair : input_dataset_params_group) {
    if (!IsDatasetTensor(pair.second)) {
      inputs->clear();
      return errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    } else {
      inputs->emplace_back(TensorValue(&pair.second));
    }
  }
  inputs->emplace_back(TensorValue(&batch_size));
  inputs->emplace_back(TensorValue(&drop_remainder));
  return Status::OK();
}

MapDatasetParams::MapDatasetParams(
    std::shared_ptr<DatasetParams> input_dataset_params,
    std::vector<Tensor> other_arguments,
    FunctionDefHelper::AttrValueWrapper func, std::vector<FunctionDef> func_lib,
    DataTypeVector type_arguments, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes,
    bool use_inter_op_parallelism, bool preserve_cardinality, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), DatasetParamsType::Map),
      other_arguments(std::move(other_arguments)),
      func(std::move(func)),
      func_lib(std::move(func_lib)),
      type_arguments(std::move(type_arguments)),
      use_inter_op_parallelism(use_inter_op_parallelism),
      preserve_cardinality(preserve_cardinality) {
  input_dataset_params_group.emplace_back(
      std::make_pair(std::move(input_dataset_params), Tensor()));
}

Status MapDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  inputs->reserve(input_dataset_params_group.size());
  for (auto& pair : input_dataset_params_group) {
    if (!IsDatasetTensor(pair.second)) {
      inputs->clear();
      return errors::Internal(
          "The input dataset is not populated as the dataset tensor yet.");
    } else {
      inputs->emplace_back(TensorValue(&pair.second));
    }
  }
  for (auto& argument : other_arguments) {
    inputs->emplace_back(TensorValue(&argument));
  }
  return Status::OK();
}

RangeDatasetParams::RangeDatasetParams(
    int64 start, int64 stop, int64 step, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), DatasetParamsType::Range),
      start(CreateTensor<int64>(TensorShape({}), {start})),
      stop(CreateTensor<int64>(TensorShape({}), {stop})),
      step(CreateTensor<int64>(TensorShape({}), {step})) {}

RangeDatasetParams::RangeDatasetParams(int64 start, int64 stop, int64 step)
    : DatasetParams({DT_INT64}, {PartialTensorShape({})}, "range_dataset",
                    DatasetParamsType::Range),
      start(CreateTensor<int64>(TensorShape({}), {start})),
      stop(CreateTensor<int64>(TensorShape({}), {stop})),
      step(CreateTensor<int64>(TensorShape({}), {step})) {}

Status RangeDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  *inputs = {TensorValue(&start), TensorValue(&stop), TensorValue(&step)};
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
