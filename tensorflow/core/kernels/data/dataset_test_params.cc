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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

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

BatchDatasetParams* DatasetParams::Batch(
    int64 batch_size, bool drop_remainder, bool parallel_copy,
    DataTypeVector output_dtypes, std::vector<PartialTensorShape> output_shapes,
    string node_name) {
  return new BatchDatasetParams(this, batch_size, drop_remainder, parallel_copy,
                                std::move(output_dtypes),
                                std::move(output_shapes), std::move(node_name));
}

MapDatasetParams* DatasetParams::Map(
    std::vector<Tensor> other_arguments,
    FunctionDefHelper::AttrValueWrapper func, std::vector<FunctionDef> func_lib,
    DataTypeVector type_arguments, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes,
    bool use_inter_op_parallelism, bool preserve_cardinality,
    string node_name) {
  return new MapDatasetParams(
      this, std::move(other_arguments), std::move(func), std::move(func_lib),
      std::move(type_arguments), std::move(output_dtypes),
      std::move(output_shapes), use_inter_op_parallelism, preserve_cardinality,
      std::move(node_name));
}

SourceDatasetParams::SourceDatasetParams(
    DataTypeVector output_dtypes, std::vector<PartialTensorShape> output_shapes,
    string node_name, DatasetParamsType type)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), type) {}

UnaryDatasetParams::UnaryDatasetParams(
    DatasetParams* input_dataset_params, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name,
    DatasetParamsType type)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), type),
      input_dataset_params(input_dataset_params) {}

BinaryDatasetParams::BinaryDatasetParams(
    DatasetParams* input_dataset_params_0,
    DatasetParams* input_dataset_params_1, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name,
    DatasetParamsType type)
    : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                    std::move(node_name), type),
      input_dataset_params_0(input_dataset_params_0),
      input_dataset_params_1(input_dataset_params_1) {}

RangeDatasetParams::RangeDatasetParams(
    int64 start, int64 stop, int64 step, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : SourceDatasetParams(std::move(output_dtypes), std::move(output_shapes),
                          std::move(node_name), DatasetParamsType::Range),
      start(CreateTensor<int64>(TensorShape({}), {start})),
      stop(CreateTensor<int64>(TensorShape({}), {stop})),
      step(CreateTensor<int64>(TensorShape({}), {step})) {}

RangeDatasetParams::RangeDatasetParams(int64 start, int64 stop, int64 step)
    : SourceDatasetParams({DT_INT64}, {PartialTensorShape({})}, "",
                          DatasetParamsType::Range),
      start(CreateTensor<int64>(TensorShape({}), {start})),
      stop(CreateTensor<int64>(TensorShape({}), {stop})),
      step(CreateTensor<int64>(TensorShape({}), {step})) {}

Status RangeDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  *inputs = {TensorValue(&start), TensorValue(&stop), TensorValue(&step)};
  return Status::OK();
}

BatchDatasetParams::BatchDatasetParams(
    DatasetParams* input_dataset_params, int64 batch_size, bool drop_remainder,
    bool parallel_copy, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes, string node_name)
    : UnaryDatasetParams(input_dataset_params, std::move(output_dtypes),
                         std::move(output_shapes), std::move(node_name),
                         DatasetParamsType::Batch),
      batch_size(CreateTensor<int64>(TensorShape({}), {batch_size})),
      drop_remainder(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
      parallel_copy(parallel_copy) {}

Status BatchDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  if (!IsDatasetTensor(input_dataset)) {
    return errors::Internal(
        "The input dataset is not populated as the dataset tensor yet.");
  }
  *inputs = {TensorValue(&input_dataset), TensorValue(&batch_size),
             TensorValue(&drop_remainder)};
  return Status::OK();
}

MapDatasetParams::MapDatasetParams(
    DatasetParams* input_dataset_params, std::vector<Tensor> other_arguments,
    FunctionDefHelper::AttrValueWrapper func, std::vector<FunctionDef> func_lib,
    DataTypeVector type_arguments, DataTypeVector output_dtypes,
    std::vector<PartialTensorShape> output_shapes,
    bool use_inter_op_parallelism, bool preserve_cardinality, string node_name)
    : UnaryDatasetParams(input_dataset_params, std::move(output_dtypes),
                         std::move(output_shapes), std::move(node_name),
                         DatasetParamsType::Map),
      other_arguments(std::move(other_arguments)),
      func(std::move(func)),
      func_lib(std::move(func_lib)),
      type_arguments(std::move(type_arguments)),
      use_inter_op_parallelism(use_inter_op_parallelism),
      preserve_cardinality(preserve_cardinality) {}

Status MapDatasetParams::MakeInputs(
    gtl::InlinedVector<TensorValue, 4>* inputs) {
  if (!IsDatasetTensor(input_dataset)) {
    return tensorflow::errors::Internal(
        "The input dataset is not populated as the dataset tensor yet.");
  }
  *inputs = {TensorValue(&input_dataset)};
  for (auto& argument : other_arguments) {
    inputs->emplace_back(TensorValue(&argument));
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
