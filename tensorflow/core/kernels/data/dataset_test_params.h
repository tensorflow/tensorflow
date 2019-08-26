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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_PARAMS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_PARAMS_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/data/batch_dataset_op.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/map_dataset_op.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {
namespace data {

constexpr char kDefaultIteratorPrefix[] = "Iterator";

// Creates a tensor with the specified dtype, shape, and value.
template <typename T>
static Tensor CreateTensor(const TensorShape& input_shape,
                           const gtl::ArraySlice<T>& input_data) {
  Tensor tensor(DataTypeToEnum<T>::value, input_shape);
  test::FillValues<T>(&tensor, input_data);
  return tensor;
}

// Creates a vector of tensors with the specified dtype, shape, and values.
template <typename T>
std::vector<Tensor> CreateTensors(
    const TensorShape& shape, const std::vector<gtl::ArraySlice<T>>& values) {
  std::vector<Tensor> result;
  result.reserve(values.size());
  for (auto& value : values) {
    result.emplace_back(CreateTensor<T>(shape, value));
  }
  return result;
}

enum class DatasetParamsType {
  Range,
  Batch,
  Map,
};

string ToString(DatasetParamsType type);

class RangeDatasetParams;
class BatchDatasetParams;
class MapDatasetParams;

class DatasetParams {
 public:
  DatasetParams(DataTypeVector output_dtypes,
                std::vector<PartialTensorShape> output_shapes, string node_name,
                DatasetParamsType type);

  ~DatasetParams() {}

  // Returns the dataset inputs as a TensorValue vector.
  virtual Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) = 0;

  // Checks if the tensor is a dataset variant tensor.
  static bool IsDatasetTensor(const Tensor& tensor);

  // Used to store all the input dataset parameters and the dataset tensors
  // generated from the parameters.
  std::vector<std::pair<std::shared_ptr<DatasetParams>, Tensor>>
      input_dataset_params_group;
  DataTypeVector output_dtypes;
  std::vector<PartialTensorShape> output_shapes;
  string node_name;
  string iterator_prefix = kDefaultIteratorPrefix;
  DatasetParamsType type;
};

class BatchDatasetParams : public DatasetParams {
 public:
  BatchDatasetParams(std::shared_ptr<DatasetParams> input_dataset_params,
                     int64 batch_size, bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name);

  ~BatchDatasetParams() {}

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Tensor batch_size;
  Tensor drop_remainder;
  bool parallel_copy;
};

class MapDatasetParams : public DatasetParams {
 public:
  MapDatasetParams(std::shared_ptr<DatasetParams> input_dataset_params,
                   std::vector<Tensor> other_arguments,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   DataTypeVector type_arguments, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   bool use_inter_op_parallelism, bool preserve_cardinality,
                   string node_name);

  ~MapDatasetParams(){};

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  std::vector<Tensor> other_arguments;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  DataTypeVector type_arguments;
  bool use_inter_op_parallelism;
  bool preserve_cardinality;
};

class RangeDatasetParams : public DatasetParams {
 public:
  RangeDatasetParams(int64 start, int64 stop, int64 step,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name);

  RangeDatasetParams(int64 start, int64 stop, int64 step);

  ~RangeDatasetParams(){};

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Tensor start;
  Tensor stop;
  Tensor step;
};

static std::shared_ptr<BatchDatasetParams> Batch(
    std::shared_ptr<DatasetParams> input_dataset_params, int64 batch_size,
    bool drop_remainder, bool parallel_copy,
    const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes, string node_name) {
  return std::make_shared<BatchDatasetParams>(
      input_dataset_params, batch_size, drop_remainder, parallel_copy,
      output_dtypes, output_shapes, node_name);
}

static std::shared_ptr<MapDatasetParams> Map(
    std::shared_ptr<DatasetParams> input_dataset_params,
    const std::vector<Tensor>& other_arguments,
    const FunctionDefHelper::AttrValueWrapper& func,
    const std::vector<FunctionDef>& func_lib, DataTypeVector type_arguments,
    const DataTypeVector& output_dtypes,
    const std::vector<PartialTensorShape>& output_shapes,
    bool use_inter_op_parallelism, bool preserve_cardinality,
    string node_name) {
  return std::make_shared<MapDatasetParams>(
      input_dataset_params, other_arguments, func, func_lib, type_arguments,
      output_dtypes, output_shapes, use_inter_op_parallelism,
      preserve_cardinality, node_name);
}

static std::shared_ptr<RangeDatasetParams> Range(int64 start, int64 stop,
                                                 int64 step) {
  return std::make_shared<RangeDatasetParams>(start, stop, step);
}

class DatasetParamsBuilder {
 public:
  DatasetParamsBuilder() = default;

  DatasetParamsBuilder Range(int64 start, int64 stop, int64 step) {
    string node_name = GetNodeName(RangeDatasetOp::kDatasetType);
    auto range_dataset_params = new RangeDatasetParams(
        start, stop, step, {DT_INT64}, {PartialTensorShape({})}, node_name);
    dataset_params_.reset(range_dataset_params);
    return *this;
  }

  DatasetParamsBuilder Batch(
      int64 batch_size, bool drop_remainder, bool parallel_copy,
      const DataTypeVector& output_dtypes,
      const std::vector<PartialTensorShape>& output_shapes) {
    string node_name = GetNodeName(BatchDatasetOp::kDatasetType);
    auto batch_dataset_params = new BatchDatasetParams(
        dataset_params_, batch_size, drop_remainder, parallel_copy,
        output_dtypes, output_shapes, node_name);
    dataset_params_.reset(batch_dataset_params);
    return *this;
  }

  DatasetParamsBuilder Map(const std::vector<Tensor>& other_arguments,
                           const FunctionDefHelper::AttrValueWrapper& func,
                           const std::vector<FunctionDef>& func_lib,
                           const DataTypeVector& type_arguments,
                           const DataTypeVector& output_dtypes,
                           const std::vector<PartialTensorShape>& output_shapes,
                           bool use_inter_op_parallelism,
                           bool preserve_cardinality) {
    string node_name = GetNodeName(MapDatasetOp::kDatasetType);
    auto map_dataset_params = new MapDatasetParams(
        dataset_params_, other_arguments, func, func_lib, type_arguments,
        output_dtypes, output_shapes, use_inter_op_parallelism,
        preserve_cardinality, node_name);
    dataset_params_.reset(map_dataset_params);
    return *this;
  }

  template <typename T>
  T GetDatasetParams() {
    auto dataset_params = dynamic_cast<T*>(dataset_params_.get());
    return *dataset_params;
  }

 private:
  string GetNodeName(const string& dataset_type) {
    string node_name = absl::StrCat(dataset_type, "_",
                                    node_name_id[RangeDatasetOp::kDatasetType]);
    node_name_id[dataset_type] += 1;
    return node_name;
  }

  std::shared_ptr<DatasetParams> dataset_params_;
  std::unordered_map<string, int64> node_name_id;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_PARAMS_H_
