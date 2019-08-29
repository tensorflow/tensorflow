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

namespace tensorflow {
namespace data {

typedef std::vector<
    std::pair<string, tensorflow::FunctionDefHelper::AttrValueWrapper>>
    AttributeVector;

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

  // Returns the dataset attributes as a vector.
  virtual Status MakeAttributes(AttributeVector* attributes) const = 0;

  // Checks if the tensor is a dataset variant tensor.
  static bool IsDatasetTensor(const Tensor& tensor);

  string node_name() const { return node_name_; }

  DataTypeVector output_dtypes() const { return output_dtypes_; }

  std::vector<PartialTensorShape> output_shapes() const {
    return output_shapes_;
  }

  string iterator_prefix() const { return iterator_prefix_; }

  DatasetParamsType type() const { return type_; }

  std::vector<std::pair<std::shared_ptr<DatasetParams>, Tensor>>&
  input_dataset_params() {
    return input_dataset_params_group_;
  }

  virtual std::vector<FunctionDef> func_lib() { return {}; }

 protected:
  // Used to store all the input dataset parameters and the dataset tensors
  // generated from the parameters.
  std::vector<std::pair<std::shared_ptr<DatasetParams>, Tensor>>
      input_dataset_params_group_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
  string node_name_;
  string iterator_prefix_ = kDefaultIteratorPrefix;
  DatasetParamsType type_;
};

class BatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  BatchDatasetParams(T input_dataset_params, int64 batch_size,
                     bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name), DatasetParamsType::Batch),
        batch_size_(CreateTensor<int64>(TensorShape({}), {batch_size})),
        drop_remainder_(CreateTensor<bool>(TensorShape({}), {drop_remainder})),
        parallel_copy_(parallel_copy) {
    auto input_dataset_params_ptr =
        std::make_shared<T>(std::move(input_dataset_params));
    input_dataset_params_group_.emplace_back(
        std::make_pair(std::move(input_dataset_params_ptr), Tensor()));
  }

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Status MakeAttributes(AttributeVector* attr_vector) const override;

 private:
  Tensor batch_size_;
  Tensor drop_remainder_;
  bool parallel_copy_;
};

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

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Status MakeAttributes(AttributeVector* attr_vector) const override;

  std::vector<FunctionDef> func_lib() override;

  int num_of_other_arguments() const;

 private:
  std::vector<Tensor> other_arguments_;
  FunctionDefHelper::AttrValueWrapper func_;
  std::vector<FunctionDef> func_lib_;
  DataTypeVector type_arguments_;
  bool use_inter_op_parallelism_;
  bool preserve_cardinality_;
};

class RangeDatasetParams : public DatasetParams {
 public:
  RangeDatasetParams(int64 start, int64 stop, int64 step,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name);

  RangeDatasetParams(int64 start, int64 stop, int64 step);

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Status MakeAttributes(AttributeVector* attr_vector) const override;

 private:
  Tensor start_;
  Tensor stop_;
  Tensor step_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_PARAMS_H_
