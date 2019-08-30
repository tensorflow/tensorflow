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
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"

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

class DatasetParams {
 public:
  DatasetParams(DataTypeVector output_dtypes,
                std::vector<PartialTensorShape> output_shapes, string node_name,
                DatasetParamsType type);

  ~DatasetParams() {}

  // Returns the dataset inputs as a TensorValue vector.
  virtual Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) = 0;

  virtual Status MakeInputPlaceholder(
      std::vector<string>* input_placeholder) const = 0;

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

  virtual std::vector<FunctionDef> func_lib() const { return {}; }

  virtual int op_version() const { return op_version_; }

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
  int op_version_ = 1;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_PARAMS_H_
