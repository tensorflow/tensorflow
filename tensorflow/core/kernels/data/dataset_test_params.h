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

class DatasetParams : public core::RefCounted {
 public:
  DatasetParams(DataTypeVector output_dtypes,
                std::vector<PartialTensorShape> output_shapes, string node_name,
                DatasetParamsType type);

  ~DatasetParams() {}

  // Returns the dataset inputs as a TensorValue vector.
  virtual Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) = 0;

  // Checks if the tensor is a dataset variant tensor.
  static bool IsDatasetTensor(const Tensor& tensor);

  // Creates a BatchDatasetParams. The ownership is transferred to the caller.
  BatchDatasetParams* Batch(int64 batch_size, bool drop_remainder,
                            bool parallel_copy, DataTypeVector output_dtypes,
                            std::vector<PartialTensorShape> output_shapes,
                            string node_name);
  // Creates a MapDatasetParams. The ownership is transferred to the caller.
  MapDatasetParams* Map(std::vector<Tensor> other_arguments,
                        FunctionDefHelper::AttrValueWrapper func,
                        std::vector<FunctionDef> func_lib,
                        DataTypeVector type_arguments,
                        DataTypeVector output_dtypes,
                        std::vector<PartialTensorShape> output_shapes,
                        bool use_inter_op_parallelism,
                        bool preserve_cardinality, string node_name);

  DataTypeVector output_dtypes;
  std::vector<PartialTensorShape> output_shapes;
  string node_name;
  string iterator_prefix = kDefaultIteratorPrefix;
  DatasetParamsType type;
};

class SourceDatasetParams : public DatasetParams {
 public:
  SourceDatasetParams(DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name, DatasetParamsType type);

  ~SourceDatasetParams() {}
};

class UnaryDatasetParams : public DatasetParams {
 public:
  UnaryDatasetParams(DatasetParams* input_dataset_params,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name, DatasetParamsType type);

  ~UnaryDatasetParams() {
    if (input_dataset_params) {
      input_dataset_params->Unref();
    }
  }

  DatasetParams* input_dataset_params = nullptr;  // owned
  Tensor input_dataset;
};

class BinaryDatasetParams : public DatasetParams {
 public:
  BinaryDatasetParams(DatasetParams* input_dataset_params_0,
                      DatasetParams* input_dataset_params_1,
                      DataTypeVector output_dtypes,
                      std::vector<PartialTensorShape> output_shapes,
                      string node_name, DatasetParamsType type);

  ~BinaryDatasetParams() override {
    if (input_dataset_params_0) {
      input_dataset_params_0->Unref();
    }
    if (input_dataset_params_1) {
      input_dataset_params_1->Unref();
    }
  }

  DatasetParams* input_dataset_params_0 = nullptr;  // owned
  DatasetParams* input_dataset_params_1 = nullptr;  // owned
  Tensor input_dataset_0;
  Tensor input_dataset_1;
};

class RangeDatasetParams : public SourceDatasetParams {
 public:
  RangeDatasetParams(int64 start, int64 stop, int64 step,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name);

  RangeDatasetParams(int64 start, int64 stop, int64 step);

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Tensor start;
  Tensor stop;
  Tensor step;
};

class BatchDatasetParams : public UnaryDatasetParams {
 public:
  BatchDatasetParams(DatasetParams* input_dataset_params, int64 batch_size,
                     bool drop_remainder, bool parallel_copy,
                     DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name);

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  Tensor batch_size;
  Tensor drop_remainder;
  bool parallel_copy;
};

class MapDatasetParams : public UnaryDatasetParams {
 public:
  MapDatasetParams(DatasetParams* input_dataset_params,
                   std::vector<Tensor> other_arguments,
                   FunctionDefHelper::AttrValueWrapper func,
                   std::vector<FunctionDef> func_lib,
                   DataTypeVector type_arguments, DataTypeVector output_dtypes,
                   std::vector<PartialTensorShape> output_shapes,
                   bool use_inter_op_parallelism, bool preserve_cardinality,
                   string node_name);

  Status MakeInputs(gtl::InlinedVector<TensorValue, 4>* inputs) override;

  std::vector<Tensor> other_arguments;
  FunctionDefHelper::AttrValueWrapper func;
  std::vector<FunctionDef> func_lib;
  DataTypeVector type_arguments;
  bool use_inter_op_parallelism;
  bool preserve_cardinality;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_PARAMS_H_
