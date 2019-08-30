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

// The return string needs to be same with `kDatasetType`.
string ToString(DatasetParamsType type) {
  switch (type) {
    case DatasetParamsType::Range:
      return "Range";
    case DatasetParamsType::Batch:
      return "Batch";
    case DatasetParamsType::Map:
      return "Map";
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

}  // namespace data
}  // namespace tensorflow
