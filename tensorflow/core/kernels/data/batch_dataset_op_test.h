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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_BATCH_DATASET_OP_TEST_H
#define TENSORFLOW_CORE_KERNELS_DATA_BATCH_DATASET_OP_TEST_H

#include "tensorflow/core/kernels/data/batch_dataset_op.h"
#include "tensorflow/core/kernels/data/dataset_test_params.h"

namespace tensorflow {
namespace data {

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
    inputs->emplace_back(TensorValue(&batch_size_));
    inputs->emplace_back(TensorValue(&drop_remainder_));
    return Status::OK();
  }

  Status MakeInputPlaceholder(
      std::vector<string>* input_placeholder) const override {
    *input_placeholder = {BatchDatasetOp::kInputDataset,
                          BatchDatasetOp::kBatchSize,
                          BatchDatasetOp::kDropRemainder};
    return Status::OK();
  }

  Status MakeAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {{BatchDatasetOp::kParallelCopy, parallel_copy_},
                    {BatchDatasetOp::kOutputTypes, output_dtypes_},
                    {BatchDatasetOp::kOutputShapes, output_shapes_}};
    return Status::OK();
  }

  int op_version() const override { return op_version_; }

 private:
  Tensor batch_size_;
  Tensor drop_remainder_;
  bool parallel_copy_;
  int op_version_ = 2;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_BATCH_DATASET_OP_TEST_H
