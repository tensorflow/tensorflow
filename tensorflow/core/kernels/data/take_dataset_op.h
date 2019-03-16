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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_TAKE_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_TAKE_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace data {

class TakeDataset : public DatasetBase {
 public:
  TakeDataset(OpKernelContext* ctx, int64 count, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)), count_(count), input_(input) {
    input_->Ref();
  }

  TakeDataset(DatasetContext::Params params, int64 count,
              const DatasetBase* input)
      : DatasetBase(DatasetContext(std::move(params))),
        count_(count),
        input_(input) {
    input_->Ref();
  }

  ~TakeDataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override { return "TakeDatasetOp::Dataset"; }

  int64 Cardinality() const override {
    int64 n = input_->Cardinality();
    if (n == kUnknownCardinality) {
      return kUnknownCardinality;
    }
    if (n == kInfiniteCardinality) {
      return count_;
    }
    return std::min(n, count_);
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  class EmptyIterator;
  class FiniteIterator;
  const int64 count_;
  const DatasetBase* const input_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_TAKE_DATASET_OP_H_
