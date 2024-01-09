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

namespace tensorflow {
namespace data {

class TakeDataset : public DatasetBase {
 public:
  TakeDataset(OpKernelContext* ctx, int64_t count, const DatasetBase* input);

  TakeDataset(DatasetContext::Params params, int64_t count,
              const DatasetBase* input);

  ~TakeDataset() override;

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override;

  const DataTypeVector& output_dtypes() const override;

  const std::vector<PartialTensorShape>& output_shapes() const override;

  string DebugString() const override;

  int64_t CardinalityInternal(CardinalityOptions options) const override;

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override;

  Status Get(OpKernelContext* ctx, int64 index,
             std::vector<Tensor>* out_tensors) const override;

  Status CheckExternalState() const override;

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override;

 private:
  class EmptyIterator;
  class FiniteIterator;
  const int64_t count_;
  const DatasetBase* const input_;
};

class TakeDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Take";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kCount = "count";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit TakeDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_TAKE_DATASET_OP_H_
