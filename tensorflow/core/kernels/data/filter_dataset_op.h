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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_FILTER_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_FILTER_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/captured_function.h"

namespace tensorflow {
namespace data {

class FilterDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char kDatasetType[] = "Filter";
  static constexpr const char kInputDataset[] = "input_dataset";
  static constexpr const char kOtherArguments[] = "other_arguments";
  static constexpr const char kPredicate[] = "predicate";
  static constexpr const char kTarguments[] = "Targuments";
  static constexpr const char kOutputTypes[] = "output_types";
  static constexpr const char kOutputShapes[] = "output_shapes";

  explicit FilterDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kPredicate, /*params=*/{},
                                                 &func_metadata_));
    OP_REQUIRES(ctx, func_metadata_->short_circuit_info().indices.size() <= 1,
                errors::InvalidArgument(
                    "predicate function has more than one return value."));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_FILTER_DATASET_OP_H_
