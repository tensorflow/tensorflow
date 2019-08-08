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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_FLAT_MAP_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_FLAT_MAP_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/captured_function.h"

namespace tensorflow {
namespace data {

class FlatMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char kDatasetType[] = "FlatMap";
  static constexpr const char kInputDataset[] = "input_dataset";
  static constexpr const char kOtherArguments[] = "other_arguments";
  static constexpr const char kF[] = "f";
  static constexpr const char kTarguments[] = "Targuments";
  static constexpr const char kOutputTypes[] = "output_types";
  static constexpr const char kOutputShapes[] = "output_shapes";

  explicit FlatMapDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(
        ctx, FunctionMetadata::Create(ctx, kF, /*params=*/{}, &func_metadata_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  const int graph_def_version_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_FLAT_MAP_DATASET_OP_H_
