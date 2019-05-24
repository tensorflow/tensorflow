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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_FILTER_BY_COMPONENT_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_FILTER_BY_COMPONENT_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

// TODO(prazek): Filter already has a logic of filtering by the given tensor,
// but it must return both components.  We could introduce kernel like
// DropComponentDatasetOp and use FilterDataset for filtering.
class FilterByLastComponentDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char kDatasetType[] = "FilterByLastComponent";
  static constexpr const char kInputDataset[] = "input_dataset";
  static constexpr const char kOutputTypes[] = "output_types";
  static constexpr const char kOutputShapes[] = "output_shapes";

  explicit FilterByLastComponentDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx),
        graph_def_version_(ctx->graph_def_version()) {
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
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_FILTER_BY_COMPONENT_DATASET_OP_H_
