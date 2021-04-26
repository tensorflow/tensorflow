/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_THREADPOOL_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_THREADPOOL_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/platform.h"

namespace tensorflow {
namespace data {
namespace experimental {

// TODO(jsimsa): Provide class-level documentation for this and the other ops.
class MaxIntraOpParallelismDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType =
      "MaxIntraOpParallelismDataset";
  static constexpr const char* const kDatasetOp =
      "MaxIntraOpParallelismDatasetOp";

  // Creates and returns a MaxIntraOpParallelismDatasetOp::Dataset in output,
  // given the input dataset, and max_intra_op_parallelism parameters. This
  // method is used to create the dataset without explicitly using the
  // MaxIntraOpParallelismDatasetOp.
  static void MakeDatasetFromOptions(OpKernelContext* ctx, DatasetBase* input,
                                     int32 max_intra_op_parallelism,
                                     DatasetBase** output);

  explicit MaxIntraOpParallelismDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
};

// TODO(jsimsa): Provide class-level documentation for this and the other ops.
class PrivateThreadPoolDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "PrivateThreadPoolDataset";
  static constexpr const char* const kDatasetOp = "PrivateThreadPoolDatasetOp";

  // Creates and returns a PrivateThreadPoolDatasetOp::Dataset in output, given
  // the input and number of threads. This method is used to create the dataset
  // without explicitly using the PrivateThreadPoolDatasetOp.
  static void MakeDatasetFromOptions(OpKernelContext* ctx, DatasetBase* input,
                                     int32 num_threads, DatasetBase** output);

  explicit PrivateThreadPoolDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_THREADPOOL_DATASET_OP_H_
