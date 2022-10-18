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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_THREADPOOL_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_THREADPOOL_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/platform.h"

namespace tensorflow {
namespace data {
namespace experimental {

class MaxIntraOpParallelismDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType =
      "MaxIntraOpParallelismDataset";
  static constexpr const char* const kDatasetOp =
      "MaxIntraOpParallelismDatasetOp";

  // Executes the logic of the MaxIntraOpParallelismDatasetOp directly (as
  // opposed to through executing the MaxIntraOpParallelismDatasetOp op kernel).
  static void MakeDatasetFromOptions(OpKernelContext* ctx, DatasetBase* input,
                                     int32_t max_intra_op_parallelism,
                                     DatasetBase** output);

  explicit MaxIntraOpParallelismDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
};

class PrivateThreadPoolDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "PrivateThreadPoolDataset";
  static constexpr const char* const kDatasetOp = "PrivateThreadPoolDatasetOp";

  // Executes the logic of the PrivateThreadpoolDatasetOp directly (as
  // opposed to through executing the PrivateThreadpoolDatasetOp op kernel).
  static void MakeDatasetFromOptions(OpKernelContext* ctx, DatasetBase* input,
                                     int32_t num_threads, DatasetBase** output);

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

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_THREADPOOL_DATASET_OP_H_
