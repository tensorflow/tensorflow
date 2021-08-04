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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_BATCH_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_BATCH_DATASET_OP_H_

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class ParallelBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ParallelBatch";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kBatchSize = "batch_size";
  static constexpr const char* const kNumParallelCalls = "num_parallel_calls";
  static constexpr const char* const kDropRemainder = "drop_remainder";
  static constexpr const char* const kParallelCopy = "parallel_copy";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kDeterministic = "deterministic";

  explicit ParallelBatchDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  DeterminismPolicy deterministic_;
  bool parallel_copy_ = false;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_BATCH_DATASET_OP_H_
