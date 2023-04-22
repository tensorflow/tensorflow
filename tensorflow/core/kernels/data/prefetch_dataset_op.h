/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/kernels/data/prefetch_autotuner.h"

namespace tensorflow {
namespace data {

class PrefetchDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Prefetch";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kBufferSize = "buffer_size";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kSlackPeriod = "slack_period";
  static constexpr const char* const kLegacyAutotune = "legacy_autotune";
  static constexpr const char* const kBufferSizeMin = "buffer_size_min";

  explicit PrefetchDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  int64 slack_period_ = 0;
  bool legacy_autotune_ = true;
  int64 buffer_size_min_ = 0;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PREFETCH_DATASET_OP_H_
