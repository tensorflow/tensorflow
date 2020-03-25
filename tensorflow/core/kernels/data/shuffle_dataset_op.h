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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_SHUFFLE_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_SHUFFLE_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class ShuffleDatasetOpBase : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kBufferSize = "buffer_size";
  static constexpr const char* const kSeed = "seed";
  static constexpr const char* const kSeed2 = "seed2";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit ShuffleDatasetOpBase(OpKernelConstruction* ctx);

 protected:
  class ShuffleDatasetBase;
};

class ShuffleDatasetOp : public ShuffleDatasetOpBase {
 public:
  static constexpr const char* const kDatasetType = "Shuffle";
  static constexpr const char* const kReshuffleEachIteration =
      "reshuffle_each_iteration";

  explicit ShuffleDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  class DatasetV2;
  class FixedSeedDataset;
  int op_version_;
  bool reshuffle_each_iteration_;
};

class ShuffleAndRepeatDatasetOp : public ShuffleDatasetOpBase {
 public:
  static constexpr const char* const kDatasetType = "ShuffleAndRepeat";
  static constexpr const char* const kCount = "count";

  explicit ShuffleAndRepeatDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_SHUFFLE_DATASET_OP_H_
