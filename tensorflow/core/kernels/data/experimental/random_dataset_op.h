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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {

// See tensorflow/core/api_def/base_api/api_def_RandomDataset.pbtxt for the
// API definition that corresponds to this kernel.
class RandomDatasetOp : public DatasetOpKernel {
 public:
  // Names of op parameters, public so that they can be accessed by test cases.
  // Make sure that these are kept in sync with the REGISTER_OP call in
  // tensorflow/core/ops/experimental_dataset_ops.cc
  static constexpr const char* const kDatasetType = "Random";
  static constexpr const char* const kSeed = "seed";
  static constexpr const char* const kSeed2 = "seed2";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit RandomDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_DATASET_OP_H_
