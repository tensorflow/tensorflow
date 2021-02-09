/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_MODEL_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_MODEL_DATASET_OP_H_

#include "tensorflow/core/platform/platform.h"

// On mobile we do not provide model dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"

namespace tensorflow {
namespace data {

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kAlgorithm = "algorithm";
  static constexpr const char* const kCpuBudget = "cpu_budget";
  static constexpr const char* const kRamBudget = "ram_budget";

  explicit ModelDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;

  model::AutotuneAlgorithm algorithm_;
  int64 cpu_budget_;
  int64 ram_budget_;
};

}  // namespace data
}  // namespace tensorflow
#else  // !IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class ModelDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit ModelDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;
};

}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM

#endif  // TENSORFLOW_CORE_KERNELS_DATA_MODEL_DATASET_OP_H_
