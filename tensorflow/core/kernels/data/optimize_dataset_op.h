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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_OPTIMIZE_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_OPTIMIZE_DATASET_OP_H_

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/platform/platform.h"

// On mobile we do not provide optimize dataset op because not all of its
// dependencies are available there. The op is replaced with a no-op.
#if !defined(IS_MOBILE_PLATFORM)
namespace tensorflow {
namespace data {

class OptimizeDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Optimize";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kOptimizations = "optimizations";
  static constexpr const char* const kOptimizationsEnabled =
      "optimizations_enabled";
  static constexpr const char* const kOptimizationsDisabled =
      "optimizations_disabled";
  static constexpr const char* const kOptimizationsDefault =
      "optimizations_default";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kOptimizationConfigs =
      "optimization_configs";
  static constexpr const char* const kOptimizeDatasetV1 = "OptimizeDataset";
  static constexpr const char* const kOptimizeDatasetV2 = "OptimizeDatasetV2";

  // Creates and returns a OptimizeDatasetOp::Dataset in output, given the
  // default optimizations and those that are enabled, disabled. This method is
  // used to create the dataset without explicitly using the OptimizeDatasetOp.
  static void MakeDatasetFromOptions(
      OpKernelContext* ctx, DatasetBase* input,
      const absl::flat_hash_set<tstring>& optimizations_enabled,
      const absl::flat_hash_set<tstring>& optimizations_disabled,
      const absl::flat_hash_set<tstring>& optimizations_default,
      const absl::flat_hash_set<tstring>& optimization_configs,
      DatasetBase** output);

  explicit OptimizeDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  absl::flat_hash_set<tstring> optimization_configs_;
  int op_version_ = 0;
};

}  // namespace data
}  // namespace tensorflow
#else  // !IS_MOBILE_PLATFORM
namespace tensorflow {
namespace data {

class OptimizeDatasetOp : public UnaryDatasetOpKernel {
 public:
  // Executes the logic of the OptimizeDatasetOp directly (as opposed to through
  // executing the OptimizeDatasetOp op kernel).
  static void MakeDatasetFromOptions(
      OpKernelContext* ctx, DatasetBase* input,
      const absl::flat_hash_set<tstring>& optimizations_enabled,
      const absl::flat_hash_set<tstring>& optimizations_disabled,
      const absl::flat_hash_set<tstring>& optimizations_default,
      const absl::flat_hash_set<tstring>& optimization_configs,
      DatasetBase** output);

  explicit OptimizeDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;
};

}  // namespace data
}  // namespace tensorflow
#endif  // !IS_MOBILE_PLATFORM

#endif  // TENSORFLOW_CORE_KERNELS_DATA_OPTIMIZE_DATASET_OP_H_
