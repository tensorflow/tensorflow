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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_GENERATOR_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_GENERATOR_DATASET_OP_H_

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

class GeneratorDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Generator";
  static constexpr const char* const kInitFuncOtherArgs =
      "init_func_other_args";
  static constexpr const char* const kNextFuncOtherArgs =
      "next_func_other_args";
  static constexpr const char* const kFinalizeFuncOtherArgs =
      "finalize_func_other_args";
  static constexpr const char* const kInitFunc = "init_func";
  static constexpr const char* const kNextFunc = "next_func";
  static constexpr const char* const kFinalizeFunc = "finalize_func";
  static constexpr const char* const kTinitFuncArgs = "Tinit_func_args";
  static constexpr const char* const kTnextFuncArgs = "Tnext_func_args";
  static constexpr const char* const kTfinalizeFuncArgs = "Tfinalize_func_args";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit GeneratorDatasetOp(OpKernelConstruction* ctx);

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;

  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::shared_ptr<FunctionMetadata> init_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> next_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> finalize_func_metadata_ = nullptr;
};

}  // namespace data
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_DATA_GENERATOR_DATASET_OP_H_
