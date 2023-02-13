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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_LOAD_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_LOAD_DATASET_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace data {
namespace experimental {

// An operation that can load a dataset from one or more files.
class LoadDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kCompression = "compression";
  static constexpr const char* const kDatasetType = "Load";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kPath = "path";
  static constexpr const char* const kReaderFunc = "reader_func";
  static constexpr const char* const kReaderFuncOtherArgs =
      "reader_func_other_args";
  static constexpr const char* const kReaderFuncTarguments =
      "Treader_func_args";

  explicit LoadDatasetOp(OpKernelConstruction* ctx);

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  // Dataset classes for different formats. V1 loads the output of a
  // `SaveDataset()`. V2 loads the output of a `DistributedSaveDataset()`.
  class DatasetV1;
  class DatasetV2;

  std::string compression_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::shared_ptr<FunctionMetadata> func_metadata_;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_LOAD_DATASET_OP_H_
