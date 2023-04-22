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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_REDUCE_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_REDUCE_DATASET_OP_H_

#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"

namespace tensorflow {
namespace data {

class ReduceDatasetOp : public HybridAsyncOpKernel {
 public:
  explicit ReduceDatasetOp(OpKernelConstruction* ctx);

 protected:
  Status DoCompute(OpKernelContext* ctx) override;

  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_REDUCE_DATASET_OP_H_
