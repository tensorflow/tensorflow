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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_ACCESS_OPS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_ACCESS_OPS_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/platform/platform.h"

namespace tensorflow {
namespace data {
namespace experimental {

// An operation that can get an element at a specified index in a dataset.
class GetElementAtIndexOp : public HybridAsyncOpKernel {
 public:
  explicit GetElementAtIndexOp(OpKernelConstruction* ctx)
      : HybridAsyncOpKernel(ctx, "tf_data_get_element_at_index") {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
  }

 protected:
  Status DoCompute(OpKernelContext* ctx) override;

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_RANDOM_ACCESS_OPS_H_
