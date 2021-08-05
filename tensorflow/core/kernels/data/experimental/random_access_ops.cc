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
#include "tensorflow/core/kernels/data/experimental/random_access_ops.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace tensorflow {
namespace data {
namespace experimental {

Status GetElementAtIndexOp::DoCompute(OpKernelContext* ctx) {
  DatasetBase* dataset;
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(ctx->input(0), &dataset));

  int64 index = 0;
  TF_RETURN_IF_ERROR(ParseScalarArgument<int64>(ctx, "index", &index));

  std::vector<Tensor> components;
  TF_RETURN_IF_ERROR(dataset->Get(ctx, index, &components));
  TF_RETURN_IF_ERROR(VerifyTypesMatch(output_types_, components));
  TF_RETURN_IF_ERROR(VerifyShapesCompatible(output_shapes_, components));

  for (int i = 0; i < components.size(); ++i) {
    ctx->set_output(i, components[i]);
  }
  return Status::OK();
}

REGISTER_KERNEL_BUILDER(Name("GetElementAtIndex").Device(DEVICE_CPU),
                        GetElementAtIndexOp);

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
