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

#include "tensorflow/core/kernels/data/experimental/compression_ops.h"

#include "tensorflow/core/data/compression_utils.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace experimental {

CompressElementOp::CompressElementOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {}

void CompressElementOp::Compute(OpKernelContext* ctx) {
  std::vector<Tensor> components;
  for (size_t i = 0; i < ctx->num_inputs(); ++i) {
    components.push_back(ctx->input(i));
  }
  CompressedElement compressed;
  OP_REQUIRES_OK(ctx, CompressElement(components, &compressed));

  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
  output->scalar<Variant>()() = std::move(compressed);
}

UncompressElementOp::UncompressElementOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void UncompressElementOp::Compute(OpKernelContext* ctx) {
  Tensor tensor = ctx->input(0);
  const Variant& variant = tensor.scalar<Variant>()();
  const CompressedElement* compressed = variant.get<CompressedElement>();

  std::vector<Tensor> components;
  OP_REQUIRES_OK(ctx, UncompressElement(*compressed, &components));
  OP_REQUIRES(ctx, components.size() == output_types_.size(),
              errors::FailedPrecondition("Expected ", output_types_.size(),
                                         " outputs from uncompress, but got ",
                                         components.size()));
  for (int i = 0; i < components.size(); ++i) {
    OP_REQUIRES(
        ctx, components[i].dtype() == output_types_[i],
        errors::FailedPrecondition("Expected a tensor of type ",
                                   DataTypeString(output_types_[i]),
                                   " but got a tensor of type ",
                                   DataTypeString(components[i].dtype())));
    ctx->set_output(i, components[i]);
  }
}

REGISTER_KERNEL_BUILDER(Name("CompressElement").Device(DEVICE_CPU),
                        CompressElementOp);
REGISTER_KERNEL_BUILDER(Name("UncompressElement").Device(DEVICE_CPU),
                        UncompressElementOp);

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
