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

#include <algorithm>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class InplaceUpdateOp : public XlaOpKernel {
 public:
  explicit InplaceUpdateOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(3) << "InplaceUpdateOp::Compile";

    DataType index_type = input_type(1);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // TF Args are X, I, V
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape i_shape = ctx->InputShape(1);
    const TensorShape v_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(i_shape) ||
                    TensorShapeUtils::IsVector(i_shape),
                errors::InvalidArgument("index must be Rank 0 or 1"));
    OP_REQUIRES(ctx, (x_shape.dims() == v_shape.dims()),
                errors::InvalidArgument("X and V must have the same Rank,"
                                        " X.shape=",
                                        x_shape.DebugString(),
                                        " V.shape=", v_shape.DebugString()));

    auto* builder = ctx->builder();
    auto const_zero = xla::ConstantR0(builder, 0);
    auto current = ctx->Input(0);

    for (int64_t i = 0; i < i_shape.num_elements(); i++) {
      std::vector<xla::XlaOp> update_indices;
      update_indices.push_back(
          xla::Reshape(xla::SliceInDim(ctx->Input(1), i, i + 1, 1, 0), {}));
      for (int xi = 1; xi < x_shape.dims(); xi++) {
        update_indices.push_back(const_zero);
      }
      current = xla::DynamicUpdateSlice(
          current, xla::SliceInDim(ctx->Input(2), i, i + 1, 1, 0),
          update_indices);
    }
    ctx->SetOutput(0, current);

    // TODO(b/118122460): Uncomment+format this code to use XLA Scatter.
    //     auto* builder = ctx->builder();
    //     const auto initial = ctx->Input(0);
    //     const auto indices = ctx->Input(1);
    //     const auto updates = ctx->Input(2);
    //
    //     auto result = XlaScatter(
    //         initial, updates, indices, /*indices_are_vectors=*/false,
    //         [](xla::XlaOp, xla::XlaOp second, xla::XlaBuilder*) { return
    //         second; }, builder);
    //     OP_REQUIRES_OK(ctx, result.status());
    //     ctx->SetOutput(0, result.ValueOrDie());
  }
};

REGISTER_XLA_OP(Name("InplaceUpdate"), InplaceUpdateOp);

class InplaceAddOp : public XlaOpKernel {
 public:
  explicit InplaceAddOp(OpKernelConstruction* context) : XlaOpKernel(context) {}

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(3) << "InplaceAddOp::Compile";

    DataType index_type = input_type(1);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // TF Args are X, I, V
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape i_shape = ctx->InputShape(1);
    const TensorShape v_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx,
                (TensorShapeUtils::IsScalar(i_shape) ||
                 ((i_shape.dims() == 1) && (i_shape.num_elements() == 1))),
                errors::InvalidArgument("index must be Rank 1 and size 1"));
    OP_REQUIRES(ctx, (x_shape.dims() == v_shape.dims()),
                errors::InvalidArgument("X and V must have the same Rank,"
                                        " X.shape=",
                                        x_shape.DebugString(),
                                        " V.shape=", v_shape.DebugString()));
    // Pad the indices out to the match the rank of params.
    auto* builder = ctx->builder();
    std::vector<xla::XlaOp> padded_indices;
    padded_indices.push_back(xla::Reshape(ctx->Input(1), {}));
    for (int i = 0; i < x_shape.dims() - 1; ++i) {
      padded_indices.push_back(XlaHelpers::Zero(builder, index_type));
    }

    std::vector<int64_t> sizes;
    sizes.push_back(1);
    for (int i = 1; i < x_shape.dims(); i++) {
      sizes.push_back(x_shape.dim_size(i));
    }

    auto prev = xla::DynamicSlice(ctx->Input(0), padded_indices, sizes);
    auto updated = xla::Add(prev, ctx->Input(2));
    auto result =
        xla::DynamicUpdateSlice(ctx->Input(0), updated, padded_indices);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("InplaceAdd"), InplaceAddOp);

}  // namespace
}  // namespace tensorflow
