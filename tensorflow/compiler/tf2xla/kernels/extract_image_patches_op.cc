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

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/slicing.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

class ExtractImagePatchesOp : public XlaOpKernel {
 public:
  explicit ExtractImagePatchesOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ksizes", &ksizes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rates", &dilations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorFormat data_format = FORMAT_NHWC;
    const int num_dims = ksizes_.size();

    OP_REQUIRES(ctx, num_dims >= 3,
                absl::InvalidArgumentError(
                    "Kernel size must have at least 3 dimensions"));
    const int num_spatial_dims = num_dims - 2;

    OP_REQUIRES(ctx, strides_.size() == num_dims,
                absl::InvalidArgumentError(
                    absl::StrCat("Sliding window strides field must "
                                 "specify ",
                                 num_dims, " dimensions")));
    OP_REQUIRES(
        ctx, dilations_.size() == num_dims,
        absl::InvalidArgumentError(absl::StrCat("Dilations field must "
                                                "specify ",
                                                num_dims, " dimensions")));

    int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
    int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
    OP_REQUIRES(
        ctx, ksizes_[batch_dim] == 1 && ksizes_[feature_dim] == 1,
        absl::UnimplementedError("Current implementation does not yet support "
                                 "kernel sizes > 1 in the batch and depth "
                                 "dimensions."));
    OP_REQUIRES(
        ctx, strides_[batch_dim] == 1 && strides_[feature_dim] == 1,
        absl::UnimplementedError("Current implementation does not yet support "
                                 "strides in the batch and depth dimensions."));
    OP_REQUIRES(ctx, dilations_[batch_dim] == 1 && dilations_[feature_dim] == 1,
                absl::UnimplementedError(
                    "Current implementation does not support "
                    "dilations in the batch and depth dimensions."));

    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      OP_REQUIRES(
          ctx, ksizes_[input_dim] >= 1,
          absl::OutOfRangeError(absl::StrCat(
              "Kernel size values must be positive; ", i,
              "th spatial dimension had kernel size ", ksizes_[input_dim])));
      OP_REQUIRES(
          ctx, strides_[input_dim] >= 1,
          absl::UnimplementedError(absl::StrCat(
              "Stride values must be positive; ", i,
              "th spatial dimension had dilation ", dilations_[input_dim])));
      OP_REQUIRES(
          ctx, dilations_[input_dim] >= 1,
          absl::UnimplementedError(absl::StrCat(
              "Dilation values must be positive; ", i,
              "th spatial dimension had dilation ", dilations_[input_dim])));
    }

    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(ctx->input_type(0), &type));

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, input_shape.dims() == num_dims,
                absl::InvalidArgumentError(
                    absl::StrCat("input must be ", num_dims, "-dimensional",
                                 input_shape.DebugString())));
    const int64_t depth = input_shape.dim_size(feature_dim);

    xla::XlaBuilder* builder = ctx->builder();

    int64_t kernel_size = 1;
    for (int i = 0; i < num_spatial_dims; ++i) {
      int input_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      kernel_size = MultiplyWithoutOverflow(kernel_size, ksizes_[input_dim]);
    }
    OP_REQUIRES(ctx, kernel_size >= 0,
                absl::InvalidArgumentError("Kernel size is too large"));

    std::vector<std::pair<int64_t, int64_t>> padding(num_dims, {0, 0});
    bool has_padding = false;
    std::vector<int64_t> output_sizes(num_spatial_dims);

    for (int i = 0; i < num_spatial_dims; ++i) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      int64_t pad_low = 0;
      int64_t pad_high = 0;
      OP_REQUIRES_OK(ctx, GetWindowedOutputSizeVerbose(
                              input_shape.dim_size(dim), ksizes_[dim],
                              dilations_[dim], strides_[dim], padding_,
                              &output_sizes[i], &pad_low, &pad_high));
      if (pad_low != 0 || pad_high != 0) {
        has_padding = true;
        padding[dim] = {pad_low, pad_high};
      }
    }
    std::vector<int64_t> padded_dims;
    padded_dims.reserve(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      padded_dims.push_back(AddWithoutOverflow(
          input_shape.dim_size(i),
          AddWithoutOverflow(padding[i].first, padding[i].second)));
    }

    std::vector<int64_t> output_dims(num_dims);
    output_dims[batch_dim] = input_shape.dim_size(batch_dim);
    output_dims[feature_dim] = MultiplyWithoutOverflow(kernel_size, depth);
    bool empty_output = output_dims[batch_dim] == 0 || depth == 0;
    for (int i = 0; i < num_spatial_dims; ++i) {
      output_dims[GetTensorSpatialDimIndex(num_dims, data_format, i)] =
          output_sizes[i];
      empty_output |= output_sizes[i] == 0;
    }
    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShape::BuildTensorShape(output_dims, &output_shape));
    // XLA shape inference aborts on a shape whose byte size overflows, so
    // validate each shape before the builder sees it.
    OP_REQUIRES_OK(
        ctx, xla::ShapeUtil::MakeValidatedShape(type, output_dims).status());
    if (empty_output) {
      ctx->SetOutput(0, xla::Broadcast(xla::Zero(builder, type), output_dims));
      return;
    }

    xla::XlaOp patches = ctx->Input(0);
    if (has_padding) {
      OP_REQUIRES_OK(
          ctx, xla::ShapeUtil::MakeValidatedShape(type, padded_dims).status());
      patches = xla::Pad(patches, xla::Zero(builder, type),
                         xla::MakeEdgePaddingConfig(padding));
    }

    // One slice per kernel position makes the graph grow with the kernel size:
    // a 40x320 kernel compiles to 12800 slices and a concatenation. An identity
    // convolution has a fixed size, but it spreads a NaN to other patch
    // elements. A gather for each spatial dimension has neither problem.
    //
    // The loop goes from the last spatial dimension to the first. Each step
    // splits one axis into an (output, kernel offset) pair, and the axes still
    // to be gathered then keep their positions.
    std::vector<int64_t> dims = padded_dims;
    for (int i = num_spatial_dims - 1; i >= 0; --i) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      const int64_t ksize = ksizes_[dim];
      OP_REQUIRES_VALUE(const xla::Shape index_shape, ctx,
                        xla::ShapeUtil::MakeValidatedShape(
                            xla::S64, {output_sizes[i], ksize}));
      const xla::XlaOp index = xla::Add(
          xla::Mul(xla::Iota(builder, index_shape, /*iota_dimension=*/0),
                   xla::ConstantR0<int64_t>(builder, strides_[dim])),
          xla::Mul(xla::Iota(builder, index_shape, /*iota_dimension=*/1),
                   xla::ConstantR0<int64_t>(builder, dilations_[dim])));
      dims.erase(dims.begin() + dim);
      dims.insert(dims.begin() + dim, {output_sizes[i], ksize});
      OP_REQUIRES_OK(ctx,
                     xla::ShapeUtil::MakeValidatedShape(type, dims).status());
      patches = xla::TorchIndexSelect(
          patches,
          xla::Reshape(index, {xla::ShapeUtil::ElementsIn(index_shape)}), dim);
      patches = xla::Reshape(patches, dims);
    }

    // The axes are now [batch, out_0, k_0, ..., out_n, k_n, depth]. Move the
    // kernel axes next to depth to get the (k_0, ..., k_n, depth) patch order.
    std::vector<int64_t> permutation;
    permutation.reserve(2 + 2 * num_spatial_dims);
    permutation.push_back(0);
    for (int i = 0; i < num_spatial_dims; ++i) {
      permutation.push_back(1 + 2 * i);
    }
    for (int i = 0; i < num_spatial_dims; ++i) {
      permutation.push_back(2 + 2 * i);
    }
    permutation.push_back(1 + 2 * num_spatial_dims);
    patches = xla::Transpose(patches, permutation);
    ctx->SetOutput(0, xla::Reshape(patches, output_dims));
  }

 protected:
  std::vector<int32_t> ksizes_;
  std::vector<int32_t> dilations_;
  std::vector<int32_t> strides_;
  Padding padding_;

 private:
  ExtractImagePatchesOp(const ExtractImagePatchesOp&) = delete;
  void operator=(const ExtractImagePatchesOp&) = delete;
};

REGISTER_XLA_OP(Name("ExtractImagePatches"), ExtractImagePatchesOp);

}  // namespace
}  // namespace tensorflow
