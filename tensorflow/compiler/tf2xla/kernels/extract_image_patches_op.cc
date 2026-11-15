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
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/types.pb.h"
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
      kernel_size *= ksizes_[input_dim];
    }

    xla::PaddingConfig padding_config = xla::MakeNoPaddingConfig(num_dims);
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
        padding_config.mutable_dimensions(dim)->set_edge_padding_low(pad_low);
        padding_config.mutable_dimensions(dim)->set_edge_padding_high(pad_high);
      }
    }

    xla::XlaOp padded_input = ctx->Input(0);
    if (has_padding) {
      padded_input =
          xla::Pad(padded_input, xla::Zero(builder, type), padding_config);
    }

    std::vector<xla::XlaOp> slices;
    slices.reserve(kernel_size);

    std::vector<int64_t> start_indices(num_dims, 0);
    std::vector<int64_t> limit_indices(num_dims, 0);
    std::vector<int64_t> slice_strides(num_dims, 1);
    limit_indices[batch_dim] = input_shape.dim_size(batch_dim);
    limit_indices[feature_dim] = depth;
    for (int i = 0; i < num_spatial_dims; ++i) {
      const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      slice_strides[dim] = strides_[dim];
    }

    for (int64_t patch_idx = 0; patch_idx < kernel_size; ++patch_idx) {
      int64_t temp = patch_idx;
      for (int i = num_spatial_dims - 1; i >= 0; --i) {
        const int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
        const int64_t ksize = ksizes_[dim];
        const int64_t patch_offset = temp % ksize;
        temp /= ksize;

        if (output_sizes[i] > 0) {
          const int64_t start = patch_offset * dilations_[dim];
          start_indices[dim] = start;
          limit_indices[dim] =
              start + (output_sizes[i] - 1) * strides_[dim] + 1;
        } else {
          start_indices[dim] = 0;
          limit_indices[dim] = 0;
        }
      }

      slices.push_back(xla::Slice(padded_input, start_indices, limit_indices,
                                  slice_strides));
    }

    if (slices.size() == 1) {
      ctx->SetOutput(0, slices[0]);
    } else {
      ctx->SetOutput(0, xla::ConcatInDim(builder, slices, feature_dim));
    }
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
