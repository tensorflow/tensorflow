/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/data_format.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class DepthToSpaceOp : public XlaOpKernel {
 public:
  explicit DepthToSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
    OP_REQUIRES(
        ctx, block_size_ > 1,
        errors::InvalidArgument("Block size should be > 1: ", block_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);

    TensorFormat data_format = data_format_;
    // If the data is in a vectorized format, reformat it into a non-vectorized
    // version first. We'll undo the transformation later.
    if (data_format == FORMAT_NCHW_VECT_C) {
      data_format = FORMAT_NCHW;
      auto input_reshaped = NCHW_VECT_CToNCHW(input);
      OP_REQUIRES_OK(ctx, input_reshaped.status());
      input = input_reshaped.value();
    }

    OP_REQUIRES(ctx, data_format == FORMAT_NCHW || data_format == FORMAT_NHWC,
                errors::InvalidArgument("Unsupported data format ",
                                        ToString(data_format_)));

    xla::XlaBuilder* builder = input.builder();
    auto input_xla_shape = builder->GetShape(input);
    OP_REQUIRES_OK(ctx, input_xla_shape.status());
    absl::Span<const int64_t> input_shape =
        input_xla_shape.value().dimensions();
    int input_rank = input_shape.size();

    static const int kRequiredDims = 4;
    OP_REQUIRES(ctx, kRequiredDims == input_rank,
                errors::InvalidArgument("Input rank should be ", kRequiredDims,
                                        "; got: ", input_rank));

    int feature_dim = GetTensorFeatureDimIndex(input_rank, data_format);
    int num_spatial_dims = GetTensorSpatialDims(input_rank, data_format);

    std::vector<int64_t> reshaped_shape;
    std::vector<int64_t> transpose_order;
    std::vector<int64_t> output_shape;
    reshaped_shape.reserve(input_rank);
    transpose_order.reserve(input_rank);
    output_shape.reserve(input_rank);
    if (data_format == FORMAT_NHWC) {
      reshaped_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(input_shape[1 + i]);
      }
      int64_t block_elems = 1;
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(block_size_);
        block_elems *= block_size_;
      }
      reshaped_shape.push_back(input_shape[feature_dim] / block_elems);

      transpose_order.push_back(0);
      for (int i = 0; i < num_spatial_dims; ++i) {
        transpose_order.push_back(i + 1);
        transpose_order.push_back(i + 1 + num_spatial_dims);
      }
      transpose_order.push_back(feature_dim + num_spatial_dims);

      output_shape.push_back(input_shape[0]);
      for (int i = 0; i < num_spatial_dims; ++i) {
        output_shape.push_back(input_shape[1 + i] * block_size_);
      }
      output_shape.push_back(input_shape[feature_dim] / block_elems);
    } else {
      // NCHW format.
      reshaped_shape.push_back(input_shape[0]);
      int64_t block_elems = 1;
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(block_size_);
        block_elems *= block_size_;
      }
      reshaped_shape.push_back(input_shape[feature_dim] / block_elems);
      for (int i = 0; i < num_spatial_dims; ++i) {
        reshaped_shape.push_back(input_shape[2 + i]);
      }

      transpose_order.push_back(0);
      transpose_order.push_back(1 + num_spatial_dims);
      for (int i = 0; i < num_spatial_dims; ++i) {
        transpose_order.push_back(2 + num_spatial_dims + i);
        transpose_order.push_back(1 + i);
      }

      output_shape.push_back(input_shape[0]);
      output_shape.push_back(input_shape[feature_dim] / block_elems);
      for (int i = 0; i < num_spatial_dims; ++i) {
        output_shape.push_back(input_shape[2 + i] * block_size_);
      }
    }

    // Note: comments are given in NHWC format; NCHW is similar with a different
    // dimension order.
    // 1. Reshape `input` to `reshaped` of shape:
    //
    //      [batch,
    //       input_shape[1],
    //       input_shape[2],
    //       block_size_,
    //       block_size_,
    //       depth / (block_size_ * block_size_)]
    OP_REQUIRES(ctx,
                input_shape[feature_dim] % (block_size_ * block_size_) == 0,
                errors::InvalidArgument(
                    "Input depth dimension (", input_shape[3],
                    ") is not divisible by square of the block size (",
                    block_size_, ")"));

    xla::XlaOp reshaped = xla::Reshape(input, reshaped_shape);

    // 2. Permute dimensions of `reshaped` to produce
    //    `permuted_reshaped` of shape:
    //
    //      [batch,
    //       input_shape[1],
    //       block_size_,
    //       input_shape[2],
    //       block_size_,
    //       depth / (block_size_ * block_size_)]
    xla::XlaOp permuted_reshaped = xla::Transpose(reshaped, transpose_order);

    // 3. Reshape `permuted_reshaped` to flatten `block_shape` into the
    //    batch dimension, producing an output tensor of shape:
    //
    //      [batch,
    //       input_shape[1] * block_size_,
    //       input_shape[2] * block_size_,
    //       depth / (block_size_ * block_size_)]
    //
    xla::XlaOp output = xla::Reshape(permuted_reshaped, output_shape);

    // If this used to be a vectorized format turn it back now.
    if (data_format != data_format_) {
      DCHECK(data_format == FORMAT_NCHW && data_format_ == FORMAT_NCHW_VECT_C);
      auto output_reshaped = NCHWToNCHW_VECT_C(output);
      OP_REQUIRES_OK(ctx, output_reshaped.status());
      output = output_reshaped.value();
    }

    ctx->SetOutput(0, output);
  }

 private:
  TensorFormat data_format_;
  int block_size_;
};
REGISTER_XLA_OP(Name("DepthToSpace"), DepthToSpaceOp);

}  // namespace
}  // namespace tensorflow
