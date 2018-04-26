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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
class RepeatOp : public OpKernel {
 public:
  explicit RepeatOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), has_axis_(false), axis_(0) {
    if (ctx->GetAttr("axis", &axis_).ok()) {
      has_axis_ = true;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);

    const Tensor& repeats_tensor = ctx->input(1);

    // repeats is either 0-D or 1-D
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsScalar(repeats_tensor.shape()) ||
            TensorShapeUtils::IsVector(repeats_tensor.shape()),
        errors::InvalidArgument("repeats must be a 0-D or 1-D, but got: ",
                                repeats_tensor.shape().DebugString()));
    auto repeats_flat = repeats_tensor.flat<int32>();

    Tensor* output_tensor = nullptr;
    if (has_axis_) {
      TensorShape output_shape(input_tensor.shape());
      if (TensorShapeUtils::IsScalar(input_tensor.shape())) {
        // If Scalar, then treat as [1]
        output_shape.AddDim(1);
      }
      OP_REQUIRES(ctx, (axis_ < output_shape.dims()),
                  errors::InvalidArgument(
                      "axis must be < ", output_shape.dims(), ", got ", axis_));

      OP_REQUIRES(
          ctx,
          (repeats_flat.size() == 1 ||
           repeats_flat.size() == output_shape.dim_size(axis_)),
          errors::InvalidArgument(
              "repeats must have the same size as input, or 1, but got input ",
              output_shape.dim_size(axis_), ", repeats ", repeats_flat.size()));

      // reshape input so that axis is in the middle
      std::vector<int64> sizes{1, output_shape.dim_size(axis_), 1};
      if (axis_ > 0) {
        for (int64 i = 0; i < axis_; i++) {
          sizes[0] *= output_shape.dim_size(i);
        }
      }
      if (axis_ + 1 < output_shape.dims()) {
        for (int64 i = axis_ + 1; i < output_shape.dims(); i++) {
          sizes[2] *= output_shape.dim_size(i);
        }
      }
      auto input = input_tensor.shaped<T, 3>(sizes);

      if (repeats_flat.size() == 1) {
        output_shape.set_dim(
            axis_, input_tensor.shape().dim_size(axis_) * repeats_flat(0));
      } else {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> output_size =
            repeats_flat.sum();
        output_shape.set_dim(axis_, output_size());
      }
      sizes[1] = output_shape.dim_size(axis_);

      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, output_shape, &output_tensor));
      auto output = output_tensor->shaped<T, 3>(sizes);

      int offset = 0;
      for (int64 i = 0; i < input.dimension(1); i++) {
        int64 repeats_value =
            repeats_flat.size() == 1 ? repeats_flat(0) : repeats_flat(i);
        for (int64 r = 0; r < repeats_value; r++) {
          output.chip(offset + r, 1) = input.chip(i, 1);
        }
        offset += repeats_flat(i);
      }
    } else {
      // If axis is not present, treat input as flat
      auto input_flat = input_tensor.flat<T>();

      OP_REQUIRES(
          ctx,
          (repeats_flat.size() == 1 ||
           repeats_flat.size() == input_flat.size()),
          errors::InvalidArgument(
              "repeats must have the same size as input, or 1, but got input ",
              input_flat.size(), ", repeats ", repeats_flat.size()));
      if (repeats_flat.size() == 1) {
        // If repeats only have one element, do a broadast
        OP_REQUIRES_OK(
            ctx,
            ctx->allocate_output(
                0, TensorShape({input_tensor.NumElements() * repeats_flat(0)}),
                &output_tensor));
        Eigen::array<int64, 2> input_reshape({input_tensor.NumElements(), 1});
        Eigen::array<int64, 2> bcast({1, repeats_flat(0)});
        Eigen::array<int64, 2> output_reshape(
            {input_tensor.NumElements() * repeats_flat(0), 1});
        auto output_vec = output_tensor->vec<T>();
        output_vec = input_flat.reshape(input_reshape)
                         .broadcast(bcast)
                         .reshape(output_reshape);
      } else {
        Eigen::Tensor<int32, 0, Eigen::RowMajor> output_size =
            repeats_flat.sum();
        OP_REQUIRES_OK(ctx,
                       ctx->allocate_output(0, TensorShape({output_size()}),
                                            &output_tensor));
        auto output_vec = output_tensor->vec<T>();
        int64 offset = 0;
        for (int64 i = 0; i < repeats_flat.size(); i++) {
          output_vec
              .slice(Eigen::array<int64, 1>({offset}),
                     Eigen::array<int64, 1>({repeats_flat(i)}))
              .setConstant(input_flat(i));
          offset += repeats_flat(i);
        }
      }
    }
  }

 private:
  bool has_axis_;
  int axis_;
};

#define REGISTER_KERNEL(type)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Repeat").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      RepeatOp<type>);                                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("RepeatFlat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      RepeatOp<type>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow