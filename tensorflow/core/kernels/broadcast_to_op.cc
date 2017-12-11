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
class BroadcastToOp : public OpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const TensorShape& input_shape = input_tensor.shape();

    const Tensor& shape_tensor = ctx->input(1);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   ctx->op_kernel().MakeShape(shape_tensor, &output_shape));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

#define BROADCAST_SHAPE(broadcast, reshape, NDIMS, input_shape, output_shape) \
  for (int i = 0; i < NDIMS; i++) {                                           \
    OP_REQUIRES(ctx, (broadcast[i] % reshape[i] == 0),                        \
                errors::InvalidArgument("invalid shape to broadcast from ",   \
                                        input_shape.DebugString(), " to ",    \
                                        output_shape.DebugString()));         \
    broadcast[i] = broadcast[i] / reshape[i];                                 \
  }

    switch (output_shape.dims()) {
      case 1: {
        auto reshape = AsEigenDSizesWithPrefix<1>(input_shape);
        auto broadcast = output_shape.AsEigenDSizes<1>();

        BROADCAST_SHAPE(broadcast, reshape, 1, input_shape, output_shape);

        auto output = output_tensor->tensor<T, 1>();
        switch (input_shape.dims()) {
          case 1: {
            auto input = input_tensor.tensor<T, 1>();
            output = input.broadcast(broadcast);
          } break;
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 2: {
        auto reshape = AsEigenDSizesWithPrefix<2>(input_shape);
        auto broadcast = output_shape.AsEigenDSizes<2>();

        BROADCAST_SHAPE(broadcast, reshape, 2, input_shape, output_shape);

        auto output = output_tensor->tensor<T, 2>();
        switch (input_shape.dims()) {
          case 1: {
            auto input = input_tensor.tensor<T, 1>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 2: {
            auto input = input_tensor.tensor<T, 2>();
            output = input.broadcast(broadcast);
          } break;
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 3: {
        auto reshape = AsEigenDSizesWithPrefix<3>(input_shape);
        auto broadcast = output_shape.AsEigenDSizes<3>();

        BROADCAST_SHAPE(broadcast, reshape, 3, input_shape, output_shape);

        auto output = output_tensor->tensor<T, 3>();
        switch (input_shape.dims()) {
          case 1: {
            auto input = input_tensor.tensor<T, 1>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 2: {
            auto input = input_tensor.tensor<T, 2>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 3: {
            auto input = input_tensor.tensor<T, 3>();
            output = input.broadcast(broadcast);
          } break;
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 4: {
        auto reshape = AsEigenDSizesWithPrefix<4>(input_shape);
        auto broadcast = output_shape.AsEigenDSizes<4>();

        BROADCAST_SHAPE(broadcast, reshape, 4, input_shape, output_shape);

        auto output = output_tensor->tensor<T, 4>();
        switch (input_shape.dims()) {
          case 1: {
            auto input = input_tensor.tensor<T, 1>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 2: {
            auto input = input_tensor.tensor<T, 2>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 3: {
            auto input = input_tensor.tensor<T, 3>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 4: {
            auto input = input_tensor.tensor<T, 4>();
            output = input.broadcast(broadcast);
          } break;
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 5: {
        auto reshape = AsEigenDSizesWithPrefix<5>(input_shape);
        auto broadcast = output_shape.AsEigenDSizes<5>();

        BROADCAST_SHAPE(broadcast, reshape, 5, input_shape, output_shape);

        auto output = output_tensor->tensor<T, 5>();
        switch (input_shape.dims()) {
          case 1: {
            auto input = input_tensor.tensor<T, 1>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 2: {
            auto input = input_tensor.tensor<T, 2>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 3: {
            auto input = input_tensor.tensor<T, 3>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 4: {
            auto input = input_tensor.tensor<T, 4>();
            output = input.reshape(reshape).broadcast(broadcast);
          } break;
          case 5: {
            auto input = input_tensor.tensor<T, 5>();
            output = input.broadcast(broadcast);
          } break;
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      default:
        ctx->CtxFailure(errors::InvalidArgument(
            "invalid shape to broadcast from ", input_shape.DebugString(),
            " to ", output_shape.DebugString()));
        break;
    }
  }

 private:
  template <int NDIMS>
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> AsEigenDSizesWithPrefix(
      const TensorShape& shape) const {
    Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes;
    for (int d = 0; d < NDIMS - shape.dims(); d++) {
      dsizes[d] = 1;
    }
    for (int d = NDIMS - shape.dims(); d < NDIMS; d++) {
      dsizes[d] = shape.dim_size(d - (NDIMS - shape.dims()));
    }
    return dsizes;
  }
};

// As MakeShape is able to handle both DT_INT32 and DT_INT64,
// no need to have TypeConstraint for `Tidx`
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BroadcastTo").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BroadcastToOp<type>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

}  // namespace tensorflow
