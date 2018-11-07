/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BROADCAST_TO_OP_H_
#define TENSORFLOW_CORE_KERNELS_BROADCAST_TO_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"

namespace tensorflow {

namespace functor {

#define BROADCAST_SHAPE(NDIMS, input_shape, output_shape)                 \
  auto reshape = AsEigenDSizesWithPrefix<NDIMS>(input_shape);             \
  auto broadcast = output_shape.AsEigenDSizes<NDIMS>();                   \
  auto reshape_32bit = AsEigenDSizesWithPrefix<NDIMS, int>(input_shape);  \
  auto broadcast_32bit = output_shape.AsEigenDSizes<NDIMS, int>();        \
  if (input_shape.dims() > 0) {                                           \
    for (int i = 0; i < NDIMS; i++) {                                     \
      if (reshape[i] != broadcast[i]) {                                   \
        OP_REQUIRES(                                                      \
            ctx, ((reshape[i] != 0) && (broadcast[i] % reshape[i] == 0)), \
            errors::InvalidArgument("invalid shape to broadcast from ",   \
                                    input_shape.DebugString(), " to ",    \
                                    output_shape.DebugString()));         \
        broadcast[i] = broadcast[i] / reshape[i];                         \
      } else {                                                            \
        broadcast[i] = 1;                                                 \
      }                                                                   \
      if (can_use_32bit) {                                                \
        broadcast_32bit[i] = static_cast<int>(broadcast[i]);              \
      }                                                                   \
    }                                                                     \
  }

#define HANDLE_BROADCAST_FROM_SCALAR()                              \
  if (std::is_same<Eigen::GpuDevice, Device>::value) {              \
    FillFunctor<Device, T>()(d, output_tensor.flat<T>(),            \
                             input_tensor.scalar<T>());             \
  } else {                                                          \
    output.device(d) = output.constant(input_tensor.scalar<T>()()); \
  }

#define HANDLE_BROADCAST_CASE(dim_i)                                        \
  case dim_i: {                                                             \
    if (can_use_32bit) {                                                    \
      auto input = input_tensor.tensor<T, dim_i>();                         \
      To32Bit(output).device(d) =                                           \
          To32Bit(input).reshape(reshape_32bit).broadcast(broadcast_32bit); \
    } else {                                                                \
      auto input = input_tensor.tensor<T, dim_i>();                         \
      output.device(d) = input.reshape(reshape).broadcast(broadcast);       \
    }                                                                       \
  } break

template <typename Device, typename T>
struct BroadcastTo {
  void operator()(const Device &d, OpKernelContext *ctx, Tensor &output_tensor,
                  const TensorShape &output_shape, const Tensor &input_tensor,
                  const TensorShape &input_shape) {
    if (output_shape.num_elements() == 0) {
      return;
    }
    if (output_shape == input_shape) {
      output_tensor.flat<T>().device(d) = input_tensor.flat<T>();
      return;
    }

    const bool can_use_32bit = std::is_same<Eigen::GpuDevice, Device>::value &&
                               output_tensor.NumElements() < kint32max &&
                               input_tensor.NumElements() < kint32max;

    switch (output_shape.dims()) {
      case 0: {
        if (input_shape.dims() > 0) {
          ctx->CtxFailure(errors::InvalidArgument(
              "invalid shape to broadcast from ", input_shape.DebugString(),
              " to ", output_shape.DebugString()));
          break;
        }
        output_tensor.scalar<T>().device(d) = input_tensor.scalar<T>();
        break;
      }
      case 1: {
        BROADCAST_SHAPE(1, input_shape, output_shape);

        auto output = output_tensor.tensor<T, 1>();
        switch (input_shape.dims()) {
          case 0: {
            HANDLE_BROADCAST_FROM_SCALAR();
          } break;
            HANDLE_BROADCAST_CASE(1);
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 2: {
        BROADCAST_SHAPE(2, input_shape, output_shape);
        auto output = output_tensor.tensor<T, 2>();
        switch (input_shape.dims()) {
          case 0: {
            HANDLE_BROADCAST_FROM_SCALAR();
          } break;
            HANDLE_BROADCAST_CASE(1);
            HANDLE_BROADCAST_CASE(2);
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 3: {
        BROADCAST_SHAPE(3, input_shape, output_shape);
        auto output = output_tensor.tensor<T, 3>();
        switch (input_shape.dims()) {
          case 0: {
            HANDLE_BROADCAST_FROM_SCALAR();
          } break;
            HANDLE_BROADCAST_CASE(1);
            HANDLE_BROADCAST_CASE(2);
            HANDLE_BROADCAST_CASE(3);
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 4: {
        BROADCAST_SHAPE(4, input_shape, output_shape);
        auto output = output_tensor.tensor<T, 4>();
        switch (input_shape.dims()) {
          case 0: {
            HANDLE_BROADCAST_FROM_SCALAR();
          } break;
            HANDLE_BROADCAST_CASE(1);
            HANDLE_BROADCAST_CASE(2);
            HANDLE_BROADCAST_CASE(3);
            HANDLE_BROADCAST_CASE(4);
          default:
            ctx->CtxFailure(errors::InvalidArgument(
                "invalid shape to broadcast from ", input_shape.DebugString(),
                " to ", output_shape.DebugString()));
            break;
        }
      } break;
      case 5: {
        BROADCAST_SHAPE(5, input_shape, output_shape);
        auto output = output_tensor.tensor<T, 5>();
        switch (input_shape.dims()) {
          case 0: {
            HANDLE_BROADCAST_FROM_SCALAR();
          } break;
            HANDLE_BROADCAST_CASE(1);
            HANDLE_BROADCAST_CASE(2);
            HANDLE_BROADCAST_CASE(3);
            HANDLE_BROADCAST_CASE(4);
            HANDLE_BROADCAST_CASE(5);
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
  template <int NDIMS, typename DimType = Eigen::DenseIndex>
  Eigen::DSizes<DimType, NDIMS> AsEigenDSizesWithPrefix(
      const TensorShape &shape) const {
    Eigen::DSizes<DimType, NDIMS> dsizes;
    for (int d = 0; d < NDIMS - shape.dims(); d++) {
      dsizes[d] = 1;
    }
    for (int d = NDIMS - shape.dims(); d < NDIMS; d++) {
      dsizes[d] =
          static_cast<DimType>(shape.dim_size(d - (NDIMS - shape.dims())));
    }
    return dsizes;
  }
};

#undef BROADCAST_SHAPE
#undef HANDLE_BROADCAST_FROM_SCALAR
#undef HANDLE_BROADCAST_CASE

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BROADCAST_TO_OP_H_
