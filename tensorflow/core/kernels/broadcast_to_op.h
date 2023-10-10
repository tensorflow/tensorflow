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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct BroadcastTo {
  template <int NDIMS>
  void DoBCast(
      const Device &device, typename TTypes<T, NDIMS>::Tensor out,
      typename TTypes<T, NDIMS>::ConstTensor in,
      const typename Eigen::array<Eigen::DenseIndex, NDIMS> &bcast) const {
    MaybeWith32BitIndexing<Device>(
        [&](auto out32, auto in32, const auto &bcast32) {
          out32.device(device) = in32.broadcast(bcast32);
        },
        out, in, bcast);
  }

  template <int NDIMS>
  void ReshapeAndBCast(const Device &device, Tensor &output_tensor,
                       const Tensor &input_tensor, const BCast &bcast) const {
    DoBCast<NDIMS>(
        device, output_tensor.template shaped<T, NDIMS>(bcast.result_shape()),
        input_tensor.template shaped<T, NDIMS>(bcast.x_reshape()),
        BCast::ToIndexArrayType<Eigen::DenseIndex, NDIMS>(bcast.x_bcast()));
  }

  // PRECONDITION: rank(input_shape) > 0 &&
  //               rank(input_shape) <= rank(output_shape)  &&
  //               output_shape.num_elements() > 0.
  void operator()(const Device &device, OpKernelContext *ctx,
                  Tensor &output_tensor, const TensorShape &output_shape,
                  const Tensor &input_tensor, const TensorShape &input_shape,
                  const BCast &bcast) const {
    const int ndims = bcast.y_reshape().size();
    switch (ndims) {
      case 1:
        ReshapeAndBCast<1>(device, output_tensor, input_tensor, bcast);
        break;
      case 2:
        ReshapeAndBCast<2>(device, output_tensor, input_tensor, bcast);
        break;
      case 3:
        ReshapeAndBCast<3>(device, output_tensor, input_tensor, bcast);
        break;
      case 4:
        ReshapeAndBCast<4>(device, output_tensor, input_tensor, bcast);
        break;
      case 5:
        ReshapeAndBCast<5>(device, output_tensor, input_tensor, bcast);
        break;
      default:
        ctx->SetStatus(errors::Unimplemented(
            "Broadcast between ", input_shape.DebugString(), " and ",
            output_shape.DebugString(), " is not supported yet."));
        break;
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BROADCAST_TO_OP_H_
