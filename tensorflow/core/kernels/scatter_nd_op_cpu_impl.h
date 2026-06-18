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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_

// Functor definitions for ScatterND ops, must be compilable by nvcc.

#define EIGEN_USE_THREADS

#include <atomic>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class OpKernelContext;

// Specialization of UpdateExecutor to CPU
namespace update_executor {

template <typename T, typename Input, typename Update, typename Output,
          scatter_nd_op::UpdateOp OP>
class UpdateExecutor {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input value,
                                          Update update, Output output);
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output,
                     scatter_nd_op::UpdateOp::ASSIGN> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
    output.device(device) = update;
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::ADD> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
    output.device(device) += update;
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::SUB> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
    output.device(device) -= update;
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::MIN> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
    output.device(device) = output.cwiseMin(update);
  }
};

template <typename T, typename Input, typename Update, typename Output>
class UpdateExecutor<T, Input, Update, Output, scatter_nd_op::UpdateOp::MAX> {
 public:
  EIGEN_STRONG_INLINE static void Execute(const T& device, Input /* input */,
                                          Update update, Output output) {
    output.device(device) = output.cwiseMax(update);
  }
};

}  // namespace update_executor

namespace functor {

// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_nd_op::UpdateOp OP, int IXDIM>
struct ScatterNdFunctor<CPUDevice, T, Index, OP, IXDIM> {
  Index operator()(
      const CPUDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput) {
    // error_loc is -1 if there's no out-of-bounds index,
    // otherwise it is the location of an OOB index in Tindices.
    Index error_loc = -1;

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);

    Index batch_strides[IXDIM];
    if (IXDIM > 0) {
      batch_strides[IXDIM - 1] = 1;
    }
    for (int dim = IXDIM - 2; dim >= 0; --dim) {
      batch_strides[dim] =
          batch_strides[dim + 1] * output_shape_prefix[dim + 1];
    }

    for (Eigen::DenseIndex loc = 0; loc < batch_size; ++loc) {
      Index i = 0;
      bool out_of_bounds = false;
      for (int dim = 0; dim < IXDIM; ++dim) {
        const Index ix_d = internal::SubtleMustCopy(Tindices(loc, dim));
        out_of_bounds |= !FastBoundsCheck(ix_d, output_shape_prefix[dim]);
        i += ix_d * batch_strides[dim];
      }
      if (TF_PREDICT_FALSE(out_of_bounds)) {
        error_loc = loc;
        // Don't break the loop here, but continue to update the rest because
        // the caller might ignore bad indices.
        continue;
      } else {
        auto input_chip = Toutput.template chip<0>(i);
        auto output_chip = input_chip;
        auto update_chip = Tupdates.template chip<0>(loc);
        update_executor::UpdateExecutor<
            CPUDevice, decltype(input_chip), decltype(update_chip),
            decltype(output_chip), OP>::Execute(d, input_chip, update_chip,
                                                output_chip);
      }
    }

    return error_loc;
  }
};

#define REGISTER_SCATTER_ND_FULL(T, Index, op)                               \
  template Index                                                             \
  ScatterNdFunctor<CPUDevice, T, Index, op, CPU_PROVIDED_IXDIM>::operator()( \
      const CPUDevice& d, const Index slice_size,                            \
      const Eigen::array<Eigen::DenseIndex, CPU_PROVIDED_IXDIM>              \
          output_shape_prefix,                                               \
      typename TTypes<T, 2>::Tensor Tparams,                                 \
      typename TTypes<Index, 2>::ConstTensor Tindices,                       \
      typename TTypes<T, 2>::ConstTensor Tupdates,                           \
      typename TTypes<T, 2>::Tensor Toutput)

#define REGISTER_SCATTER_ND_INDEX(type, op)  \
  REGISTER_SCATTER_ND_FULL(type, int32, op); \
  REGISTER_SCATTER_ND_FULL(type, int64, op)

#define REGISTER_SCATTER_ND_UPDATE(type) \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MATH(type)                           \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::SUB);

#define REGISTER_SCATTER_ND_MIN_MAX(type)                        \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::MAX); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::MIN);

TF_CALL_ALL_TYPES(REGISTER_SCATTER_ND_UPDATE);
REGISTER_SCATTER_ND_INDEX(tstring, scatter_nd_op::UpdateOp::ADD);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_MATH);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_SCATTER_ND_MIN_MAX);
TF_CALL_bool(REGISTER_SCATTER_ND_MATH);

#undef REGISTER_SCATTER_ND_MATH
#undef REGISTER_SCATTER_ND_MIN_MAX
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_INDEX
#undef REGISTER_SCATTER_ND_FULL
}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
