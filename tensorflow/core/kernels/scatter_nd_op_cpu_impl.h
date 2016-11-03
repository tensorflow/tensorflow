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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_

// Functor definitions for ScatterND ops, must be compilable by nvcc.

#define EIGEN_USE_THREADS

#include <atomic>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class OpKernelContext;

// Specialization of UpdateExecutor to CPU
namespace generator {

template <typename T, typename Index, scatter_nd_op::UpdateOp op>
class UpdateExecutor {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size);
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::ASSIGN> {
 public:
  static void Update(T* /* unused */, const T* updates, T* output,
                     Index slice_size) {
    std::copy_n(updates, slice_size, output);
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::ADD> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output, std::plus<T>());
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::SUB> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output, std::minus<T>());
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::MUL> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output,
                   std::multiplies<T>());
  }
};

template <typename T, typename Index>
class UpdateExecutor<T, Index, scatter_nd_op::UpdateOp::DIV> {
 public:
  static void Update(T* input, const T* updates, T* output, Index slice_size) {
    std::transform(input, input + slice_size, updates, output,
                   std::divides<T>());
  }
};

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
class ScatterNdSliceGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE ScatterNdSliceGenerator(
      const Index slice_size, typename TTypes<T, IXDIM + 1>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, IXDIM + 1>::Tensor Toutput,
      std::atomic<Index>* error_loc)
      : slice_size_(slice_size),
        Tparams_(Tparams),
        Tindices_(Tindices),
        Tupdates_(Tupdates),
        Toutput_(Toutput),
        error_loc_(error_loc) {}

  EIGEN_DEVICE_FUNC bool GenerateIndices(
      const Index loc, Eigen::array<Eigen::DenseIndex, IXDIM + 1>* ix) const {
    (*ix)[IXDIM] = 0;
    bool out_of_bounds = false;
    for (int i = 0; i < IXDIM; ++i) {
      const Index ix_i = internal::SubtleMustCopy(Tindices_(loc, i));
      (*ix)[i] = ix_i;
      out_of_bounds |= !FastBoundsCheck(ix_i, Tparams_.dimension(i));
    }
    return out_of_bounds;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE int32
  operator()(const Eigen::array<Eigen::DenseIndex, 1>& loc_array) const {
    auto loc = loc_array[0];
    Eigen::array<Eigen::DenseIndex, IXDIM + 1> ix_params;
    Eigen::array<Eigen::DenseIndex, 2> ix_updates;
    ix_updates[0] = loc;
    ix_updates[1] = 0;
    const bool out_of_bounds = GenerateIndices(loc, &ix_params);
    if (TF_PREDICT_FALSE(out_of_bounds)) {
      error_loc_->store(loc);
    } else {
      UpdateExecutor<T, Index, op>::Update(&Tparams_(ix_params),
                                           &Tupdates_(ix_updates),
                                           &Toutput_(ix_params), slice_size_);
    }
    return static_cast<int32>(0);  // Return something...
  }

 protected:
  const Index slice_size_;
  mutable typename TTypes<T, IXDIM + 1>::Tensor Tparams_;
  const typename TTypes<Index, 2>::ConstTensor Tindices_;
  const typename TTypes<T, 2>::ConstTensor Tupdates_;
  mutable typename TTypes<T, IXDIM + 1>::Tensor Toutput_;
  std::atomic<Index>* error_loc_;
};

}  // namespace generator

namespace functor {

// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdFunctor<CPUDevice, T, Index, op, IXDIM> {
  Index operator()(const CPUDevice& d, const Index slice_size,
                   typename TTypes<Index>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::Tensor Tparams,
                   typename TTypes<Index, 2>::ConstTensor Tindices,
                   typename TTypes<T, 2>::ConstTensor Tupdates,
                   typename TTypes<T, IXDIM + 1>::Tensor Toutput) {
    std::atomic<Index> error_loc(-1);

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::Tensor<Eigen::DenseIndex, 1>::Dimensions reshape_dims{{ 1 }};
    Eigen::array<Eigen::DenseIndex, 1> broadcast_dims{{ batch_size }};
#else
    Eigen::IndexList<Eigen::type2index<1> > reshape_dims;
    Eigen::IndexList<Eigen::DenseIndex> broadcast_dims;
    broadcast_dims.set(0, batch_size);
#endif

    generator::ScatterNdSliceGenerator<T, Index, op, IXDIM> generator(
        slice_size, Tparams, Tindices, Tupdates, Toutput, &error_loc);
    Tscratch.device(d) = Tscratch.reshape(reshape_dims)
                             .broadcast(broadcast_dims)
                             .generate(generator)
                             .sum();

    // error_loc() returns -1 if there's no out-of-bounds index,
    // otherwise it returns the location of an OOB index in Tindices.
    return error_loc.load();
  }
};

#define REGISTER_SCATTER_ND_FULL(T, Index, op)                               \
  template Index                                                             \
  ScatterNdFunctor<CPUDevice, T, Index, op, CPU_PROVIDED_IXDIM>::operator()( \
      const CPUDevice& d, const Index slice_size,                            \
      typename TTypes<Index>::Scalar Tscratch,                               \
      typename TTypes<T, CPU_PROVIDED_IXDIM + 1>::Tensor Tparams,            \
      typename TTypes<Index, 2>::ConstTensor Tindices,                       \
      typename TTypes<T, 2>::ConstTensor Tupdates,                           \
      typename TTypes<T, CPU_PROVIDED_IXDIM + 1>::Tensor Toutput)

#define REGISTER_SCATTER_ND_INDEX(type, op)  \
  REGISTER_SCATTER_ND_FULL(type, int32, op); \
  REGISTER_SCATTER_ND_FULL(type, int64, op)

#define REGISTER_SCATTER_ND_UPDATE(type) \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MATH(type)                           \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::SUB); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::MUL); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::DIV);

TF_CALL_ALL_TYPES(REGISTER_SCATTER_ND_UPDATE);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_MATH)

#undef REGISTER_SCATTER_ND_MATH
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_INDEX
#undef REGISTER_SCATTER_ND_FULL

}  // namespace functor

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
