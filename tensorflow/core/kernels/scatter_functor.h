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

#ifndef TENSORFLOW_KERNELS_SCATTER_FUNCTOR_H_
#define TENSORFLOW_KERNELS_SCATTER_FUNCTOR_H_

#include <type_traits>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpKernelContext;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif // TENSORFLOW_USE_SYCL

namespace scatter_op {

enum class UpdateOp { ASSIGN, ADD, SUB, MUL, DIV };

namespace internal {

template <scatter_op::UpdateOp Op>
struct Assign {};
template <>
struct Assign<scatter_op::UpdateOp::ASSIGN> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::ADD> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p += u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::SUB> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p -= u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MUL> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p *= u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::DIV> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p /= u;
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <scatter_op::UpdateOp Op>
struct AssignSYCL {};
template <>
struct AssignSYCL<scatter_op::UpdateOp::ASSIGN> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = u;
  }
};

template <>
struct AssignSYCL<scatter_op::UpdateOp::ADD> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) += u;
  }
};

template <>
struct AssignSYCL<scatter_op::UpdateOp::SUB> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) -= u;
  }
};

template <>
struct AssignSYCL<scatter_op::UpdateOp::MUL> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p * u;
  }
};

template <>
struct AssignSYCL<scatter_op::UpdateOp::DIV> {
  template <typename Device, typename Params, typename Update>
  static void Run(Device d, Params p, Update u) {
    p.device(d) = p / u;
  }
};
#endif // TENSORFLOW_USE_SYCL

}  // namespace internal
}  // namespace scatter_op

namespace functor {
template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices);
};

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctorBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  An earlier version of the
      // code checked it and then grabbed it from memory a second time, which
      // was a security risk since it could have changed in between.
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::Assign<op>::Run(params.template chip<0>(index),
                                            updates.template chip<0>(i));
    }
    return -1;
  }
};

#ifdef TENSORFLOW_USE_SYCL
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctorBase <SYCLDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const SYCLDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  An earlier version of the
      // code checked it and then grabbed it from memory a second time, which
      // was a security risk since it could have changed in between.
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::AssignSYCL<op>::Run(d, params.template chip<0>(index),
                                            updates.template chip<0>(i));
    }
    return -1;
  }
};
#endif // TENSORFLOW_USE_SYCL

template <typename T, typename Index>
struct ScatterFunctorBase<CPUDevice, T, Index, scatter_op::UpdateOp::ASSIGN> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    if (!std::is_same<T, string>::value) {
      for (Index i = 0; i < N; i++) {
        // Grab the index and check its validity.  An earlier version of the
        // code checked it and then grabbed it from memory a second time, which
        // was a security risk since it could have changed in between.
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) return i;
        memmove(params.data() + index * params.dimension(1),
                updates.data() + i * updates.dimension(1),
                updates.dimension(1) * sizeof(T));
      }
    } else {
      for (Index i = 0; i < N; i++) {
        // Grab the index and check its validity.  An earlier version of the
        // code checked it and then grabbed it from memory a second time, which
        // was a security risk since it could have changed in between.
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) return i;
        // Copy last Ndim-1 dimensions of updates[i] to params[index]
        scatter_op::internal::Assign<scatter_op::UpdateOp::ASSIGN>::Run(
            params.template chip<0>(index), updates.template chip<0>(i));
      }
    }
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<CPUDevice, T, Index, op>
        : ScatterFunctorBase<CPUDevice, T, Index, op>{};

#ifdef TENSORFLOW_USE_SYCL
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctorSYCL {
  Index operator()(OpKernelContext* c, const SYCLDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::Flat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::AssignSYCL<op>::Run(
          d, params.template chip<0>(index), updates.template chip<0>(i));
    }
    return -1;
  }
};
#endif // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SCATTER_FUNCTOR_H_
