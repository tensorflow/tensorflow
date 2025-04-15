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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_H_

#include <type_traits>

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

class OpKernelContext;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace scatter_op {

enum class UpdateOp { ASSIGN, ADD, SUB, MUL, DIV, MIN, MAX };

namespace internal {

template <scatter_op::UpdateOp Op>
struct Assign {};
template <>
struct Assign<scatter_op::UpdateOp::ASSIGN> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p.setConstant(u);
  }
};
template <>
struct Assign<scatter_op::UpdateOp::ADD> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p += u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p + u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::SUB> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p -= u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p + static_cast<Update>(-u);
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MUL> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p *= u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p * u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::DIV> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p /= u;
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p / u;
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MIN> {
  // This method requires that Params and Update are tensor types.
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = p.cwiseMin(u);
  }
  // Same thing, but for Update being a scalar type.
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p.cwiseMin(u);
  }
};
template <>
struct Assign<scatter_op::UpdateOp::MAX> {
  template <typename Params, typename Update>
  static void Run(Params p, Update u) {
    p = p.cwiseMax(u);
  }
  template <typename Params, typename Update>
  static void RunScalar(Params p, Update u) {
    p = p.cwiseMax(u);
  }
};


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
  Index ParallelExecute(OpKernelContext* c, const Device& d,
                        typename TTypes<T>::Matrix params,
                        typename TTypes<T>::ConstMatrix updates,
                        typename TTypes<Index>::ConstFlat indices) {
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index kMaxLocks = 1024;
    const Index entries_per_lock = (limit + kMaxLocks - 1) / kMaxLocks;
    // To reduce the number of locks and the memory usage, we divide the whole
    // index space into kMaxLocks regions with each lock serializing access to
    // a region.
    mutex accessed[kMaxLocks];
    std::atomic<Index> bad_index(-1);
    auto ParallelScatter = [&](Index start, Index end) {
      for (Index i = start; i < end; ++i) {
        // Grab the index and check its validity.  Do this carefully,
        // to avoid checking the value and grabbing it again from
        // memory a second time (a security risk since it may change in
        // between).
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) {
          bad_index = i;
          return;
        }
        const Index lock_id = index / entries_per_lock;
        // Copy last Ndim-1 dimensions of updates[i] to params[index]
        {
          mutex_lock l(accessed[lock_id]);
          scatter_op::internal::Assign<op>::Run(params.template chip<0>(index),
                                                updates.template chip<0>(i));
        }
      }
    };
    const float kMovingCost = 2.5f;
    float shard_cost = kMovingCost * params.dimension(1);
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(c->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, N, shard_cost,
          ParallelScatter);  // TODO: Come up with a good cost estimate.
    return bad_index;
  }
  Index SerialExecute(OpKernelContext* c, const Device& d,
                      typename TTypes<T>::Matrix params,
                      typename TTypes<T>::ConstMatrix updates,
                      typename TTypes<Index>::ConstFlat indices) {
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; ++i) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in
      // between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      scatter_op::internal::Assign<op>::Run(params.template chip<0>(index),
                                            updates.template chip<0>(i));
    }
    return -1;
  }

  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
#ifdef PLATFORM_GOOGLE
    // The parallel version is significantly slower internally. Only call the
    // serial version for now.
    // TODO(penporn): Avoid locking in parallelization (sort beforehand).
    return SerialExecute(c, d, params, updates, indices);
#else
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index min_n_threshold = 1024;
    const Index ser_par_ratio = 10000;
    // For parallelizing the updates, duplicate entries need to be handled
    // correctly. Multiple updates to the same index has to be serialized.
    // This can lead to lock contention which may nullify the benefits of
    // parallelization. Assuming uniform random distribution of the indices, we
    // come up with a rough heuristic and determine whether the updates execute
    // serially or parallelly. Also if 'N' is small, overheads of parallel
    // execution outweigh its benefits and hence we check the value of N.
    const bool execute_serial = N < min_n_threshold ||
                                (N / limit) > ser_par_ratio ||
                                OpDeterminismRequired();
    if (execute_serial)
      return SerialExecute(c, d, params, updates, indices);
    else
      return ParallelExecute(c, d, params, updates, indices);
#endif  // PLATFORM_GOOGLE
  }
};

template <typename Device, typename Index>
struct ScatterFunctorVariantAssignBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<Variant>::Matrix params,
                   typename TTypes<Variant>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index cols = static_cast<Index>(params.dimension(1));
    DCHECK_EQ(N, updates.dimension(0));
    DCHECK_EQ(cols, updates.dimension(1));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Copy last Ndim-1 dimensions of updates[i] to params[index]
      for (int j = 0; j < cols; ++j) {
        const Variant& to_scatter = updates(i, j);
        params(index, j) = to_scatter;
      }
    }
    return -1;
  }
};

template <typename Index>
struct ScatterFunctor<CPUDevice, Variant, Index, scatter_op::UpdateOp::ASSIGN>
    : ScatterFunctorVariantAssignBase<CPUDevice, Index> {};

template <typename Index>
struct ScatterFunctor<GPUDevice, Variant, Index, scatter_op::UpdateOp::ASSIGN>
    : ScatterFunctorVariantAssignBase<GPUDevice, Index> {};


template <typename T, typename Index>
struct ScatterFunctorBase<CPUDevice, T, Index, scatter_op::UpdateOp::ASSIGN> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    if (!std::is_same<T, tstring>::value) {
      for (Index i = 0; i < N; i++) {
        // Grab the index and check its validity.  Do this carefully,
        // to avoid checking the value and grabbing it again from
        // memory a second time (a security risk since it may change in
        // between).
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) return i;
        memmove(params.data() + index * params.dimension(1),
                updates.data() + i * updates.dimension(1),
                updates.dimension(1) * sizeof(T));
      }
    } else {
      for (Index i = 0; i < N; i++) {
        // Grab the index and check its validity.  Do this carefully,
        // to avoid checking the value and grabbing it again from
        // memory a second time (a security risk since it may change in
        // between).
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
    : ScatterFunctorBase<CPUDevice, T, Index, op> {};


template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices);
};

template <typename Device, typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctorBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      scatter_op::internal::Assign<op>::RunScalar(
          params.template chip<0>(index), update());
    }
    return -1;
  }
};

template <typename Device, typename Index>
struct ScatterScalarFunctorVariantAssignBase {
  Index operator()(OpKernelContext* c, const Device& d,
                   typename TTypes<Variant>::Matrix params,
                   const typename TTypes<Variant>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    const Index cols = static_cast<Index>(params.dimension(1));
    const Variant& to_scatter = update();
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      for (Index j = 0; j < cols; ++j) {
        params(index, j) = to_scatter;
      }
    }
    return -1;
  }
};

template <typename Index>
struct ScatterScalarFunctor<CPUDevice, Variant, Index,
                            scatter_op::UpdateOp::ASSIGN>
    : ScatterScalarFunctorVariantAssignBase<CPUDevice, Index> {};
template <typename Index>
struct ScatterScalarFunctor<GPUDevice, Variant, Index,
                            scatter_op::UpdateOp::ASSIGN>
    : ScatterScalarFunctorVariantAssignBase<GPUDevice, Index> {};


template <typename T, typename Index>
struct ScatterScalarFunctorBase<CPUDevice, T, Index,
                                scatter_op::UpdateOp::ASSIGN> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    for (Index i = 0; i < N; i++) {
      // Grab the index and check its validity.  Do this carefully,
      // to avoid checking the value and grabbing it again from
      // memory a second time (a security risk since it may change in between).
      const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, limit)) return i;
      // Broadcast update to params[index]
      scatter_op::internal::Assign<scatter_op::UpdateOp::ASSIGN>::RunScalar(
          params.template chip<0>(index), update());
    }
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<CPUDevice, T, Index, op>
    : ScatterScalarFunctorBase<CPUDevice, T, Index, op> {};


}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_H_
