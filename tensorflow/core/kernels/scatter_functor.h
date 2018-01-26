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
#include <atomic>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

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

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<CPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const CPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // Indices and params sizes were validated in DoCompute().
    const Index N = static_cast<Index>(indices.size());
    const Index limit = static_cast<Index>(params.dimension(0));
    auto worker_threads = c->device()->tensorflow_cpu_worker_threads();

    // Limit the max number of flags in case of too much indices.
    // Based on the shard strategy in work_sharder.cc to limit the flags' size.
    // Assume each cost unit ues one flag, and the minCostPerShard is 10000,
    //   so limit the factor to 10000.
    static constexpr int64 kMaxFactorPerThread = 10000;
    Index flags_size = static_cast<Index>(kMaxFactorPerThread * worker_threads->num_threads);
    flags_size = N <= flags_size ? N : flags_size;
    // use atomic_flag and spin lock for multiple thread.
    auto flags = std::unique_ptr<std::atomic_flag[]>(
            new std::atomic_flag[flags_size]{ATOMIC_FLAG_INIT});

    // Store the result.
    mutex mu;
    Index result = -1;
    auto work = [&params, &updates, &indices, &flags, flags_size, limit, &mu, &result]
            (int64 start, int64 end) {
      std::function<void(Index index, Index i)> assign_func;
      if (op != scatter_op::UpdateOp::ASSIGN || std::is_same<T, string>::value) {
        assign_func = [&params, &updates](Index index, Index i) {
          scatter_op::internal::Assign<op>::Run(params.template chip<0>(index),
                                                updates.template chip<0>(i));
        };
      } else {
        assign_func = [&params, &updates](Index index, Index i) {
          memmove(params.data() + index * params.dimension(1),
                  updates.data() + i * updates.dimension(1),
                  updates.dimension(1) * sizeof(T));
        };
      }
      Index flags_idx;
      for (Index i = start; i < end; i++) {
        // Grab the index and check its validity.  An earlier version of the
        // code checked it and then grabbed it from memory a second time, which
        // was a security risk since it could have changed in between.
        const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, limit)) {
          mutex_lock l(mu);
          result = i;
          return;
        }
        // Acquire lock:
        flags_idx = index % flags_size;
        while (flags[flags_idx].test_and_set(std::memory_order_acquire))
          ;
        // Copy last Ndim-1 dimensions of updates[i] to params[index].
        assign_func(index, i);
        // release lock
        flags[flags_idx].clear(std::memory_order_release);
      }
    };

    Shard(worker_threads->num_threads, worker_threads->workers, N,
          updates.dimension(1) * sizeof(T), work);
    return result;
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
