/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_CPU_IMPL_H_

// Specialization of GatherNdSlice to CPU

#define EIGEN_USE_THREADS

#include <atomic>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace generator {

template <typename T, typename Index, int IXDIM>
class GatherNdSliceGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE GatherNdSliceGenerator(
      const Index slice_size, typename TTypes<Index>::ConstMatrix Tindices,
      typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
      typename TTypes<T>::Matrix Tout, std::atomic<Index>* error_loc)
      : slice_size_(slice_size),
        Tindices_(Tindices),
        Tparams_(Tparams),
        Tout_(Tout),
        error_loc_(error_loc) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE bool GenerateIndices(
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
    const Index loc = loc_array[0];
    Eigen::array<Eigen::DenseIndex, IXDIM + 1> ix;
    Eigen::array<Eigen::DenseIndex, 2> ix_out;
    ix_out[0] = loc;
    ix_out[1] = 0;
    const bool out_of_bounds = GenerateIndices(loc, &ix);
    if (TF_PREDICT_FALSE(out_of_bounds)) {
      error_loc_->store(loc);
      std::fill_n(&Tout_(ix_out), slice_size_, T());
    } else {
      std::copy_n(&Tparams_(ix), slice_size_, &Tout_(ix_out));
    }

    return static_cast<int32>(0);  // Return something...
  }

 private:
  const Index slice_size_;
  const typename TTypes<Index>::ConstMatrix Tindices_;
  const typename TTypes<T, IXDIM + 1>::ConstTensor Tparams_;
  mutable typename TTypes<T>::Matrix Tout_;
  std::atomic<Index>* error_loc_;
};

}  // namespace generator

namespace functor {

template <typename T, typename Index, int IXDIM>
struct GatherNdSlice<CPUDevice, T, Index, IXDIM> {
  Index operator()(const CPUDevice& d, const Index slice_size,
                   typename TTypes<int32>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout) {
    std::atomic<Index> error_loc(-1);
    const Eigen::Index batch_size = Tindices.dimension(0);
    generator::GatherNdSliceGenerator<T, Index, IXDIM> gather_nd_generator(
        slice_size, Tindices, Tparams, Tout, &error_loc);

    auto compute_shard = [&](Eigen::Index begin, Eigen::Index end) {
      for (Eigen::Index i = begin; i < end; ++i) {
        const Eigen::array<Eigen::Index, 1> loc{i};
        gather_nd_generator(loc);
      }
    };
    Eigen::Index bytes_moved = sizeof(T) * (slice_size + IXDIM);
    auto cost = Eigen::TensorOpCost(bytes_moved /* bytes loaded */,
                                    bytes_moved /* bytes stored */,
                                    slice_size + IXDIM /* compute cycles */);
    d.parallelFor(batch_size, cost, compute_shard);

    // error_loc() returns -1 if there's no out-of-bounds index,
    // otherwise it returns the location of an OOB index in Tindices.
    return error_loc.load();
  }
};

#define REGISTER_GATHER_ND_FULL(T, Index)                                     \
  template Index GatherNdSlice<CPUDevice, T, Index, CPU_PROVIDED_IXDIM>::     \
  operator()(const CPUDevice& d, const Index slice_size,                      \
             typename TTypes<int32>::Scalar Tscratch,                         \
             typename TTypes<T, CPU_PROVIDED_IXDIM + 1>::ConstTensor Tparams, \
             typename TTypes<Index>::ConstMatrix Tindices,                    \
             typename TTypes<T>::Matrix Tout);

#define REGISTER_GATHER_ND_CPU(type)    \
  REGISTER_GATHER_ND_FULL(type, int32); \
  REGISTER_GATHER_ND_FULL(type, int64)

TF_CALL_ALL_TYPES(REGISTER_GATHER_ND_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_GATHER_ND_CPU);

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_ND_OP_CPU_IMPL_H_
