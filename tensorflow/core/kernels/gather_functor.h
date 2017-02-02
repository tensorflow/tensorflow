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

#ifndef TENSORFLOW_KERNELS_GATHER_FUNCTOR_H_
#define TENSORFLOW_KERNELS_GATHER_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

// Helper method to copy using memcpy.
template <typename T, typename Index, typename SliceIndex,
          SliceIndex static_slice_elems>
SliceIndex HandleCopies(typename TTypes<T>::ConstMatrix params,
                        typename TTypes<Index>::ConstFlat indices,
                        SliceIndex slice_elems,
                        typename TTypes<T>::Matrix out) {
  const SliceIndex first_dim_size =
      static_cast<SliceIndex>(indices.dimension(0));
  const Index limit = static_cast<Index>(params.dimension(0));
  T* out_base = &out(0, 0);
  const T* params_base = &params(0, 0);
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    slice_elems = static_slice_elems;
  }
  // Compute slice_bytes here so that static knowledge is available
  const size_t slice_bytes = slice_elems * sizeof(T);
  for (SliceIndex i = 0; i < first_dim_size; i++) {
    const SliceIndex j = i + 1;
    if (j < first_dim_size) {
      port::prefetch<port::PREFETCH_HINT_T0>(&params(indices(j), 0));
      port::prefetch<port::PREFETCH_HINT_T0>(&out(j, 0));
    }
    // Grab the index and check its validity.  An earlier version of the
    // code checked it and then grabbed it from memory a second time, which
    // was a security risk since it could have changed in between.
    const Index index = internal::SubtleMustCopy(indices(i));
    if (!FastBoundsCheck(index, limit)) return i;
    // Copy using memcpy if possible, otherwise an Eigen loop
    // TODO(cwhipkey): avoid linking to framework to get Allocator (to improve
    // ahead-of-time compilation binary size).
    if (is_simple_type<T>::value) {
      memcpy(out_base + i * slice_elems, params_base + index * slice_elems,
             slice_bytes);
    } else {
      out.template chip<0>(i) = params.template chip<0>(index);
    }
  }
  return -1;
}

template <typename T, typename Index>
struct GatherFunctorCPU {
  int64 operator()(typename TTypes<T>::ConstMatrix params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T>::Matrix out) {
    const int64 N = indices.size();
    const int64 slice_size = out.size() / N;
    int64 bad_i;

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      N > std::numeric_limits<int32>::max());
#define CALL(elems)                                                   \
  do {                                                                \
    if (use_large) {                                                  \
      bad_i = HandleCopies<T, Index, int64, elems>(params, indices,   \
                                                   slice_size, out);  \
    } else {                                                          \
      const int32 small_slice = static_cast<int32>(slice_size);       \
      bad_i = HandleCopies<T, Index, int32, elems>(params, indices,   \
                                                   small_slice, out); \
    }                                                                 \
  } while (0)

    if (slice_size == 10)
      CALL(10);
    else if (slice_size == 20)
      CALL(20);
    else
      CALL(-1);
#undef CALL

    return bad_i;
  }
};

template <typename Device, typename T, typename Index>
struct GatherFunctor {
  int64 operator()(const Device& d, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T>::Matrix out);
};

template <typename T, typename Index>
struct GatherFunctor<CPUDevice, T, Index> {
  int64 operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T>::Matrix out) {
    return GatherFunctorCPU<T, Index>()(params, indices, out);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_GATHER_FUNCTOR_H_
