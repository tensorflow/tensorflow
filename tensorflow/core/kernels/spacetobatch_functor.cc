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

// Specialization of SpaceToBatchFunctor for a CPUDevice.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/spacetobatch_functor.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

namespace {

// Implementation of nested loops for SpaceToBatchOpFunctor.
//
// To simplify template implementation given lack of constexpr if, both the
// input and output pointers are non-const.
template <int N, bool B2S>
struct SpaceToBatchHelper {
  template <typename T>
  static void run(T* space_tensor_ptr, const int64* space_tensor_shape,
                  const int64* space_tensor_strides, const int64* block_shape,
                  const int64* pad_start, const int64* block_offsets,
                  const int64* batch_tensor_shape,
                  const int64* batch_tensor_strides, T* batch_tensor_ptr) {
    for (int64 batch_tensor_pos = 0; batch_tensor_pos < batch_tensor_shape[0];
         ++batch_tensor_pos) {
      const int64 space_tensor_pos =
          batch_tensor_pos * block_shape[0] + block_offsets[0] - pad_start[0];
      if (space_tensor_pos >= 0 && space_tensor_pos < space_tensor_shape[0]) {
        SpaceToBatchHelper<N - 1, B2S>::run(
            space_tensor_ptr + space_tensor_pos * space_tensor_strides[0],
            space_tensor_shape + 1, space_tensor_strides + 1, block_shape + 1,
            pad_start + 1, block_offsets + 1, batch_tensor_shape + 1,
            batch_tensor_strides + 1, batch_tensor_ptr);
      } else {
        if (B2S == false) {
          // Copy in padding.
          for (int64 i = 0; i < batch_tensor_strides[0]; ++i) {
            batch_tensor_ptr[i] = static_cast<T>(0);
          }
        }
      }
      batch_tensor_ptr += batch_tensor_strides[0];
    }
  }
};

template <bool B2S>
struct SpaceToBatchHelper<0, B2S> {
  template <typename T>
  static void run(T* space_tensor_ptr, const int64* space_tensor_shape,
                  const int64* space_tensor_strides, const int64* block_shape,
                  const int64* pad_start, const int64* block_offsets,
                  const int64* batch_tensor_shape,
                  const int64* batch_tensor_strides, T* batch_tensor_ptr) {
    for (int64 i = 0; i < batch_tensor_strides[-1]; ++i) {
      if (B2S == false) {
        batch_tensor_ptr[i] = space_tensor_ptr[i];
      } else {
        space_tensor_ptr[i] = batch_tensor_ptr[i];
      }
    }
  }
};

}  // namespace

template <typename T, int NUM_BLOCK_DIMS, bool B2S>
struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, B2S> {
  using SpaceT = typename std::conditional<B2S, T, const T>::type;
  using BatchT = typename std::conditional<B2S, const T, T>::type;
  Status operator()(
      const CPUDevice& d,
      typename TTypes<SpaceT, NUM_BLOCK_DIMS + 2>::Tensor space_tensor,
      const int64 block_shape_tensor[NUM_BLOCK_DIMS],
      const int64 paddings_tensor[NUM_BLOCK_DIMS * 2],
      typename TTypes<BatchT, NUM_BLOCK_DIMS + 2>::Tensor batch_tensor) {
    const int64 batch_tensor_batch = batch_tensor.dimension(0);

    const int64 space_tensor_batch = space_tensor.dimension(0);

    // Copy into local array so that the compiler is free to place in a
    // register.
    int64 pad_start[NUM_BLOCK_DIMS];
    int64 block_shape[NUM_BLOCK_DIMS];
    int64 space_tensor_shape[NUM_BLOCK_DIMS],
        batch_tensor_shape[NUM_BLOCK_DIMS];
    for (int block_dim = 0; block_dim < NUM_BLOCK_DIMS; ++block_dim) {
      pad_start[block_dim] = paddings_tensor[block_dim * 2];
      block_shape[block_dim] = block_shape_tensor[block_dim];
      space_tensor_shape[block_dim] = space_tensor.dimension(block_dim + 1);
      batch_tensor_shape[block_dim] = batch_tensor.dimension(block_dim + 1);
    }

    int64 space_tensor_strides[NUM_BLOCK_DIMS + 2],
        batch_tensor_strides[NUM_BLOCK_DIMS + 2];
    space_tensor_strides[NUM_BLOCK_DIMS + 1] =
        batch_tensor_strides[NUM_BLOCK_DIMS + 1] = 1;
    for (int dim = NUM_BLOCK_DIMS; dim >= 0; --dim) {
      space_tensor_strides[dim] =
          space_tensor_strides[dim + 1] * space_tensor.dimension(dim + 1);
      batch_tensor_strides[dim] =
          batch_tensor_strides[dim + 1] * batch_tensor.dimension(dim + 1);
    }

    // Use non-const pointers for both input and output to simplify template
    // implementation given lack of constexpr if.
    T* space_tensor_ptr = const_cast<T*>(space_tensor.data());
    T* batch_tensor_ptr = const_cast<T*>(batch_tensor.data());

    for (int64 batch_tensor_b = 0; batch_tensor_b < batch_tensor_batch;
         ++batch_tensor_b) {
      const int64 space_tensor_b = batch_tensor_b % space_tensor_batch;
      int64 block_index = batch_tensor_b / space_tensor_batch;
      int64 block_offsets[NUM_BLOCK_DIMS];
      for (int block_dim = NUM_BLOCK_DIMS - 1; block_dim >= 0; --block_dim) {
        // Skip unnecessary remainder operation for block_dim == 0.
        block_offsets[block_dim] =
            block_dim > 0 ? block_index % block_shape[block_dim] : block_index;
        block_index /= block_shape[block_dim];
      }

      // The compiler should inline the nested loops generated by this template.
      SpaceToBatchHelper<NUM_BLOCK_DIMS, B2S>::run(
          space_tensor_ptr + space_tensor_b * space_tensor_strides[0],
          space_tensor_shape, &space_tensor_strides[1], block_shape, pad_start,
          block_offsets, batch_tensor_shape, &batch_tensor_strides[1],
          batch_tensor_ptr + batch_tensor_b * batch_tensor_strides[0]);
    }
    return Status::OK();
  }
};

// Instantiate.
#define INSTANTIATE(NUM_BLOCK_DIMS, T)                                      \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, false>; \
  template struct SpaceToBatchFunctor<CPUDevice, T, NUM_BLOCK_DIMS, true>;  \
  /**/

#define INSTANTIATE_FOR_T(T) \
  TF_SPACETOBATCH_FOR_EACH_NUM_BLOCK_DIMS(INSTANTIATE, T)

TF_CALL_REAL_NUMBER_TYPES(INSTANTIATE_FOR_T)

#undef INSTANTIATE_FOR_T
#undef INSTANTIATE

}  // namespace functor
}  // end namespace tensorflow
