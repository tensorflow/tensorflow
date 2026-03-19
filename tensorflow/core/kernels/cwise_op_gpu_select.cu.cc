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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"

namespace tensorflow {
namespace functor {

// Used by bfloat16 even when MLIR_GENERATED_GPU_KERNELS_ENABLED are enabled
// #if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
template <typename T, int NDIMS>
struct BCastSelectFunctor<GPUDevice, T, NDIMS> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T, NDIMS>::Tensor output_tensor,
                  typename TTypes<bool, NDIMS>::ConstTensor cond_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor then_tensor,
                  typename TTypes<T, NDIMS>::ConstTensor else_tensor,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> cond_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> then_bcast,
                  typename Eigen::array<Eigen::DenseIndex, NDIMS> else_bcast) {
    output_tensor.device(d) = cond_tensor.broadcast(cond_bcast)
                                  .select(then_tensor.broadcast(then_bcast),
                                          else_tensor.broadcast(else_bcast));
  }
};
// #endif

template <typename T>
struct SelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32, auto cond_flat32, auto then_flat32, auto else_flat32) {
          out32.device(d) = cond_flat32.select(then_flat32, else_flat32);
        },
        out, cond_flat, then_flat, else_flat);
  }
};

template <typename T>
struct SelectScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    Eigen::IndexList<Eigen::type2index<1> > rank1;
    const int size = then_flat.dimension(0);
    Eigen::array<int, 1> broadcast_dims{size};

    MaybeWith32BitIndexing<GPUDevice>(
        [&](auto out32) {
          out32.device(d) = cond.reshape(rank1)
                                .broadcast(broadcast_dims)
                                .select(then_flat, else_flat);
        },
        out);
  }
};

template <typename T>
struct BatchSelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T>::Matrix output_flat_outer_dims,
                  TTypes<bool>::ConstVec cond_vec,
                  typename TTypes<T>::ConstMatrix then_flat_outer_dims,
                  typename TTypes<T>::ConstMatrix else_flat_outer_dims) {
    const int batch = cond_vec.size();
    const int all_but_batch = then_flat_outer_dims.dimension(1);

    Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<int, Eigen::type2index<1> > reshape_dims;
    reshape_dims.set(0, batch);

    // TODO(ebrevdo): Figure out why this leads to erroneous memory access.
    //
    // To32Bit(output_flat_outer_dims).device(d) =
    //     To32Bit(cond_vec)
    //         .reshape(reshape_dims)
    //         .broadcast(broadcast_dims)
    //         .select(To32Bit(then_flat_outer_dims),
    //                 To32Bit(else_flat_outer_dims));
    output_flat_outer_dims.device(d) =
        cond_vec.reshape(reshape_dims)
            .broadcast(broadcast_dims)
            .select(then_flat_outer_dims, else_flat_outer_dims);
  }
};

#define SELECT_FUNCTOR(T)                            \
  template struct SelectFunctor<GPUDevice, T>;       \
  template struct SelectScalarFunctor<GPUDevice, T>; \
  template struct BatchSelectFunctor<GPUDevice, T>;

#define SELECT_AND_BCAST_SELECT_FUNCTOR(T)             \
  template struct BCastSelectFunctor<GPUDevice, T, 1>; \
  template struct BCastSelectFunctor<GPUDevice, T, 2>; \
  template struct BCastSelectFunctor<GPUDevice, T, 3>; \
  template struct BCastSelectFunctor<GPUDevice, T, 4>; \
  template struct BCastSelectFunctor<GPUDevice, T, 5>; \
  template struct BCastSelectFunctor<GPUDevice, T, 6>; \
  template struct BCastSelectFunctor<GPUDevice, T, 7>; \
  template struct BCastSelectFunctor<GPUDevice, T, 8>; \
  SELECT_FUNCTOR(T)

#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
SELECT_FUNCTOR(bool);
SELECT_FUNCTOR(Eigen::half);
SELECT_FUNCTOR(float);
SELECT_FUNCTOR(double);
SELECT_FUNCTOR(int32);
SELECT_FUNCTOR(int64);
SELECT_FUNCTOR(complex64);
SELECT_FUNCTOR(complex128);
#else
SELECT_AND_BCAST_SELECT_FUNCTOR(bool);
SELECT_AND_BCAST_SELECT_FUNCTOR(Eigen::half);
SELECT_AND_BCAST_SELECT_FUNCTOR(float);
SELECT_AND_BCAST_SELECT_FUNCTOR(double);
SELECT_AND_BCAST_SELECT_FUNCTOR(int32);
SELECT_AND_BCAST_SELECT_FUNCTOR(int64);
SELECT_AND_BCAST_SELECT_FUNCTOR(complex64);
SELECT_AND_BCAST_SELECT_FUNCTOR(complex128);
#endif
SELECT_AND_BCAST_SELECT_FUNCTOR(bfloat16);

#undef SELECT_FUNCTOR

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
