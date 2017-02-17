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

#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cwise_ops_gpu_common.cu.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename T>
struct SelectFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstFlat cond_flat,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {
    To32Bit(out).device(d) =
        To32Bit(cond_flat).select(To32Bit(then_flat), To32Bit(else_flat));
  }
};

template <typename T>
struct SelectScalarFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<bool>::ConstScalar cond,
                  typename TTypes<T>::ConstFlat then_flat,
                  typename TTypes<T>::ConstFlat else_flat) {

#if !defined(EIGEN_HAS_INDEX_LIST)
  Eigen::array<int, 1> rank1{1};
#else
  Eigen::IndexList<Eigen::type2index<1>> rank1;
#endif
  const int size  = then_flat.dimension(0);
  Eigen::array<int, 1> broadcast_dims{size};

  To32Bit(out).device(d) = cond.reshape(rank1)
                               .broadcast(broadcast_dims)
                               .select(then_flat, else_flat);

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

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 2> broadcast_dims{{ 1, all_but_batch }};
    Eigen::Tensor<int, 2>::Dimensions reshape_dims{{ batch, 1 }};
#else
    Eigen::IndexList<Eigen::type2index<1>, int> broadcast_dims;
    broadcast_dims.set(1, all_but_batch);
    Eigen::IndexList<int, Eigen::type2index<1> > reshape_dims;
    reshape_dims.set(0, batch);
#endif

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

#define SELECT_FUNCTOR(T)                      \
  template struct SelectFunctor<GPUDevice, T>; \
  template struct SelectScalarFunctor<GPUDevice, T>; \
  template struct BatchSelectFunctor<GPUDevice, T>;

SELECT_FUNCTOR(Eigen::half);
SELECT_FUNCTOR(float);
SELECT_FUNCTOR(double);
SELECT_FUNCTOR(int32);
SELECT_FUNCTOR(int64);
SELECT_FUNCTOR(complex64);
SELECT_FUNCTOR(complex128);

#undef SELECT_FUNCTOR

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
