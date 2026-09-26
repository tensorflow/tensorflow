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

#ifndef TENSORFLOW_CORE_KERNELS_XENT_OP_H_
#define TENSORFLOW_CORE_KERNELS_XENT_OP_H_
// Functor definition for XentOp, must be compilable by nvcc.

#include <type_traits>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by XentOp to do the computations.
template <typename Device, typename T>
struct XentFunctor {
  // Computes Cross Entropy loss and backprop.
  //
  // logits: batch_size, num_classes.
  // labels: batch_size, num_classes.
  // scratch: temporary tensor, dims: batch_size, 1
  // loss: output tensor for the loss, dims: batch_size.
  // backprop: output tensor for the backprop, dims: batch_size, num_classes.
  // tail_scratch: per-row temporary for double, null for other types.
  void operator()(const Device &d,
                  const Eigen::DSizes<Eigen::DenseIndex, 2> &shape,
                  const Eigen::array<Eigen::DenseIndex, 2> &logits_bcast,
                  const Eigen::array<Eigen::DenseIndex, 2> &labels_bcast,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop,
                  T* tail_scratch);
};

// Eigen code implementing XentFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename T>
struct XentEigenImpl {
  static void Compute(const Device &d,
                      const Eigen::DSizes<Eigen::DenseIndex, 2> &shape,
                      const Eigen::array<Eigen::DenseIndex, 2> &logits_bcast,
                      const Eigen::array<Eigen::DenseIndex, 2> &labels_bcast,
                      typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<T>::ConstMatrix labels,
                      typename TTypes<T>::Matrix scratch,
                      typename TTypes<T>::Vec loss,
                      typename TTypes<T>::Matrix backprop,
                      T* tail_scratch) {
    // NOTE(touts): This duplicates some of the computations in softmax_op
    // because we need the intermediate (logits -max(logits)) values to
    // avoid a log(exp()) in the computation of the loss.

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = shape[kBatchDim];
    const int num_classes = shape[kClassDim];

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<int> batch_only;
    batch_only.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);

    // max_logits along classes.
    scratch.reshape(batch_only).device(d) =
        logits.broadcast(logits_bcast).maximum(along_class);

    // logits - max_logits.
    backprop.device(d) =
        logits.broadcast(logits_bcast) - scratch.broadcast(one_by_class);

    // sum(exp(logits - max_logits)) along classes.
    scratch.reshape(batch_only).device(d) = backprop.exp().sum(along_class);

    // NOTE(keveman): Eigen on GPU dispatches to an optimized implementation
    // for an expression of the form lhs = rhs.sum().
    // lhs = -rhs.sum() doesn't match the above pattern, so folding in the
    // negation before calling sum().
    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    loss.device(d) = (labels.broadcast(labels_bcast) *
                      (scratch.log().eval().broadcast(one_by_class) - backprop))
                         .eval()
                         .sum(along_class);

    // When a float64 probability rounds to one, subtracting a unit label
    // erases its tail gradient. Sum the non-maximal terms separately so the
    // complement remains representable. Other labels retain prob - labels,
    // including unnormalized and zero-masked distributions.
    if constexpr (std::is_same_v<T, double>) {
      typename TTypes<T>::Vec tail(tail_scratch, batch_size);
      tail.device(d) =
          (backprop == backprop.constant(T(0)))
              .select(backprop.constant(T(0)), backprop.exp())
              .sum(along_class);
      const auto labels_broadcast = labels.broadcast(labels_bcast);
      const auto denominator = scratch.broadcast(one_by_class);
      const auto tail_broadcast =
          tail.reshape(batch_by_one).broadcast(one_by_class);
      const auto dominant_unit_label =
          (backprop == backprop.constant(T(0))) &&
          (denominator < denominator.constant(T(2))) &&
          (labels_broadcast == labels_broadcast.constant(T(1)));
      backprop.device(d) = dominant_unit_label.select(
          -tail_broadcast / (tail_broadcast + T(1)),
          backprop.exp() / denominator - labels_broadcast);
    } else {
      // backprop: prob - labels, where
      //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
      backprop.device(d) = (backprop.exp() / scratch.broadcast(one_by_class)) -
                           labels.broadcast(labels_bcast);
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_XENT_OP_H_
