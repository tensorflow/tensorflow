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

// Reads a full row only when its softmax denominator rounded to one.
template <typename T>
class XentTailGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE XentTailGenerator(
      typename TTypes<T>::Matrix shifted_logits,
      typename TTypes<T>::Matrix denominator, int num_classes)
      : shifted_logits_(shifted_logits),
        denominator_(denominator),
        num_classes_(num_classes) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<Eigen::DenseIndex, 1>& coords) const {
    const Eigen::DenseIndex row = coords[0];
    if (denominator_(row, 0) != T(1)) {
      return T(0);
    }
    T tail = T(0);
    for (int col = 0; col < num_classes_; ++col) {
      const T shifted = shifted_logits_(row, col);
      if (shifted != T(0)) {
        tail += Eigen::numext::exp(shifted);
      }
    }
    return tail;
  }

 private:
  typename TTypes<T>::Matrix shifted_logits_;
  typename TTypes<T>::Matrix denominator_;
  const int num_classes_;
};

// Functor used by XentOp to do the computations.
template <typename Device, typename T>
struct XentFunctor {
  // Computes Cross Entropy loss and backprop.
  //
  // logits: batch_size, num_classes.
  // labels: batch_size, num_classes.
  // scratch: temporary tensor, dims: batch_size, 1 or 2 for double.
  // loss: output tensor for the loss, dims: batch_size.
  // backprop: output tensor for the backprop, dims: batch_size, num_classes.
  void operator()(const Device &d,
                  const Eigen::DSizes<Eigen::DenseIndex, 2> &shape,
                  const Eigen::array<Eigen::DenseIndex, 2> &logits_bcast,
                  const Eigen::array<Eigen::DenseIndex, 2> &labels_bcast,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop);
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
                      typename TTypes<T>::Matrix scratch_storage,
                      typename TTypes<T>::Vec loss,
                      typename TTypes<T>::Matrix backprop) {
    // NOTE(touts): This duplicates some of the computations in softmax_op
    // because we need the intermediate (logits -max(logits)) values to
    // avoid a log(exp()) in the computation of the loss.

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = shape[kBatchDim];
    const int num_classes = shape[kClassDim];
    // The first batch_size elements hold the row maximum, then denominator.
    typename TTypes<T>::Matrix scratch(scratch_storage.data(), batch_size, 1);

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

    // When the float64 denominator rounds to one, subtracting a unit label
    // erases its tail gradient. Only those rows need a non-maximal sum.
    if constexpr (std::is_same_v<T, double>) {
      if (scratch_storage.dimension(1) == 2) {
        // The packed second half may not be aligned when batch_size is odd.
        T* tail_data = scratch_storage.data() + batch_size;
        typename TTypes<T>::UnalignedVec tail(tail_data, batch_size);
        XentTailGenerator<T> tail_gen(backprop, scratch, num_classes);
        tail.device(d) = tail.generate(tail_gen);

        const auto labels_broadcast = labels.broadcast(labels_bcast);
        const auto denominator = scratch.broadcast(one_by_class);
        const auto cancellation_rows =
            (scratch == scratch.constant(T(1))).eval();
        const auto rounded_dominant =
            cancellation_rows.broadcast(one_by_class) &&
            (backprop == backprop.constant(T(0)));
        const auto tail_broadcast =
            tail.reshape(batch_by_one).broadcast(one_by_class);
        const auto one_minus_labels =
            backprop.constant(T(1)) - labels_broadcast;
        const auto corrected =
            (tail_broadcast == tail_broadcast.constant(T(0)))
                .select(one_minus_labels,
                        one_minus_labels - tail_broadcast / denominator);
        backprop.device(d) = rounded_dominant.select(
            corrected, backprop.exp() / denominator - labels_broadcast);
        return;
      }
    }
    // Preserve the original path for non-double types and 9-argument callers
    // that provide only a single-column scratch tensor.
    backprop.device(d) = (backprop.exp() / scratch.broadcast(one_by_class)) -
                         labels.broadcast(labels_bcast);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_XENT_OP_H_
