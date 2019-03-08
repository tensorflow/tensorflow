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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_XENT_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_XENT_OP_H_
// Functor definition for SparseXentOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace sparse_xent_helpers {

template <typename T>
typename TTypes<const T, 1>::Tensor32Bit To32BitConst(
    typename TTypes<T>::Vec in) {
  return To32Bit(typename TTypes<T>::ConstVec(in.data(), in.dimensions()));
}

template <typename T>
typename TTypes<const T, 2>::Tensor32Bit To32BitConst(
    typename TTypes<T>::Matrix in) {
  return To32Bit(typename TTypes<T>::ConstMatrix(in.data(), in.dimensions()));
}

}  // namespace sparse_xent_helpers

namespace generator {

// Generator for calculation of the sparse Xent loss.
// This generator takes the logits, the sum of the exponentiated
// logits, and the label indices.  For each minibatch entry, ignoring
// the batch index b, it calculates:
//
//   loss[j] = (log(sum_exp_logits) - logits[j]) * 1{ j == label }
//
// for j = 0 .. num_classes.  This value must be summed over all j for
// the final loss.
template <typename T, typename Index>
class SparseXentLossGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseXentLossGenerator(
      typename TTypes<const T, 2>::Tensor32Bit logits,
      typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits,
      typename TTypes<const Index, 1>::Tensor32Bit labels,
      const Index max_depth)
      : logits_(logits),
        sum_exp_logits_(sum_exp_logits),
        labels_(labels),
        max_depth_(max_depth) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& coords) const {
    const int batch = coords[0];
    const int depth = coords[1];
    const Index label = tensorflow::internal::SubtleMustCopy(labels_(batch));
    if (!FastBoundsCheck(label, max_depth_)) {
      return Eigen::NumTraits<T>::quiet_NaN();
    }
    return TF_PREDICT_FALSE(label == depth)
               ? (Eigen::numext::log(sum_exp_logits_(batch)) - logits_(coords))
               : T(0.0);
  };

 private:
  typename TTypes<const T, 2>::Tensor32Bit logits_;
  typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits_;
  typename TTypes<const Index, 1>::Tensor32Bit labels_;
  const Index max_depth_;
};

// Generator for calculation of the sparse Xent gradient.
// This generator takes the exponentiated logits, their sums, and the label
// indices. For each minibatch entry, ignoring the batch index b, it calculates:
//
//   exp_logits[j] / sum_exp_logits - 1{ j == label }
//
// for j = 0 .. num_classes.
template <typename T, typename Index>
class SparseXentGradGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseXentGradGenerator(
      typename TTypes<const T, 2>::Tensor32Bit exp_logits,
      typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits,
      typename TTypes<const Index, 1>::Tensor32Bit labels,
      const Index max_depth)
      : exp_logits_(exp_logits),
        sum_exp_logits_(sum_exp_logits),
        labels_(labels),
        max_depth_(max_depth) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& coords) const {
    const int batch = coords[0];
    const int depth = coords[1];
    const Index label = tensorflow::internal::SubtleMustCopy(labels_(batch));
    if (!FastBoundsCheck(label, max_depth_)) {
      return Eigen::NumTraits<T>::quiet_NaN();
    }
    T subtract = TF_PREDICT_FALSE(depth == label) ? T(1.0) : T(0.0);
    return exp_logits_(coords) / sum_exp_logits_(batch) - subtract;
  };

 private:
  typename TTypes<const T, 2>::Tensor32Bit exp_logits_;
  typename TTypes<const T, 1>::Tensor32Bit sum_exp_logits_;
  typename TTypes<const Index, 1>::Tensor32Bit labels_;
  const Index max_depth_;
};

}  // namespace generator

namespace functor {

template <typename Device, typename T>
struct RowMaxReduction {
  // Computes the maximum across the rows of logits
  //
  // logits: batch_size, num_classes.
  // maximum: temporary tensor, dims: batch_size, 1
  static inline void Compute(OpKernelContext* ctx,
                             typename TTypes<T>::ConstMatrix logits,
                             typename TTypes<T>::Vec maximum) {
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> along_row;
    along_row[0] = 1;
#else
    Eigen::IndexList<Eigen::type2index<1> > along_row;
#endif
    Device d = ctx->eigen_device<Device>();
    To32Bit(maximum).device(d) = To32Bit(logits).maximum(along_row);
  }
};

// Functor used by SparseXentOp to do the computations.
template <typename Device, typename T, typename Index>
struct SparseXentFunctor {
  // Computes Cross Entropy loss and backprop.
  //
  // logits: batch_size, num_classes.
  // labels: num_classes.
  // scratch: temporary tensor, dims: batch_size, 1
  // loss: output tensor for the loss, dims: batch_size.
  // backprop: output tensor for the backprop, dims: batch_size, num_classes.
  void operator()(OpKernelContext* ctx, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop);
};

// Eigen code implementing SparseXentFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename T, typename Index>
struct SparseXentEigenImpl {
  static void Compute(OpKernelContext* ctx,
                      typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<Index>::ConstVec labels,
                      typename TTypes<T>::Vec scratch,
                      typename TTypes<T>::Vec loss,
                      typename TTypes<T>::Matrix backprop) {
    // NOTE(touts): This duplicates some of the computations in softmax_op
    // because we need the intermediate (logits -max(logits)) values to
    // avoid a log(exp()) in the computation of the loss.

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::array<int, 1> along_class;
    along_class[0] = kClassDim;
    Eigen::array<int, 1> batch_only;
    batch_only[0] = batch_size;
    Eigen::array<int, 2> batch_by_one;
    batch_by_one[0] = batch_size;
    batch_by_one[1] = 1;
    Eigen::array<int, 2> one_by_class;
    one_by_class[0] = 1;
    one_by_class[1] = num_classes;
#else
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<int> batch_only;
    batch_only.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);
#endif

    // scratch = max_logits along classes.
    RowMaxReduction<Device, T>::Compute(ctx, logits, scratch);

    Device d = ctx->eigen_device<Device>();
    // backprop = logits - max_logits.
    To32Bit(backprop).device(d) =
        To32Bit(logits) -
        To32Bit(scratch).reshape(batch_by_one).broadcast(one_by_class);

    // scratch = sum(exp(logits - max_logits)) along classes.
    To32Bit(scratch).device(d) = To32Bit(backprop).exp().sum(along_class);

    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    generator::SparseXentLossGenerator<T, Index> sparse_xent_loss_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(loss).device(d) =
        To32Bit(backprop).generate(sparse_xent_loss_gen).sum(along_class);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    To32Bit(backprop).device(d) = To32Bit(backprop).exp();
    generator::SparseXentGradGenerator<T, Index> sparse_xent_grad_gen(
        sparse_xent_helpers::To32BitConst<T>(backprop),
        sparse_xent_helpers::To32BitConst<T>(scratch), To32Bit(labels),
        backprop.dimension(1) /* max_depth */);
    To32Bit(backprop).device(d) =
        To32Bit(backprop).generate(sparse_xent_grad_gen);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_XENT_OP_H_
