#ifndef TENSORFLOW_KERNELS_XENT_OP_H_
#define TENSORFLOW_KERNELS_XENT_OP_H_
// Functor definition for XentOp, must be compilable by nvcc.

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

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
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
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
  static void Compute(const Device& d, typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<T>::ConstMatrix labels,
                      typename TTypes<T>::Matrix scratch,
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

    // max_logits along classes.
    scratch.reshape(batch_only).device(d) = logits.maximum(along_class);

    // logits - max_logits.
    backprop.device(d) = logits - scratch.broadcast(one_by_class);

    // sum(exp(logits - max_logits)) along classes.
    scratch.reshape(batch_only).device(d) = backprop.exp().sum(along_class);

    // NOTE(keveman): Eigen on GPU dispatches to an optimized implementaion
    // for an expression of the form lhs = rhs.sum().
    // lhs = -rhs.sum() doesn't match the above pattern, so folding in the
    // negation before calling sum().
    //  sum(-labels *
    //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
    //  along classes
    loss.device(d) =
        (labels * (scratch.log().eval().broadcast(one_by_class) - backprop))
            .eval()
            .sum(along_class);

    // backprop: prob - labels, where
    //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
    backprop.device(d) =
        (backprop.exp() / scratch.broadcast(one_by_class)) - labels;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_XENT_OP_H_
