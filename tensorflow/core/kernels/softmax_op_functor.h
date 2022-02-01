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

#ifndef TENSORFLOW_CORE_KERNELS_SOFTMAX_OP_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_SOFTMAX_OP_FUNCTOR_H_
// Functor definition for SoftmaxOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by SoftmaxOp to do the computations.
template <typename Device, typename T>
struct SoftmaxFunctor {
  // Computes Softmax or LogSoftmax activation.
  //
  // logits: dim: batch_size, num_classes.
  // softmax: dims: batch_size, num_classes.
  // log: boolean
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax, const bool log);
};

// Eigen code implementing SoftmaxFunctor::operator() or
// LogSoftmaxFunctor::operator().
// This code works for both CPU and GPU and is used by the functor
// specializations for both device types.
template <typename Device, typename T>
struct SoftmaxEigenImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstMatrix logits,
                      typename TTypes<T>::Matrix softmax, const bool log) {
    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);

    // shifted_logits = logits - max(logits along classes);
    auto shifted_logits = (logits - logits.maximum(along_class)
                                        .eval()
                                        .reshape(batch_by_one)
                                        .broadcast(one_by_class));
    if (log) {
      // Calculate the log of the softmax
      // softmax = logits - max(logits along classes);
      softmax.device(d) = shifted_logits;
      // softmax = softmax - log(sum(exp(softmax along classes)));
      softmax.device(d) = (softmax - softmax.exp()
                                         .sum(along_class)
                                         .log()
                                         .eval()
                                         .reshape(batch_by_one)
                                         .broadcast(one_by_class));
    } else {
      // NOTE(touts): If you modify this implementation please run
      // the BM_ImageNetSoftmaxFwd benchmark in nn_ops_test.cc.
      //
      // softmax = exp(logits - max(logits along classes));
      softmax.device(d) = shifted_logits.exp();
      // softmax = softmax * (1 / sum(softmax along classes));
      softmax.device(d) = (softmax * softmax.sum(along_class)
                                         .inverse()
                                         .eval()
                                         .reshape(batch_by_one)
                                         .broadcast(one_by_class));
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SOFTMAX_OP_FUNCTOR_H_
