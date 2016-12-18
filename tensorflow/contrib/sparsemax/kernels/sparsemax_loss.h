/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_SPARSEMAX_LOSS_OP_H_
#define TENSORFLOW_KERNELS_SPARSEMAX_LOSS_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SparsemaxLoss {
  void operator()(const Device& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix sparsemax,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Vec losses) {
    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
#else
    Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
    Eigen::IndexList<Eigen::type2index<1> > depth_dim;
    Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);
#endif

    T zero = static_cast<T>(0);
    T half = static_cast<T>(0.5);

    // shifted_logits = logits - max(logits along classes);
    auto shifted_logits = (logits -
                           logits.mean(along_class)
                               .eval()
                               .reshape(batch_by_one)
                               .broadcast(one_by_class));

    // sum over support
    auto support = (sparsemax > zero).template cast<T>();
    auto sum_s = support * sparsemax * (shifted_logits - half * sparsemax);

    // - z_k + ||q||^2
    auto q_part = labels * (half * labels - shifted_logits);

    losses.device(d) = (sum_s + q_part)
      .sum(along_class)
      .eval();
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SPARSEMAX_LOSS_OP_H_
