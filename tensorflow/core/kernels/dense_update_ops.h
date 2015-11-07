#ifndef TENSORFLOW_KERNELS_DENSE_UPDATE_OPS_H_
#define TENSORFLOW_KERNELS_DENSE_UPDATE_OPS_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

enum DenseUpdateType { ADD, SUB, ASSIGN };

namespace functor {

template <typename Device, typename T, DenseUpdateType OP>
struct DenseUpdate;

template <typename Device, typename T>
struct DenseUpdate<Device, T, ADD> {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename Device, typename T>
struct DenseUpdate<Device, T, SUB> {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

template <typename Device, typename T>
struct DenseUpdate<Device, T, ASSIGN> {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DENSE_UPDATE_OPS_H_
