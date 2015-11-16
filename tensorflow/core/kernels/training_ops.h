#ifndef TENSORFLOW_KERNELS_TRAINING_OPS_H_
#define TENSORFLOW_KERNELS_TRAINING_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Each training algorithm has a ApplyXYZ functor struct declared in
// this header file. They are specialized for different devices
// (CPUDevice in training_ops.cc or GPUDevice in training_ops_gpu.cc).

template <typename Device, typename T>
struct ApplyGradientDescent {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::ConstFlat delta);
};

template <typename Device, typename T>
struct ApplyAdagrad {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyMomentum {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum);
};

template <typename Device, typename T>
struct ApplyAdam {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyRMSProp {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TRAINING_OPS_H_
