/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_TRAINING_OPS_H_
#define TENSORFLOW_CORE_KERNELS_TRAINING_OPS_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

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
struct ApplyAdadelta {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T, typename Tindex>
struct SparseApplyAdadelta {
  void operator()(const Device& d, typename TTypes<T>::Matrix var,
                  typename TTypes<T>::Matrix accum,
                  typename TTypes<T>::Matrix accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstMatrix grad,
                  typename TTypes<Tindex>::ConstFlat indices);
};

template <typename Device, typename T>
struct FobosElasticNet {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyProximalGradientDescent {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyAdagrad {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad, bool update_slots);
};

template <typename Device, typename T>
struct ApplyAdagradV2 {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool update_slots);
};

template <typename Device, typename T>
struct ApplyAdagradDA {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat gradient_accum,
                  typename TTypes<T>::Flat gradient_squared_accum,
                  typename TTypes<T>::ConstScalar lr, int64_t global_step,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T, typename Tindex, bool has_epsilon>
struct SparseApplyAdagrad {
  // Note that epsilon is ignored if has_epsilon is false.
  absl::Status operator()(const Device& d, typename TTypes<T>::Matrix var,
                          typename TTypes<T>::Matrix accum,
                          typename TTypes<T>::ConstScalar lr,
                          typename TTypes<T>::ConstScalar epsilon,
                          typename TTypes<T>::ConstMatrix grad,
                          typename TTypes<Tindex>::ConstVec indices,
                          int64_t inner_dim, bool update_slots);
};

template <typename Device, typename T>
struct ApplyProximalAdagrad {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T, typename Tindex>
struct SparseApplyProximalAdagrad {
  absl::Status operator()(const Device& d, typename TTypes<T>::Matrix var,
                          typename TTypes<T>::Matrix accum,
                          typename TTypes<T>::ConstScalar lr,
                          typename TTypes<T>::ConstScalar l1,
                          typename TTypes<T>::ConstScalar l2,
                          typename TTypes<T>::ConstMatrix grad,
                          typename TTypes<Tindex>::ConstVec indices,
                          int64_t inner_dim);
};

template <typename Device, typename T>
struct ApplyFtrl {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power);
};

template <typename Device, typename T>
struct ApplyFtrlMultiplyLinearByLr {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power);
};

template <typename Device, typename T>
struct ApplyFtrlV2 {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power);
};

template <typename Device, typename T>
struct ApplyFtrlV2MultiplyLinearByLr {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power);
};

template <typename Device, typename T, typename Tindex, bool has_l2_shrinkage>
struct SparseApplyFtrl {
  absl::Status operator()(const Device& d, typename TTypes<T>::Matrix var_flat,
                          typename TTypes<T>::Matrix accum_flat,
                          typename TTypes<T>::Matrix linear_flat,
                          typename TTypes<T>::ConstScalar lr,
                          typename TTypes<T>::ConstScalar l1,
                          typename TTypes<T>::ConstScalar l2,
                          typename TTypes<T>::ConstScalar l2_shrinkage,
                          typename TTypes<T>::ConstScalar lr_power,
                          typename TTypes<T>::ConstMatrix grad_flat,
                          typename TTypes<Tindex>::ConstVec indices_vec,
                          int64_t inner_dim, bool multiply_linear_by_lr);
};

template <typename Device, typename T>
struct ApplyMomentum {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov);
};

template <typename Device, typename T>
struct ApplyKerasMomentum {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov);
};

template <typename Device, typename T, typename Tindex>
struct SparseApplyKerasMomentum {
  Tindex operator()(const Device& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstFlat indices,
                    typename TTypes<T>::ConstScalar momentum,
                    bool use_nesterov);
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
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov);
};

template <typename Device, typename T>
struct ApplyAdamWithAmsgrad {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat vhat,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyAdaMax {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
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

template <typename Device, typename T>
struct ApplyCenteredRMSProp {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,
                  typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyAddSign {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad);
};

template <typename Device, typename T>
struct ApplyPowerSign {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar logbase,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad);
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAINING_OPS_H_
