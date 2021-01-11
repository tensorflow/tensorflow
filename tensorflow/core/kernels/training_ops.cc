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

#define EIGEN_USE_THREADS
#include "tensorflow/core/kernels/training_ops.h"

#include <algorithm>  // NOLINT

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using Index = Eigen::Index;

namespace {
template <class T>
inline T sgn(const T x) {
  T zero(0);
  T one(1);
  return (x == zero ? zero : (x < zero ? -one : one));
}
}  // namespace

namespace functor {
template <typename T>
struct ApplyGradientDescent<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    var.device(d) -= grad * lr();
  }
};

template <typename T>
struct ApplyAdadelta<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    accum.device(d) =
        accum * rho() + grad.square() * (static_cast<T>(1) - rho());
    const auto update =
        (accum_update + epsilon()).sqrt() * (accum + epsilon()).rsqrt() * grad;
    var.device(d) -= update * lr();
    accum_update.device(d) =
        accum_update * rho() + update.square() * (static_cast<T>(1) - rho());
  }
};

template <typename T>
struct ApplyProximalGradientDescent<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    // Note that here is Fobos update, for details please refer:
    // http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf
    // TODO(xbing): merge the logic for ProximalGradientDescent and
    // ProximalAdagrad.
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(d) -= grad * lr();
    if (l1() > 0) {
      // compute sign(v) * max(|v| - lr * l1, 0)
      var.device(d) =
          prox_var.sign() *
          (prox_var.abs() - var.constant(lr() * l1())).cwiseMax(T(0.0)) /
          (var.constant(1.0) + var.constant(l2() * lr()));
    } else {
      var.device(d) =
          prox_var / (var.constant(1.0) + var.constant(l2() * lr()));
    }
  }
};

template <typename T>
struct ApplyAdagradDA<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat gradient_accum,
                  typename TTypes<T>::Flat gradient_squared_accum,
                  typename TTypes<T>::ConstScalar lr, int64 global_step,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    // Accumulate gradient, and gradient_squared
    gradient_accum.device(d) += grad;
    gradient_squared_accum.device(d) += grad.square();

    // AdagradDA update:
    // Let g to be gradient accumulator, gg to be gradient squared accumulator,
    // T be the global step, lr is the learning rate, and k the initial
    // gradient squared accumulator value.
    // w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
    if (l1() > 0) {
      var.device(d) =
          lr() * var.constant(-1.0) * gradient_accum.sign() *
          (gradient_accum.abs() -
           var.constant(static_cast<float>(global_step)) * var.constant(l1()))
              .cwiseMax(T(0.0)) /
          (var.constant(l2()) *
               var.constant(static_cast<float>(global_step) * lr()) +
           gradient_squared_accum.sqrt());
    } else {
      var.device(d) =
          lr() * gradient_accum * var.constant(-1.0) /
          (var.constant(l2()) *
               var.constant(static_cast<float>(global_step) * lr()) +
           gradient_squared_accum.sqrt());
    }
  }
};

template <typename T>
struct ApplyAdagrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    var.device(d) -= grad * lr() * accum.rsqrt();
  }
};

template <typename T>
struct ApplyAdagradV2<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    var.device(d) -= grad * lr() / (accum.sqrt() + epsilon());
  }
};

template <typename T, typename Tindex, bool has_epsilon>
struct SparseApplyAdagrad<CPUDevice, T, Tindex, has_epsilon> {
  Status operator()(const CPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar epsilon,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                    bool update_slots) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();
    const Tindex first_dim_size = static_cast<Tindex>(var.dimension(0));
    const T lr_scalar = lr();
    const int in_bytes = inner_dim * sizeof(T) * 3;
    const int out_bytes = inner_dim * sizeof(T) * 2;
    const int cycles = inner_dim * (Eigen::TensorOpCost::AddCost<T>() * 2 +
                                    Eigen::TensorOpCost::MulCost<T>() * 2);
    const Eigen::TensorOpCost cost(in_bytes, out_bytes, cycles);

    if (inner_dim > 1) {
      for (Tindex i = 0; i < N; ++i) {
        const Tindex index = internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, first_dim_size)) {
          return errors::InvalidArgument(
              strings::StrCat("Index ", index, " at offset ", i,
                              " in indices is out of range"));
        }
      }

      const auto shard = [&](Tindex start_idx, Tindex end_idx) -> void {
        for (Tindex i = start_idx; i < end_idx; ++i) {
          const Tindex index = internal::SubtleMustCopy(indices(i));
          auto a = accum.template chip<0>(index);
          auto g = grad.template chip<0>(i);
          auto v = var.template chip<0>(index);
          if (update_slots) {
            a += g.square();
          }
          if (has_epsilon) {
            v -= g.constant(lr_scalar) * g / (a.sqrt() + a.constant(epsilon()));
          } else {
            v -= g.constant(lr_scalar) * g * a.rsqrt();
          }
        }
      };

      d.parallelFor(N, cost, shard);
    } else {
      for (Tindex i = 0; i < N; ++i) {
        const Tindex index = internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, first_dim_size)) {
          return errors::InvalidArgument(
              strings::StrCat("Index ", index, " at offset ", i,
                              " in indices is out of range"));
        }
      }

      const auto shard = [&](Tindex start_idx, Tindex end_idx) -> void {
        for (Tindex i = start_idx; i < end_idx; ++i) {
          const Tindex index = internal::SubtleMustCopy(indices(i));
          T& a = accum(index);
          const T& g = grad(i);
          if (update_slots) {
            a += g * g;
          }
          if (has_epsilon) {
            var(index) -= lr_scalar * g / (Eigen::numext::sqrt(a) + epsilon());
          } else {
            var(index) -= lr_scalar * g / Eigen::numext::sqrt(a);
          }
        }
      };

      d.parallelFor(N, cost, shard);
    }

    return Status::OK();
  }
};

template <typename T>
struct ApplyProximalAdagrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
    // Fobos update per paper with Adagrad learning rate.
    accum.device(d) += grad.square();
    // Adagrad learning rate.
    auto learning_rate = accum.constant(lr()) * accum.rsqrt();
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(d) -= grad * learning_rate;
    if (l1() > 0) {
      // compute sign(v) * max(|v| - lr * l1, 0)
      var.device(d) = prox_var.sign() *
                      (prox_var.abs() - learning_rate * prox_var.constant(l1()))
                          .cwiseMax(T(0.0)) /
                      (var.constant(1.0) + var.constant(l2()) * learning_rate);
    } else {
      var.device(d) =
          prox_var / (var.constant(1.0) + var.constant(l2()) * learning_rate);
    }
  }
};

template <typename T, typename Tindex>
struct SparseApplyProximalAdagrad<CPUDevice, T, Tindex> {
  Status operator()(const CPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar l1,
                    typename TTypes<T>::ConstScalar l2,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices,
                    int64 inner_dim) {
    const Tindex N = static_cast<Tindex>(indices.dimension(0));
    if (N == 0) return Status::OK();
    const Tindex first_dim_size = static_cast<Tindex>(var.dimension(0));
    const T lr_scalar = lr();
    const T l1_scalar = l1();
    const T l2_scalar = l2();
    if (inner_dim > 1) {
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, first_dim_size)) {
          return errors::InvalidArgument(
              strings::StrCat("Index ", index, " at offset ", i,
                              " in indices is out of range"));
        }
        auto a = accum.template chip<0>(index);
        auto g = grad.template chip<0>(i);
        auto v = var.template chip<0>(index);
        a += g.square();
        // compute learning_rate for current step.
        auto learning_rate = a.constant(lr_scalar) * a.rsqrt();
        auto prox_v = v;
        // v = w - g * learning_rate.
        prox_v -= g * learning_rate;
        if (l1_scalar > 0) {
          // compute sign(v) * max(|v|, 0)
          v = prox_v.sign() *
              (prox_v.abs() - learning_rate * prox_v.constant(l1_scalar))
                  .cwiseMax(static_cast<T>(0.0)) /
              (v.constant(1.0) + v.constant(l2_scalar) * learning_rate);
        } else {
          v = prox_v /
              (v.constant(1.0) + v.constant(l2_scalar) * learning_rate);
        }
      }
    } else {
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = internal::SubtleMustCopy(indices(i));
        if (!FastBoundsCheck(index, first_dim_size)) {
          return errors::InvalidArgument(
              strings::StrCat("Index ", index, " at offset ", i,
                              " in indices is out of range"));
        }
        T& a = accum(index);
        const T& g = grad(i);
        a += g * g;
        auto learning_rate = lr_scalar / std::sqrt(a);
        auto prox_v = var(index);
        prox_v -= learning_rate * g;
        if (l1_scalar > 0) {
          var(index) = sgn(prox_v) *
                       std::max(std::abs(prox_v) - learning_rate * l1_scalar,
                                static_cast<T>(0.0)) /
                       (1.0 + l2_scalar * learning_rate);
        } else {
          var(index) = prox_v / (1.0 + l2_scalar * learning_rate);
        }
      }
    }
    return Status::OK();
  }
};

template <typename T>
struct ApplyFtrlV2<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power) {
    auto grad_with_shrinkage = grad + static_cast<T>(2) * l2_shrinkage() * var;
    auto new_accum = accum + grad * grad;
    // special case for which lr_power=-0.5.
    if (lr_power() == static_cast<T>(-0.5)) {
      linear.device(d) +=
          grad_with_shrinkage - (new_accum.sqrt() - accum.sqrt()) / lr() * var;
    } else {
      linear.device(d) +=
          grad_with_shrinkage -
          (new_accum.pow(-lr_power()) - accum.pow(-lr_power())) / lr() * var;
    }
    auto x = (linear.constant(l1()) * linear.sign() - linear);
    if (lr_power() == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() / new_accum.constant(lr()) +
               linear.constant(static_cast<T>(2) * l2());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));

    } else {
      auto y = new_accum.pow(-lr_power()) / new_accum.constant(lr()) +
               linear.constant(static_cast<T>(2) * l2());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));
    }
    accum.device(d) += grad * grad;
  }
};

template <typename T>
struct ApplyFtrlV2MultiplyLinearByLr<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power) {
    auto grad_with_shrinkage = grad + static_cast<T>(2) * l2_shrinkage() * var;
    auto new_accum = accum + grad * grad;
    // special case for which lr_power=-0.5.
    if (lr_power() == static_cast<T>(-0.5)) {
      linear.device(d) +=
          grad_with_shrinkage * lr() - (new_accum.sqrt() - accum.sqrt()) * var;
    } else {
      linear.device(d) +=
          grad_with_shrinkage * lr() -
          (new_accum.pow(-lr_power()) - accum.pow(-lr_power())) * var;
    }
    auto x = (linear.constant(l1() * lr()) * linear.sign() - linear);
    if (lr_power() == static_cast<T>(-0.5)) {
      auto y =
          new_accum.sqrt() + linear.constant(static_cast<T>(2) * l2() * lr());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1() * lr()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));

    } else {
      auto y = new_accum.pow(-lr_power()) +
               linear.constant(static_cast<T>(2) * l2() * lr());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1() * lr()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));
    }
    accum.device(d) += grad * grad;
  }
};

template <typename T>
struct ApplyFtrl<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power) {
    auto new_accum = accum + grad.square();
    // special case for which lr_power=-0.5.
    if (lr_power() == static_cast<T>(-0.5)) {
      linear.device(d) += grad - (new_accum.sqrt() - accum.sqrt()) / lr() * var;
    } else {
      linear.device(d) +=
          grad -
          (new_accum.pow(-lr_power()) - accum.pow(-lr_power())) / lr() * var;
    }
    auto x = (linear.constant(l1()) * linear.sign() - linear);
    if (lr_power() == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() / new_accum.constant(lr()) +
               linear.constant(static_cast<T>(2) * l2());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));

    } else {
      auto y = new_accum.pow(-lr_power()) / new_accum.constant(lr()) +
               linear.constant(static_cast<T>(2) * l2());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));
    }
    accum.device(d) += grad.square();
  }
};

template <typename T>
struct ApplyFtrlMultiplyLinearByLr<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power) {
    auto new_accum = accum + grad.square();
    // special case for which lr_power=-0.5.
    if (lr_power() == static_cast<T>(-0.5)) {
      linear.device(d) += grad * lr() - (new_accum.sqrt() - accum.sqrt()) * var;
    } else {
      linear.device(d) +=
          grad * lr() -
          (new_accum.pow(-lr_power()) - accum.pow(-lr_power())) * var;
    }
    auto x = (linear.constant(l1()) * lr() * linear.sign() - linear);
    if (lr_power() == static_cast<T>(-0.5)) {
      auto y =
          new_accum.sqrt() + linear.constant(static_cast<T>(2) * l2() * lr());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1() * lr()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));

    } else {
      auto y = new_accum.pow(-lr_power()) +
               linear.constant(static_cast<T>(2) * l2() * lr());
      auto pre_shrink = x / y;
      var.device(d) = (linear.abs() > linear.constant(l1() * lr()))
                          .select(pre_shrink, var.constant(static_cast<T>(0)));
    }
    accum.device(d) += grad.square();
  }
};

namespace {

template <typename T>
inline T FtrlCompute(const T& accum, const T& linear, const T& lr, const T& l1,
                     const T& l2, const T& lr_power,
                     const bool multiply_linear_by_lr) {
  T quadratic;
  if (multiply_linear_by_lr) {
    if (lr_power == static_cast<T>(-0.5)) {
      quadratic = Eigen::numext::sqrt(accum) + static_cast<T>(2) * l2 * lr;
    } else {
      quadratic =
          Eigen::numext::pow(accum, -lr_power) + static_cast<T>(2) * l2 * lr;
    }
    auto l1_reg_adjust = std::max(std::min(linear, l1 * lr), -l1 * lr);
    return (l1_reg_adjust - linear) / quadratic;
  } else {
    if (lr_power == static_cast<T>(-0.5)) {
      quadratic = Eigen::numext::sqrt(accum) / lr + static_cast<T>(2) * l2;
    } else {
      quadratic =
          Eigen::numext::pow(accum, -lr_power) / lr + static_cast<T>(2) * l2;
    }
    auto l1_reg_adjust = std::max(std::min(linear, l1), -l1);
    return (l1_reg_adjust - linear) / quadratic;
  }
}

template <typename T, typename GradTy, typename GradeMaybeWithShrinkageTy,
          typename AccumTy, typename LinearTy, typename VarTy>
void ComputeFtrl(GradTy grad,
                 GradeMaybeWithShrinkageTy grad_maybe_with_shrinkage,
                 AccumTy accum, LinearTy linear, VarTy var, T l1_scalar,
                 T l2_scalar, bool multiply_linear_by_lr, T lr_power_scalar,
                 T lr_scalar) {
  auto new_accum = accum + grad.square();
  if (multiply_linear_by_lr) {
    if (lr_power_scalar == static_cast<T>(-0.5)) {
      linear += grad_maybe_with_shrinkage * lr_scalar -
                (new_accum.sqrt() - accum.sqrt()) * var;
    } else {
      linear +=
          grad_maybe_with_shrinkage * lr_scalar -
          (new_accum.pow(-lr_power_scalar) - accum.pow(-lr_power_scalar)) * var;
    }
  } else {
    if (lr_power_scalar == static_cast<T>(-0.5)) {
      linear += grad_maybe_with_shrinkage -
                (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;
    } else {
      linear += grad_maybe_with_shrinkage - (new_accum.pow(-lr_power_scalar) -
                                             accum.pow(-lr_power_scalar)) /
                                                lr_scalar * var;
    }
  }
  auto l1_reg_adjust =
      (multiply_linear_by_lr ? linear.cwiseMin(l1_scalar * lr_scalar)
                                   .cwiseMax(-l1_scalar * lr_scalar)
                             : linear.cwiseMin(l1_scalar).cwiseMax(-l1_scalar));
  auto x = l1_reg_adjust - linear;
  if (multiply_linear_by_lr) {
    if (lr_power_scalar == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() +
               linear.constant(static_cast<T>(2) * l2_scalar * lr_scalar);
      var = x / y;
    } else {
      auto y = new_accum.pow(-lr_power_scalar) +
               linear.constant(static_cast<T>(2) * l2_scalar * lr_scalar);
      var = x / y;
    }
  } else {
    if (lr_power_scalar == static_cast<T>(-0.5)) {
      auto y = new_accum.sqrt() / new_accum.constant(lr_scalar) +
               linear.constant(static_cast<T>(2) * l2_scalar);
      var = x / y;
    } else {
      auto y = new_accum.pow(-lr_power_scalar) / new_accum.constant(lr_scalar) +
               linear.constant(static_cast<T>(2) * l2_scalar);
      var = x / y;
    }
  }
  accum += grad.square();
}
}  // namespace

template <typename T, typename Tindex, bool has_l2_shrinkage>
struct SparseApplyFtrl<CPUDevice, T, Tindex, has_l2_shrinkage> {
  Status operator()(const CPUDevice& d, typename TTypes<T>::Matrix var_flat,
                    typename TTypes<T>::Matrix accum_flat,
                    typename TTypes<T>::Matrix linear_flat,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar l1,
                    typename TTypes<T>::ConstScalar l2,
                    typename TTypes<T>::ConstScalar l2_shrinkage,
                    typename TTypes<T>::ConstScalar lr_power,
                    typename TTypes<T>::ConstMatrix grad_flat,
                    typename TTypes<Tindex>::ConstVec indices_vec,
                    int64 inner_dim, bool multiply_linear_by_lr) {
    const Tindex N = static_cast<Tindex>(indices_vec.dimension(0));
    if (N > 0) {
      T lr_scalar = lr();
      T l1_scalar = l1();
      T l2_scalar = l2();
      T l2_shrinkage_scalar;
      if (has_l2_shrinkage) {
        l2_shrinkage_scalar = l2_shrinkage();
      }
      T lr_power_scalar = lr_power();
      if (inner_dim > 1) {
        const Tindex first_dim_size =
            static_cast<Tindex>(var_flat.dimension(0));

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          if (!FastBoundsCheck(index, first_dim_size)) {
            return errors::InvalidArgument(
                strings::StrCat("Index ", index, " at offset ", i,
                                " in indices is out of range"));
          }
          auto accum = accum_flat.template chip<0>(index);
          auto linear = linear_flat.template chip<0>(index);
          auto grad = grad_flat.template chip<0>(i);
          auto var = var_flat.template chip<0>(index);

          if (has_l2_shrinkage) {
            auto grad_with_shrinkage =
                grad + static_cast<T>(2) * l2_shrinkage_scalar * var;
            ComputeFtrl(/*grad=*/grad,
                        /*grad_maybe_with_shrinkage=*/grad_with_shrinkage,
                        /*accum=*/accum, /*linear=*/linear, /*var=*/var,
                        /*l1_scalar=*/l1_scalar, /*l2_scalar=*/l2_scalar,
                        /*multiply_linear_by_lr=*/multiply_linear_by_lr,
                        /*lr_power_scalar=*/lr_power_scalar,
                        /*lr_scalar=*/lr_scalar);
          } else {
            ComputeFtrl(/*grad=*/grad, /*grad_maybe_with_shrinkage=*/grad,
                        /*accum=*/accum, /*linear=*/linear, /*var=*/var,
                        /*l1_scalar=*/l1_scalar, /*l2_scalar=*/l2_scalar,
                        /*multiply_linear_by_lr=*/multiply_linear_by_lr,
                        /*lr_power_scalar=*/lr_power_scalar,
                        /*lr_scalar=*/lr_scalar);
          }
        }
      } else {
        const Tindex first_dim_size = accum_flat.size();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          if (!FastBoundsCheck(index, first_dim_size)) {
            return errors::InvalidArgument(
                strings::StrCat("Index ", index, " at offset ", i,
                                " in indices is out of range"));
          }
          T& a = accum_flat(index);
          T& l = linear_flat(index);
          T& v = var_flat(index);
          T g;
          if (has_l2_shrinkage) {
            g = grad_flat(i) +
                (static_cast<T>(2) * l2_shrinkage_scalar * var_flat(index));
          } else {
            g = grad_flat(i);
          }

          T updated_a = a + grad_flat(i) * grad_flat(i);
          using Eigen::numext::pow;
          T sigma = pow(updated_a, -lr_power_scalar) - pow(a, -lr_power_scalar);
          if (!multiply_linear_by_lr) {
            sigma /= lr_scalar;
          }
          T updated_l = (multiply_linear_by_lr ? l + g * lr_scalar - sigma * v
                                               : l + g - sigma * v);
          v = FtrlCompute(updated_a, updated_l, lr_scalar, l1_scalar, l2_scalar,
                          lr_power_scalar, multiply_linear_by_lr);
          a = updated_a;
          l = updated_l;
        }
      }
    }
    return Status::OK();
  }
};

template <typename T>
struct ApplyMomentum<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    accum.device(d) = accum * momentum() + grad;
    if (use_nesterov) {
      var.device(d) -= grad * lr() + accum * momentum() * lr();
    } else {
      var.device(d) -= accum * lr();
    }
  }
};

template <typename T>
struct ApplyKerasMomentum<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    accum.device(d) = accum * momentum() - grad * lr();
    if (use_nesterov) {
      var.device(d) += (accum * momentum() - grad * lr());
    } else {
      var.device(d) += accum;
    }
  }
};

template <typename T, typename Tindex>
struct SparseApplyKerasMomentum<CPUDevice, T, Tindex> {
  Tindex operator()(const CPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstFlat indices,
                    typename TTypes<T>::ConstScalar momentum,
                    bool use_nesterov) {
    const Tindex N = static_cast<Tindex>(indices.size());
    const Tindex first_dim_size = static_cast<Tindex>(var.dimension(0));
    for (Tindex i = 0; i < N; i++) {
      const Tindex index = internal::SubtleMustCopy(indices(i));
      if (!FastBoundsCheck(index, first_dim_size)) return i;
      auto a = accum.template chip<0>(index);
      auto g = grad.template chip<0>(i);
      auto v = var.template chip<0>(index);
      a = a * a.constant(momentum()) - g * g.constant(lr());
      if (use_nesterov) {
        v += a * a.constant(momentum()) - g * g.constant(lr());
      } else {
        v += a;
      }
    }
    return -1;
  }
};

template <typename Device, typename T>
struct ApplyAdamNonCuda {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
    // Get params length and check if they can be vectorized by packet size.
    Index length = var.size();
    Index packet_size = Eigen::internal::packet_traits<T>::size;
    if (length % packet_size == 0) {
      length = length / packet_size;
    } else {
      packet_size = 1;
    }

    T* var_ptr = var.data();
    T* m_ptr = m.data();
    T* v_ptr = v.data();
    const T* g_ptr = grad.data();
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());
    // beta1 == μ
    // beta2 == ν
    // v     == n
    // var   == θ

    auto shard = [this, var_ptr, m_ptr, v_ptr, g_ptr, alpha, beta1, beta2,
                  epsilon, use_nesterov, packet_size](int begin, int end) {
      int t_size = (end - begin) * packet_size;
      begin = begin * packet_size;
      auto var = typename TTypes<T>::UnalignedTensor(var_ptr + begin, t_size);
      auto m = typename TTypes<T>::UnalignedTensor(m_ptr + begin, t_size);
      auto v = typename TTypes<T>::UnalignedTensor(v_ptr + begin, t_size);
      auto g = typename TTypes<T>::UnalignedConstTensor(g_ptr + begin, t_size);

      if (use_nesterov) {
        m += (g - m) * (T(1) - beta1());
        v += (g.square() - v) * (T(1) - beta2());
        var -= ((g * (T(1) - beta1()) + beta1() * m) * alpha) /
               (v.sqrt() + epsilon());
      } else {
        m += (g - m) * (T(1) - beta1());
        v += (g.square() - v) * (T(1) - beta2());
        var -= (m * alpha) / (v.sqrt() + epsilon());
      }
    };

    // Input data: var, v, m, grad.
    // Output data: var, v, m.
    const int input_bytes = length * packet_size * sizeof(T) * 4;
    const int output_bytes = length * packet_size * sizeof(T) * 3;
    const int compute_cycles =
        // Consider Sub as Add
        (Eigen::TensorOpCost::AddCost<int>() * 5 +
         Eigen::TensorOpCost::MulCost<int>() * 2 +
         Eigen::TensorOpCost::AddCost<T>() * 10 +
         Eigen::TensorOpCost::MulCost<T>() * 6 +
         Eigen::TensorOpCost::DivCost<T>()) *
        length;
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);

    // Eigen device must update 3 variables with 3 different expressions,
    // which is bad for cache locality on CPU. Here use ParallelFor instead of
    // "regular" tensor expressions to get better performance.
    d.parallelFor(length, cost, shard);
  }
};

template <typename T>
struct ApplyAdam<CPUDevice, T> : ApplyAdamNonCuda<CPUDevice, T> {};

template <typename T>
struct ApplyAdamWithAmsgrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat vhat,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    const T alpha = lr() * Eigen::numext::sqrt(T(1) - beta2_power()) /
                    (T(1) - beta1_power());

    m.device(d) += (grad - m) * (T(1) - beta1());
    v.device(d) += (grad.square() - v) * (T(1) - beta2());
    vhat.device(d) = vhat.cwiseMax(v);
    var.device(d) -= (m * alpha) / (vhat.sqrt() + epsilon());
  }
};

template <typename Device, typename T>
struct ApplyAdaMaxNonCuda {
  void operator()(const Device& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    m.device(d) += (grad - m) * (T(1) - beta1());
    // Here v is u in section 7.1
    v.device(d) = (beta2() * v).cwiseMax(grad.abs());
    // var is θ in section 7.1
    var.device(d) -= lr() / (T(1) - beta1_power()) * (m / (v + epsilon()));
  }
};

template <typename T>
struct ApplyAdaMax<CPUDevice, T> : ApplyAdaMaxNonCuda<CPUDevice, T> {};

template <typename T>
struct ApplyRMSProp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    ms.device(d) += (grad.square() - ms) * (static_cast<T>(1) - rho());
    mom.device(d) =
        mom * momentum() + (grad * lr()) / ((ms + epsilon()).sqrt());
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyCenteredRMSProp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,
                  typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    ms.device(d) += (grad.square() - ms) * (static_cast<T>(1) - rho());
    mg.device(d) += (grad - mg) * (static_cast<T>(1) - rho());
    auto denom = (ms - mg.square()) + epsilon();
    mom.device(d) = mom * momentum() + (grad * lr()) / denom.sqrt();
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyAddSign<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    auto sign_gm = grad.sign() * m.sign();
    var.device(d) -= lr() * (alpha() + sign_decay() * sign_gm) * grad;
  }
};

template <typename T>
struct ApplyPowerSign<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar logbase,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    auto sign_gm = grad.sign() * m.sign();
    auto grad_scale = (logbase() * sign_decay() * sign_gm).exp();
    var.device(d) -= lr() * grad_scale * grad;
  }
};

}  // namespace functor

template <typename Device, typename T>
class ApplyGradientDescentOp : public OpKernel {
 public:
  explicit ApplyGradientDescentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    const Tensor& alpha = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& delta = ctx->input(2);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(delta.shape()),
        errors::InvalidArgument("var and delta do not have the same shape",
                                var.shape().DebugString(), " ",
                                delta.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyGradientDescent<Device, T>()(
        device, var.flat<T>(), alpha.scalar<T>(), delta.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyGradientDescent").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyGradientDescentOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyGradientDescent")                \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyGradientDescentOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void ApplyGradientDescent<GPUDevice, T>::operator()(  \
      const GPUDevice& d, typename TTypes<T>::Flat var, \
      typename TTypes<T>::ConstScalar alpha,            \
      typename TTypes<T>::ConstFlat delta);             \
  extern template struct ApplyGradientDescent<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdadeltaOp : public OpKernel {
 public:
  explicit ApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    DoValidate(ctx);
    if (!ctx->status().ok()) return;
    DoCompute(ctx);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var;
    const bool sparse = false;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor accum_update;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_,
                                                   sparse, &accum_update));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, accum_update.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& epsilon = ctx->input(5);
    const Tensor& grad = ctx->input(6);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var;
    const bool sparse = false;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor accum_update;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 2, use_exclusive_lock_,
                                                   sparse, &accum_update));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& epsilon = ctx->input(5);
    const Tensor& grad = ctx->input(6);

    functor::ApplyAdadelta<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), accum_update.flat<T>(),
        lr.scalar<T>(), rho.scalar<T>(), epsilon.scalar<T>(), grad.flat<T>());
  }
};

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyAdadelta").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdadeltaOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdadelta")                \
                              .Device(DEVICE_##D)                      \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .HostMemory("accum_update")              \
                              .TypeConstraint<T>("T"),                 \
                          ApplyAdadeltaOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void ApplyAdadelta<GPUDevice, T>::operator()(                                \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat accum, typename TTypes<T>::Flat accum_update,   \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar rho, \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyAdadelta<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyAdadeltaOp : public OpKernel {
 public:
  explicit SparseApplyAdadeltaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    DoCompute(ctx);
  }

  void DoCompute(OpKernelContext* ctx) {
    Tensor var;
    const bool sparse = true;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum_grad;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum_grad));
    Tensor accum_update;
    OP_REQUIRES_OK(ctx,
                   GetInputTensorFromVariable<CPUDevice, T>(
                       ctx, 2, use_exclusive_lock_, sparse, &accum_update));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum_grad.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, accum_update.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum_grad.shape()),
        errors::InvalidArgument("var and accum_grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum_grad.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(accum_update.shape()),
                errors::InvalidArgument(
                    "var and accum_update do not have the same shape",
                    var.shape().DebugString(), " ",
                    accum_update.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& rho = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    const Tensor& epsilon = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    const Tensor& indices = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      auto var_flat = var.flat_outer_dims<T>();
      auto accum_grad_flat = accum_grad.flat_outer_dims<T>();
      auto accum_update_flat = accum_update.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T rho_scalar = rho.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        auto accum_ = accum_grad_flat.template chip<0>(index);
        auto accum_update_ = accum_update_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        accum_ = accum_ * accum_.constant(rho_scalar) +
                 grad_.square() * grad_.constant(T(1) - rho_scalar);
        const auto update =
            (accum_update_ + accum_update_.constant(epsilon_scalar)).sqrt() *
            (accum_ + accum_.constant(epsilon_scalar)).rsqrt() * grad_;
        auto v = var_flat.template chip<0>(index);
        v -= update * update.constant(lr_scalar);
        accum_update_ =
            accum_update_ * accum_update_.constant(rho_scalar) +
            update.square() * update.constant(static_cast<T>(1) - rho_scalar);
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdadelta")                \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyAdadeltaOp<T, Tindices>);       \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdadelta")        \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyAdadeltaOp<T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T>
class ApplyProximalGradientDescentOp : public OpKernel {
 public:
  explicit ApplyProximalGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    const Tensor& alpha = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& l1 = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& delta = ctx->input(4);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(delta.shape()),
        errors::InvalidArgument("var and delta do not have the same shape",
                                var.shape().DebugString(), " ",
                                delta.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyProximalGradientDescent<Device, T>()(
        device, var.flat<T>(), alpha.scalar<T>(), l1.scalar<T>(),
        l2.scalar<T>(), delta.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                           \
  REGISTER_KERNEL_BUILDER(Name("ApplyProximalGradientDescent")           \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T"),                   \
                          ApplyProximalGradientDescentOp<D##Device, T>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyProximalGradientDescent")   \
                              .HostMemory("var")                         \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T"),                   \
                          ApplyProximalGradientDescentOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyProximalGradientDescentOp : public OpKernel {
 public:
  explicit SparseApplyProximalGradientDescentOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(2);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));
    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();

        // TODO(xbing): extract the common logic for the Fobos update.
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          // compute learning_rate for current step.
          auto learning_rate = v.constant(lr_scalar);
          auto prox_v = v;
          // v = w - g * learning_rate.
          prox_v -= g * learning_rate;
          if (l1_scalar > 0) {
            // compute sign(v) * max(|v|, 0)
            v = prox_v.sign() *
                (prox_v.abs() - learning_rate * prox_v.constant(l1_scalar))
                    .cwiseMax(static_cast<T>(0.0)) /
                (v.constant(1.0) + v.constant(l2_scalar) * learning_rate);
          } else {
            v = prox_v /
                (v.constant(1.0) + v.constant(l2_scalar) * learning_rate);
          }
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        const Tindex first_dim_size = var_flat.size();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          const T& g = grad_flat(i);
          auto learning_rate = lr_scalar;
          auto prox_v = var_flat(index);
          prox_v -= learning_rate * g;
          if (l1_scalar > 0) {
            var_flat(index) =
                sgn(prox_v) *
                std::max(std::abs(prox_v) - learning_rate * l1_scalar,
                         static_cast<T>(0.0)) /
                (1.0 + l2_scalar * learning_rate);
          } else {
            var_flat(index) = prox_v / (1.0 + l2_scalar * learning_rate);
          }
        }
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                         \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyProximalGradientDescent")          \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<Tindices>("Tindices"),          \
                          SparseApplyProximalGradientDescentOp<T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyProximalGradientDescent")  \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .TypeConstraint<Tindices>("Tindices"),          \
                          SparseApplyProximalGradientDescentOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdagradOp : public OpKernel {
 public:
  explicit ApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagrad<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                       lr.scalar<T>(), grad.flat<T>(),
                                       update_slots_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyAdagrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdagradOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagrad")                \
                              .HostMemory("var")                      \
                              .HostMemory("accum")                    \
                              .Device(DEVICE_##D)                     \
                              .TypeConstraint<T>("T"),                \
                          ApplyAdagradOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyAdagrad<GPUDevice, T>::operator()(                            \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad, bool update_slots);             \
  extern template struct ApplyAdagrad<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdagradV2Op : public OpKernel {
 public:
  explicit ApplyAdagradV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& epsilon = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(4);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagradV2<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                         lr.scalar<T>(), epsilon.scalar<T>(),
                                         grad.flat<T>(), update_slots_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ApplyAdagradV2").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdagradV2Op<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagradV2")                \
                              .HostMemory("var")                        \
                              .HostMemory("accum")                      \
                              .Device(DEVICE_##D)                       \
                              .TypeConstraint<T>("T"),                  \
                          ApplyAdagradV2Op<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyAdagradV2<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstScalar epsilon,                            \
      typename TTypes<T>::ConstFlat grad, bool update_slots);             \
  extern template struct ApplyAdagradV2<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyProximalAdagradOp : public OpKernel {
 public:
  explicit ApplyProximalAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     lr.scalar<T>()() > static_cast<T>(0)),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(3);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l1.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(4);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l2.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const Tensor& grad = ctx->input(5);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyProximalAdagrad<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), lr.scalar<T>(), l1.scalar<T>(),
        l2.scalar<T>(), grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyProximalAdagrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyProximalAdagradOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyProximalAdagrad")                \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .HostMemory("accum")                            \
                              .TypeConstraint<T>("T"),                        \
                          ApplyProximalAdagradOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void ApplyProximalAdagrad<GPUDevice, T>::operator()(                        \
      const GPUDevice& d, typename TTypes<T>::Flat var,                       \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr,     \
      typename TTypes<T>::ConstScalar l1, typename TTypes<T>::ConstScalar l2, \
      typename TTypes<T>::ConstFlat grad);                                    \
  extern template struct ApplyProximalAdagrad<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class SparseApplyAdagradOp : public OpKernel {
 public:
  explicit SparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Device& device = ctx->template eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, functor::SparseApplyAdagrad<Device, T, Tindex,
                                         /*has_epsilon = */ false>()(
                 device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
                 // Note: Passing lr as a placeholder for unused epsilon.
                 lr.scalar<T>(), lr.scalar<T>(), grad.flat_outer_dims<T>(),
                 indices.vec<Tindex>(), inner_dim, update_slots_));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T, Tindices)                                 \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagrad")                     \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindices>("Tindices"),     \
                          SparseApplyAdagradOp<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagrad")             \
                              .Device(DEVICE_##D)                        \
                              .TypeConstraint<T>("T")                    \
                              .TypeConstraint<Tindices>("Tindices"),     \
                          SparseApplyAdagradOp<D##Device, T, Tindices>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(CPU, T, int32); \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                            \
  template <>                                                                  \
  Status                                                                       \
  SparseApplyAdagrad<GPUDevice, T, Tindex, /*has_epsilon=*/false>::operator()( \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                      \
      typename TTypes<T>::Matrix accum, typename TTypes<T>::ConstScalar lr,    \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstMatrix grad,                                    \
      typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,              \
      bool update_slots);                                                      \
  extern template struct SparseApplyAdagrad<GPUDevice, T, Tindex,              \
                                            /*has_epsilon=*/false>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, Eigen::half, int64);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, float, int64);
REGISTER_KERNELS(GPU, double, int32);
REGISTER_KERNELS(GPU, double, int64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class SparseApplyAdagradV2Op : public OpKernel {
 public:
  explicit SparseApplyAdagradV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& epsilon = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& grad = ctx->input(4);
    const Tensor& indices = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Device& device = ctx->template eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, functor::SparseApplyAdagrad<Device, T, Tindex,
                                         /*has_epsilon = */ true>()(
                 device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
                 lr.scalar<T>(), epsilon.scalar<T>(), grad.flat_outer_dims<T>(),
                 indices.vec<Tindex>(), inner_dim, update_slots_));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
};

#define REGISTER_KERNELS(D, T, Tindices)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagradV2")                     \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices"),       \
                          SparseApplyAdagradV2Op<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagradV2")             \
                              .Device(DEVICE_##D)                          \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<Tindices>("Tindices"),       \
                          SparseApplyAdagradV2Op<D##Device, T, Tindices>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(CPU, T, int32); \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                           \
  template <>                                                                 \
  Status                                                                      \
  SparseApplyAdagrad<GPUDevice, T, Tindex, /*has_epsilon=*/true>::operator()( \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                     \
      typename TTypes<T>::Matrix accum, typename TTypes<T>::ConstScalar lr,   \
      typename TTypes<T>::ConstScalar epsilon,                                \
      typename TTypes<T>::ConstMatrix grad,                                   \
      typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,             \
      bool update_slots);                                                     \
  extern template struct SparseApplyAdagrad<GPUDevice, T, Tindex,             \
                                            /*has_epsilon=*/true>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, Eigen::half, int64);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, float, int64);
REGISTER_KERNELS(GPU, double, int32);
REGISTER_KERNELS(GPU, double, int64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex>
class SparseApplyProximalAdagradOp : public OpKernel {
 public:
  explicit SparseApplyProximalAdagradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     lr.scalar<T>()() > static_cast<T>(0)),
                errors::InvalidArgument("lr is not a positive scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(3);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l1.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(4);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l2.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));

    const Tensor& grad = ctx->input(5);
    const Tensor& indices = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Device& device = ctx->template eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, functor::SparseApplyProximalAdagrad<Device, T, Tindex>()(
                 device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
                 lr.scalar<T>(), l1.scalar<T>(), l2.scalar<T>(),
                 grad.flat_outer_dims<T>(), indices.vec<Tindex>(), inner_dim));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T, Tindices)                     \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("SparseApplyProximalAdagrad")                     \
          .Device(DEVICE_##D)                                \
          .TypeConstraint<T>("T")                            \
          .TypeConstraint<Tindices>("Tindices"),             \
      SparseApplyProximalAdagradOp<D##Device, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(                                   \
      Name("ResourceSparseApplyProximalAdagrad")             \
          .Device(DEVICE_##D)                                \
          .TypeConstraint<T>("T")                            \
          .TypeConstraint<Tindices>("Tindices"),             \
      SparseApplyProximalAdagradOp<D##Device, T, Tindices>);

REGISTER_KERNELS(CPU, float, int32);
REGISTER_KERNELS(CPU, float, int64);
REGISTER_KERNELS(CPU, double, int32);
REGISTER_KERNELS(CPU, double, int64);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                           \
  template <>                                                                 \
  Status SparseApplyProximalAdagrad<GPUDevice, T, Tindex>::operator()(        \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                     \
      typename TTypes<T>::Matrix accum, typename TTypes<T>::ConstScalar lr,   \
      typename TTypes<T>::ConstScalar l1, typename TTypes<T>::ConstScalar l2, \
      typename TTypes<T>::ConstMatrix grad,                                   \
      typename TTypes<Tindex>::ConstVec indices, int64 inner_dim);            \
  extern template struct SparseApplyProximalAdagrad<GPUDevice, T, Tindex>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, Eigen::half, int64);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, float, int64);
REGISTER_KERNELS(GPU, double, int32);
REGISTER_KERNELS(GPU, double, int64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdagradDAOp : public OpKernel {
 public:
  explicit ApplyAdagradDAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor gradient_accum;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(ctx, 1, use_exclusive_lock_,
                                                   sparse, &gradient_accum));
    Tensor gradient_squared_accum;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<Device, T>(
                 ctx, 2, use_exclusive_lock_, sparse, &gradient_squared_accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, gradient_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, gradient_squared_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_squared_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_squared_accum.shape().DebugString()));

    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(5);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(6);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));
    const Tensor& global_step = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdagradDA<Device, T>()(
        device, var.flat<T>(), gradient_accum.flat<T>(),
        gradient_squared_accum.flat<T>(), lr.scalar<T>(),
        global_step.scalar<int64>()(), l1.scalar<T>(), l2.scalar<T>(),
        grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ApplyAdagradDA").Device(DEVICE_##D).TypeConstraint<T>("T"),   \
      ApplyAdagradDAOp<D##Device, T>);                                    \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdagradDA")                  \
                              .Device(DEVICE_##D)                         \
                              .HostMemory("var")                          \
                              .HostMemory("gradient_accumulator")         \
                              .HostMemory("gradient_squared_accumulator") \
                              .TypeConstraint<T>("T"),                    \
                          ApplyAdagradDAOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyAdagradDAOp : public OpKernel {
 public:
  explicit SparseApplyAdagradDAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor gradient_accum;
    OP_REQUIRES_OK(ctx,
                   GetInputTensorFromVariable<CPUDevice, T>(
                       ctx, 1, use_exclusive_lock_, sparse, &gradient_accum));
    Tensor gradient_squared_accum;
    OP_REQUIRES_OK(
        ctx, GetInputTensorFromVariable<CPUDevice, T>(
                 ctx, 2, use_exclusive_lock_, sparse, &gradient_squared_accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, gradient_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, gradient_squared_accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(gradient_squared_accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                gradient_squared_accum.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l1.shape()),
        errors::InvalidArgument("l1 regularization strength is not a scalar: ",
                                l1.shape().DebugString()));

    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(l2.shape()),
        errors::InvalidArgument("l2 regularization strength is not a scalar: ",
                                l2.shape().DebugString()));

    const Tensor& global_step = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(global_step.shape()),
                errors::InvalidArgument("global_step is not a scalar: ",
                                        global_step.shape().DebugString()));

    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    // AdagradDA update:
    // Let g to be gradient accumulator, gg to be gradient squared accumulator,
    // T be the global step, lr is the learning rate, and k the initial
    // gradient squared accumulator value.
    // w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
    if (N > 0) {
      if (inner_dim > 1) {
        const Tindex first_dim_size = var.dim_size(0);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat_outer_dims<T>();
        auto gradient_accum_flat = gradient_accum.flat_outer_dims<T>();
        auto gradient_squared_accum_flat =
            gradient_squared_accum.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T global_step_scalar = global_step.scalar<int64>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        const double gs_lr = global_step_scalar * lr_scalar;

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          auto ga = gradient_accum_flat.template chip<0>(index);
          auto da = gradient_squared_accum_flat.template chip<0>(index);
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          ga += g;
          da += g.square();
          if (l1_scalar > 0) {
            v = ga.constant(-1.0) * ga.sign() *
                ((ga.abs() / ga.constant(global_step_scalar)) -
                 ga.constant(l1_scalar))
                    .cwiseMax(static_cast<T>(0.0)) /
                (v.constant(l2_scalar) + da.sqrt() / v.constant(gs_lr));
          } else {
            v = ga.constant(-1.0) * (ga / ga.constant(global_step_scalar)) /
                (v.constant(l2_scalar) + da.sqrt() / v.constant(gs_lr));
          }
        }
      } else {
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto gradient_accum_flat = gradient_accum.flat<T>();
        auto gradient_squared_accum_flat = gradient_squared_accum.flat<T>();
        auto grad_flat = grad.flat<T>();
        const double lr_scalar = lr.scalar<T>()();
        const int64 global_step_scalar = global_step.scalar<int64>()();
        const double l1_scalar = l1.scalar<T>()();
        const double l2_scalar = l2.scalar<T>()();
        const Tindex first_dim_size = var_flat.size();
        const double gs_l1 = global_step_scalar * l1_scalar;
        const double gs_l2_lr = global_step_scalar * l2_scalar * lr_scalar;

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = internal::SubtleMustCopy(indices_vec(i));
          OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                      errors::InvalidArgument(
                          strings::StrCat("Index ", index, " at offset ", i,
                                          " in indices is out of range")));
          T& ga = gradient_accum_flat(index);
          T& da = gradient_squared_accum_flat(index);
          const double g = grad_flat(i);
          ga += g;
          da += g * g;
          if (l1_scalar > 0) {
            var_flat(index) = sgn(-ga) * lr_scalar *
                              std::max((std::abs(ga) - gs_l1), 0.0) /
                              (gs_l2_lr + std::sqrt(da));
          } else {
            var_flat(index) = (-ga * lr_scalar) / (gs_l2_lr + std::sqrt(da));
          }
        }
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                     \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagradDA")                    \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<Tindices>("Tindices"),      \
                          SparseApplyAdagradDAOp<T, Tindices>);           \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyAdagradDA")            \
                              .Device(DEVICE_CPU)                         \
                              .HostMemory("var")                          \
                              .HostMemory("gradient_accumulator")         \
                              .HostMemory("gradient_squared_accumulator") \
                              .TypeConstraint<T>("T")                     \
                              .TypeConstraint<Tindices>("Tindices"),      \
                          SparseApplyAdagradDAOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T, bool has_l2_shrinkage>
class ApplyFtrlOp : public OpKernel {
 public:
  explicit ApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("multiply_linear_by_lr", &multiply_linear_by_lr_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor linear;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &linear));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& l1 = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1.shape()),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2.shape()),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 8 : 7;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power.shape()),
                errors::InvalidArgument("lr_power is not a scalar",
                                        lr_power.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    if (has_l2_shrinkage) {
      const Tensor& l2_shrinkage = ctx->input(7);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(l2_shrinkage.shape()),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a scalar: ",
                                  l2_shrinkage.shape().DebugString()));
      if (multiply_linear_by_lr_) {
        functor::ApplyFtrlV2<Device, T>()(
            device, var.flat<T>(), accum.flat<T>(), linear.flat<T>(),
            grad.flat<T>(), lr.scalar<T>(), l1.scalar<T>(), l2.scalar<T>(),
            l2_shrinkage.scalar<T>(), lr_power.scalar<T>());
      } else {
        functor::ApplyFtrlV2MultiplyLinearByLr<Device, T>()(
            device, var.flat<T>(), accum.flat<T>(), linear.flat<T>(),
            grad.flat<T>(), lr.scalar<T>(), l1.scalar<T>(), l2.scalar<T>(),
            l2_shrinkage.scalar<T>(), lr_power.scalar<T>());
      }
    } else if (multiply_linear_by_lr_) {
      functor::ApplyFtrlMultiplyLinearByLr<Device, T>()(
          device, var.flat<T>(), accum.flat<T>(), linear.flat<T>(),
          grad.flat<T>(), lr.scalar<T>(), l1.scalar<T>(), l2.scalar<T>(),
          lr_power.scalar<T>());
    } else {
      functor::ApplyFtrl<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                      linear.flat<T>(), grad.flat<T>(),
                                      lr.scalar<T>(), l1.scalar<T>(),
                                      l2.scalar<T>(), lr_power.scalar<T>());
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool multiply_linear_by_lr_;
};

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyFtrl").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyFtrlOp<D##Device, T, /*has_l2_shrinkage=*/false>);      \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ResourceApplyFtrl")                                    \
          .HostMemory("var")                                       \
          .HostMemory("accum")                                     \
          .HostMemory("linear")                                    \
          .Device(DEVICE_##D)                                      \
          .TypeConstraint<T>("T"),                                 \
      ApplyFtrlOp<D##Device, T, /*has_l2_shrinkage=*/false>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void ApplyFtrl<GPUDevice, T>::operator()(                                   \
      const GPUDevice& d, typename TTypes<T>::Flat var,                       \
      typename TTypes<T>::Flat accum, typename TTypes<T>::Flat linear,        \
      typename TTypes<T>::ConstFlat grad, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstScalar l1, typename TTypes<T>::ConstScalar l2, \
      typename TTypes<T>::ConstScalar lr_power);                              \
  extern template struct ApplyFtrl<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(D, T)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ApplyFtrlV2").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyFtrlOp<D##Device, T, /*has_l2_shrinkage=*/true>);         \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ResourceApplyFtrlV2")                                    \
          .HostMemory("var")                                         \
          .HostMemory("accum")                                       \
          .HostMemory("linear")                                      \
          .Device(DEVICE_##D)                                        \
          .TypeConstraint<T>("T"),                                   \
      ApplyFtrlOp<D##Device, T, /*has_l2_shrinkage=*/true>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void ApplyFtrlV2<GPUDevice, T>::operator()(                                 \
      const GPUDevice& d, typename TTypes<T>::Flat var,                       \
      typename TTypes<T>::Flat accum, typename TTypes<T>::Flat linear,        \
      typename TTypes<T>::ConstFlat grad, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstScalar l1, typename TTypes<T>::ConstScalar l2, \
      typename TTypes<T>::ConstScalar l2_shrinkage,                           \
      typename TTypes<T>::ConstScalar lr_power);                              \
  extern template struct ApplyFtrlV2<GPUDevice, T>;                           \
  template <>                                                                 \
  void ApplyFtrlV2MultiplyLinearByLr<GPUDevice, T>::operator()(               \
      const GPUDevice& d, typename TTypes<T>::Flat var,                       \
      typename TTypes<T>::Flat accum, typename TTypes<T>::Flat linear,        \
      typename TTypes<T>::ConstFlat grad, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstScalar l1, typename TTypes<T>::ConstScalar l2, \
      typename TTypes<T>::ConstScalar l2_shrinkage,                           \
      typename TTypes<T>::ConstScalar lr_power);                              \
  extern template struct ApplyFtrlV2MultiplyLinearByLr<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T, typename Tindex, bool has_l2_shrinkage>
class SparseApplyFtrlOp : public OpKernel {
 public:
  explicit SparseApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("multiply_linear_by_lr", &multiply_linear_by_lr_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    Tensor linear;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &linear));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(linear.shape()),
        errors::InvalidArgument("var and linear do not have the same shape",
                                var.shape().DebugString(), " ",
                                linear.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    // Note: The range checks on lr, l1, l2, and lr_power below are disabled
    // for non-CPU devices because their values cannot be accessed directly from
    // the host. The GPU kernel will not crash if these conditions are not met,
    // it will simply produce a bogus answer (possibly inf/nan).
    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsScalar(lr.shape()) &&
            (!std::is_same<Device, CPUDevice>::value ||
             lr.scalar<T>()() > static_cast<T>(0) ||
             (multiply_linear_by_lr_ && lr.scalar<T>()() >= static_cast<T>(0))),
        errors::InvalidArgument("lr is not a positive scalar (or zero if "
                                "multiply_linear_by_lr is set): ",
                                lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l1.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l1.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l1 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(l2.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     l2.scalar<T>()() >= static_cast<T>(0)),
                errors::InvalidArgument("l2 regularization strength is not a "
                                        "non-negative scalar: ",
                                        l2.shape().DebugString()));
    const int lr_power_index = has_l2_shrinkage ? 9 : 8;
    const Tensor& lr_power = ctx->input(lr_power_index);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr_power.shape()) &&
                    (!std::is_same<Device, CPUDevice>::value ||
                     lr_power.scalar<T>()() <= static_cast<T>(0)),
                errors::InvalidArgument("lr_power is not a "
                                        "non-positive scalar: ",
                                        lr_power.shape().DebugString()));
    int64 inner_dim = 1;
    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
      inner_dim *= grad.dim_size(d);
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    OP_REQUIRES(ctx, inner_dim > 0,
                errors::InvalidArgument(
                    "Inner dimension should be greater than zero."));

    const Tensor* l2_shrinkage;
    if (has_l2_shrinkage) {
      l2_shrinkage = &ctx->input(8);
      OP_REQUIRES(
          ctx,
          TensorShapeUtils::IsScalar(l2_shrinkage->shape()) &&
              (!std::is_same<Device, CPUDevice>::value ||
               l2_shrinkage->scalar<T>()() >= static_cast<T>(0)),
          errors::InvalidArgument("l2 shrinkage regularization strength "
                                  "is not a non-negative scalar: ",
                                  l2_shrinkage->shape().DebugString()));
    }

    const Device& device = ctx->template eigen_device<Device>();
    auto indices_vec = indices.vec<Tindex>();
    OP_REQUIRES_OK(
        ctx, functor::SparseApplyFtrl<Device, T, Tindex, has_l2_shrinkage>()(
                 device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
                 linear.flat_outer_dims<T>(), lr.scalar<T>(), l1.scalar<T>(),
                 l2.scalar<T>(),
                 // Note: Passing l2 as a placeholder when not has_l2_shrinkage
                 // (it will not be used).
                 has_l2_shrinkage ? l2_shrinkage->scalar<T>() : l2.scalar<T>(),
                 lr_power.scalar<T>(), grad.flat_outer_dims<T>(), indices_vec,
                 inner_dim, multiply_linear_by_lr_));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool multiply_linear_by_lr_;
};

#define REGISTER_KERNELS(D, T, Tindices)                                      \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseApplyFtrl")                                                 \
          .Device(DEVICE_##D)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/false>); \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ResourceSparseApplyFtrl")                                         \
          .Device(DEVICE_##D)                                                 \
          .TypeConstraint<T>("T")                                             \
          .TypeConstraint<Tindices>("Tindices"),                              \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/false>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(CPU, T, int32); \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                           \
  template <>                                                                 \
  Status SparseApplyFtrl<GPUDevice, T, Tindex, /*has_l2_shrinkage=*/false>::  \
  operator()(                                                                 \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                     \
      typename TTypes<T>::Matrix accum, typename TTypes<T>::Matrix linear,    \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar l1, \
      typename TTypes<T>::ConstScalar l2,                                     \
      typename TTypes<T>::ConstScalar l2_shrinkage,                           \
      typename TTypes<T>::ConstScalar lr_power,                               \
      typename TTypes<T>::ConstMatrix grad,                                   \
      typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,             \
      bool multiply_linear_by_lr);                                            \
  extern template struct SparseApplyFtrl<GPUDevice, T, Tindex,                \
                                         /*has_l2_shrinkage=*/false>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, Eigen::half, int64);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, float, int64);
REGISTER_KERNELS(GPU, double, int32);
REGISTER_KERNELS(GPU, double, int64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

#define REGISTER_KERNELS(D, T, Tindices)                                     \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseApplyFtrlV2")                                              \
          .Device(DEVICE_##D)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/true>); \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("ResourceSparseApplyFtrlV2")                                      \
          .Device(DEVICE_##D)                                                \
          .TypeConstraint<T>("T")                                            \
          .TypeConstraint<Tindices>("Tindices"),                             \
      SparseApplyFtrlOp<D##Device, T, Tindices, /*has_l2_shrinkage=*/true>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(CPU, T, int32); \
  REGISTER_KERNELS(CPU, T, int64);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                           \
  template <>                                                                 \
  Status SparseApplyFtrl<GPUDevice, T, Tindex, /*has_l2_shrinkage=*/true>::   \
  operator()(                                                                 \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                     \
      typename TTypes<T>::Matrix accum, typename TTypes<T>::Matrix linear,    \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar l1, \
      typename TTypes<T>::ConstScalar l2,                                     \
      typename TTypes<T>::ConstScalar l2_shrinkage,                           \
      typename TTypes<T>::ConstScalar lr_power,                               \
      typename TTypes<T>::ConstMatrix grad,                                   \
      typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,             \
      bool multiply_linear_by_lr);                                            \
  extern template struct SparseApplyFtrl<GPUDevice, T, Tindex,                \
                                         /*has_l2_shrinkage=*/true>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half, int32);
REGISTER_KERNELS(GPU, Eigen::half, int64);
REGISTER_KERNELS(GPU, float, int32);
REGISTER_KERNELS(GPU, float, int64);
REGISTER_KERNELS(GPU, double, int32);
REGISTER_KERNELS(GPU, double, int64);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyMomentumOp : public OpKernel {
 public:
  explicit ApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& momentum = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyMomentum<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                        lr.scalar<T>(), grad.flat<T>(),
                                        momentum.scalar<T>(), use_nesterov_);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyMomentum").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyMomentumOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyMomentum")                \
                              .Device(DEVICE_##D)                      \
                              .HostMemory("var")                       \
                              .HostMemory("accum")                     \
                              .TypeConstraint<T>("T"),                 \
                          ApplyMomentumOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyMomentum<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad,                                 \
      typename TTypes<T>::ConstScalar momentum, bool use_nesterov);       \
  extern template struct ApplyMomentum<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyMomentumOp : public OpKernel {
 public:
  explicit SparseApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    const Tensor& momentum = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      auto indices_vec = indices.vec<Tindex>();
      auto var_flat = var.flat_outer_dims<T>();
      auto accum_flat = accum.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      T lr_scalar = lr.scalar<T>()();
      T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = internal::SubtleMustCopy(indices_vec(i));
        OP_REQUIRES(ctx, FastBoundsCheck(index, first_dim_size),
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
        auto a = accum_flat.template chip<0>(index);
        auto g = grad_flat.template chip<0>(i);
        auto v = var_flat.template chip<0>(index);
        a = a * a.constant(momentum_scalar) + g;
        if (use_nesterov_) {
          v -= g.constant(lr_scalar) * g +
               a.constant(lr_scalar) * a.constant(momentum_scalar) * a;
        } else {
          v -= a.constant(lr_scalar) * a;
        }
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyMomentum")                \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyMomentumOp<T, Tindices>);       \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyMomentum")        \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyMomentumOp<T, Tindices>);
#define REGISTER_CPU_KERNELS(T) \
  REGISTER_KERNELS(T, int32);   \
  REGISTER_KERNELS(T, int64);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyKerasMomentumOp : public OpKernel {
 public:
  explicit ApplyKerasMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Tensor& momentum = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyKerasMomentum<Device, T>()(
        device, var.flat<T>(), accum.flat<T>(), lr.scalar<T>(), grad.flat<T>(),
        momentum.scalar<T>(), use_nesterov_);
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(D, T)                               \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyKerasMomentum") \
                              .Device(DEVICE_##D)            \
                              .HostMemory("var")             \
                              .HostMemory("accum")           \
                              .TypeConstraint<T>("T"),       \
                          ApplyKerasMomentumOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyKerasMomentum<GPUDevice, T>::operator()(                      \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad,                                 \
      typename TTypes<T>::ConstScalar momentum, bool use_nesterov);       \
  extern template struct ApplyKerasMomentum<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Device, typename Tindex>
class SparseApplyKerasMomentumOp : public OpKernel {
 public:
  explicit SparseApplyKerasMomentumOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor accum;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &accum));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    const Tensor& grad = ctx->input(3);
    const Tensor& indices = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(ctx, var.dim_size(d) == grad.dim_size(d),
                  errors::InvalidArgument(strings::StrCat(
                      "var and grad must match in dimension ", d)));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    const Tensor& momentum = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    auto indices_flat = indices.flat<Tindex>();
    const Tindex bad_i = functor::SparseApplyKerasMomentum<Device, T, Tindex>()(
        device, var.flat_outer_dims<T>(), accum.flat_outer_dims<T>(),
        lr.scalar<T>(), grad.flat_outer_dims<T>(), indices_flat,
        momentum.scalar<T>(), use_nesterov_);
    OP_REQUIRES(
        ctx, bad_i < 0,
        errors::InvalidArgument(
            "indices", SliceDebugString(indices.shape(), bad_i), " = ",
            indices_flat(bad_i), " is not in [0, ", var.dim_size(0), ")"));

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(T, D, Tindices)                             \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyKerasMomentum")   \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyKerasMomentumOp<T, D##Device, Tindices>);
#define REGISTER_CPU_KERNELS(T)    \
  REGISTER_KERNELS(T, CPU, int32); \
  REGISTER_KERNELS(T, CPU, int64);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                         \
  template <>                                                               \
  Tindex SparseApplyKerasMomentum<GPUDevice, T, Tindex>::operator()(        \
      const GPUDevice& d, typename TTypes<T>::Matrix var,                   \
      typename TTypes<T>::Matrix accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstMatrix grad,                                 \
      typename TTypes<Tindex>::ConstFlat indices,                           \
      typename TTypes<T>::ConstScalar momentum, bool use_nesterov);         \
  extern template struct SparseApplyKerasMomentum<GPUDevice, T, Tindex>;
DECLARE_GPU_SPEC(Eigen::half, int32);
DECLARE_GPU_SPEC(Eigen::half, int64);
DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);
DECLARE_GPU_SPEC(double, int32);
DECLARE_GPU_SPEC(double, int64);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
DECLARE_GPU_SPEC(complex64, int32);
DECLARE_GPU_SPEC(complex64, int64);
DECLARE_GPU_SPEC(complex128, int32);
DECLARE_GPU_SPEC(complex128, int64);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNELS(T)    \
  REGISTER_KERNELS(T, GPU, int32); \
  REGISTER_KERNELS(T, GPU, int64);

REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
REGISTER_GPU_KERNELS(complex64);
REGISTER_GPU_KERNELS(complex128);
#endif
#undef REGISTER_GPU_KERNELS
#endif
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdamOp : public OpKernel {
 public:
  explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdam<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(),
        beta1_power.scalar<T>(), beta2_power.scalar<T>(), lr.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        grad.flat<T>(), use_nesterov_);

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyAdam").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdamOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdam")                \
                              .HostMemory("var")                   \
                              .HostMemory("m")                     \
                              .HostMemory("v")                     \
                              .Device(DEVICE_##D)                  \
                              .TypeConstraint<T>("T"),             \
                          ApplyAdamOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                   \
  template <>                                                 \
  void ApplyAdam<GPUDevice, T>::operator()(                   \
      const GPUDevice& d, typename TTypes<T>::Flat var,       \
      typename TTypes<T>::Flat m, typename TTypes<T>::Flat v, \
      typename TTypes<T>::ConstScalar beta1_power,            \
      typename TTypes<T>::ConstScalar beta2_power,            \
      typename TTypes<T>::ConstScalar lr,                     \
      typename TTypes<T>::ConstScalar beta1,                  \
      typename TTypes<T>::ConstScalar beta2,                  \
      typename TTypes<T>::ConstScalar epsilon,                \
      typename TTypes<T>::ConstFlat grad, bool use_nesterov); \
  extern template struct ApplyAdam<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdamWithAmsgradOp : public OpKernel {
 public:
  explicit ApplyAdamWithAmsgradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    Tensor vhat;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, sparse, &vhat));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, vhat.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& beta1_power = ctx->input(4);
    const Tensor& beta2_power = ctx->input(5);
    const Tensor& lr = ctx->input(6);
    const Tensor& beta1 = ctx->input(7);
    const Tensor& beta2 = ctx->input(8);
    const Tensor& epsilon = ctx->input(9);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(10);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdamWithAmsgrad<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(), vhat.flat<T>(),
        beta1_power.scalar<T>(), beta2_power.scalar<T>(), lr.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                 \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdamWithAmsgrad") \
                              .HostMemory("var")               \
                              .HostMemory("m")                 \
                              .HostMemory("v")                 \
                              .HostMemory("vhat")              \
                              .Device(DEVICE_##D)              \
                              .TypeConstraint<T>("T"),         \
                          ApplyAdamWithAmsgradOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                   \
  template <>                                                 \
  void ApplyAdamWithAmsgrad<GPUDevice, T>::operator()(        \
      const GPUDevice& d, typename TTypes<T>::Flat var,       \
      typename TTypes<T>::Flat m, typename TTypes<T>::Flat v, \
      typename TTypes<T>::Flat vhat,                          \
      typename TTypes<T>::ConstScalar beta1_power,            \
      typename TTypes<T>::ConstScalar beta2_power,            \
      typename TTypes<T>::ConstScalar lr,                     \
      typename TTypes<T>::ConstScalar beta1,                  \
      typename TTypes<T>::ConstScalar beta2,                  \
      typename TTypes<T>::ConstScalar epsilon,                \
      typename TTypes<T>::ConstFlat grad);                    \
  extern template struct ApplyAdamWithAmsgrad<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdaMaxOp : public OpKernel {
 public:
  explicit ApplyAdaMaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& beta1 = ctx->input(5);
    const Tensor& beta2 = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(8);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAdaMax<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), v.flat<T>(),
        beta1_power.scalar<T>(), lr.scalar<T>(), beta1.scalar<T>(),
        beta2.scalar<T>(), epsilon.scalar<T>(), grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("ApplyAdaMax").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdaMaxOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdaMax")                \
                              .HostMemory("var")                     \
                              .HostMemory("m")                       \
                              .HostMemory("v")                       \
                              .Device(DEVICE_##D)                    \
                              .TypeConstraint<T>("T"),               \
                          ApplyAdaMaxOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                   \
  template <>                                                 \
  void ApplyAdaMax<GPUDevice, T>::operator()(                 \
      const GPUDevice& d, typename TTypes<T>::Flat var,       \
      typename TTypes<T>::Flat m, typename TTypes<T>::Flat v, \
      typename TTypes<T>::ConstScalar beta1_power,            \
      typename TTypes<T>::ConstScalar lr,                     \
      typename TTypes<T>::ConstScalar beta1,                  \
      typename TTypes<T>::ConstScalar beta2,                  \
      typename TTypes<T>::ConstScalar epsilon,                \
      typename TTypes<T>::ConstFlat grad);                    \
  extern template struct ApplyAdaMax<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyRMSPropOp : public OpKernel {
 public:
  explicit ApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor ms;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &ms));
    Tensor mom;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &mom));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyRMSProp<Device, T>()(device, var.flat<T>(), ms.flat<T>(),
                                       mom.flat<T>(), lr.scalar<T>(),
                                       rho.scalar<T>(), momentum.scalar<T>(),
                                       epsilon.scalar<T>(), grad.flat<T>());

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

template <typename Device, typename T>
class ApplyCenteredRMSPropOp : public OpKernel {
 public:
  explicit ApplyCenteredRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor mg;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &mg));
    Tensor ms;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &ms));
    Tensor mom;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 3, use_exclusive_lock_, sparse, &mom));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, mg.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));

    const Tensor& lr = ctx->input(4);
    const Tensor& rho = ctx->input(5);
    const Tensor& momentum = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);
    const Tensor& grad = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mg.shape()),
                errors::InvalidArgument("var and mg do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyCenteredRMSProp<Device, T>()(
        device, var.flat<T>(), mg.flat<T>(), ms.flat<T>(), mom.flat<T>(),
        lr.scalar<T>(), rho.scalar<T>(), momentum.scalar<T>(),
        epsilon.scalar<T>(), grad.flat<T>());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"),         \
      ApplyRMSPropOp<D##Device, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyCenteredRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyCenteredRMSPropOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyRMSProp")                        \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .HostMemory("ms")                               \
                              .HostMemory("mom")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyRMSPropOp<D##Device, T>);                      \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyCenteredRMSProp")                \
                              .Device(DEVICE_##D)                             \
                              .HostMemory("var")                              \
                              .HostMemory("mg")                               \
                              .HostMemory("ms")                               \
                              .HostMemory("mom")                              \
                              .TypeConstraint<T>("T"),                        \
                          ApplyCenteredRMSPropOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_FLOAT_TYPES(REGISTER_CPU_KERNELS);
TF_CALL_COMPLEX_TYPES(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void ApplyRMSProp<GPUDevice, T>::operator()(                                 \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,               \
      typename TTypes<T>::ConstScalar lr, typename TTypes<T>::ConstScalar rho, \
      typename TTypes<T>::ConstScalar momentum,                                \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyRMSProp<GPUDevice, T>;                           \
  template <>                                                                  \
  void ApplyCenteredRMSProp<GPUDevice, T>::operator()(                         \
      const GPUDevice& d, typename TTypes<T>::Flat var,                        \
      typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,                \
      typename TTypes<T>::Flat mom, typename TTypes<T>::ConstScalar lr,        \
      typename TTypes<T>::ConstScalar rho,                                     \
      typename TTypes<T>::ConstScalar momentum,                                \
      typename TTypes<T>::ConstScalar epsilon,                                 \
      typename TTypes<T>::ConstFlat grad);                                     \
  extern template struct ApplyCenteredRMSProp<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
REGISTER_KERNELS(GPU, complex64);
REGISTER_KERNELS(GPU, complex128);
#endif
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyRMSPropOp : public OpKernel {
 public:
  explicit SparseApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor ms;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &ms));
    Tensor mom;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &mom));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);
    const Tensor& indices = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(
          ctx, var.dim_size(d) == grad.dim_size(d),
          errors::InvalidArgument("var and grad must match in dimension ", d));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      auto var_flat = var.flat_outer_dims<T>();
      auto ms_flat = ms.flat_outer_dims<T>();
      auto mom_flat = mom.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T rho_scalar = rho.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();
      const T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);

        auto ms_ = ms_flat.template chip<0>(index);
        auto mom_ = mom_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        ms_ = ms_ * ms_.constant(rho_scalar) +
              grad_.square() * grad_.constant(T(1) - rho_scalar);
        mom_ = mom_ * mom_.constant(momentum_scalar) +
               (ms_ + ms_.constant(epsilon_scalar)).rsqrt() *
                   ms_.constant(lr_scalar) * grad_;

        auto v = var_flat.template chip<0>(index);
        v -= mom_;
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyCenteredRMSPropOp : public OpKernel {
 public:
  explicit SparseApplyCenteredRMSPropOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override TF_NO_THREAD_SAFETY_ANALYSIS {
    const bool sparse = true;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1, 2, 3});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor mg;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &mg));
    Tensor ms;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 2, use_exclusive_lock_, sparse, &ms));
    Tensor mom;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, T>(
                            ctx, 3, use_exclusive_lock_, sparse, &mom));

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));

    const Tensor& lr = ctx->input(4);
    const Tensor& rho = ctx->input(5);
    const Tensor& momentum = ctx->input(6);
    const Tensor& epsilon = ctx->input(7);
    const Tensor& grad = ctx->input(8);
    const Tensor& indices = ctx->input(9);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(rho.shape()),
                errors::InvalidArgument("rho is not a scalar: ",
                                        rho.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum.shape()),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mg.shape()),
                errors::InvalidArgument("var and mg do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        mg.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(ms.shape()),
                errors::InvalidArgument("var and ms do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        ms.shape().DebugString()));

    OP_REQUIRES(ctx, var.shape().IsSameSize(mom.shape()),
                errors::InvalidArgument(
                    "var and mom do not have the same shape",
                    var.shape().DebugString(), " ", mom.shape().DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be one-dimensional"));

    for (int d = 1; d < var.dims(); d++) {
      OP_REQUIRES(
          ctx, var.dim_size(d) == grad.dim_size(d),
          errors::InvalidArgument("var and grad must match in dimension ", d));
    }
    const Tindex N = indices.dim_size(0);
    OP_REQUIRES(
        ctx, grad.dim_size(0) == N,
        errors::InvalidArgument(
            "grad must be the same size as indices in the first dimension."));

    if (N > 0) {
      const Tindex first_dim_size = var.dim_size(0);
      // Validate all the indices are in range
      auto indices_vec = indices.vec<Tindex>();
      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        OP_REQUIRES(ctx, index >= 0 && index < first_dim_size,
                    errors::InvalidArgument(
                        strings::StrCat("Index ", index, " at offset ", i,
                                        " in indices is out of range")));
      }

      auto var_flat = var.flat_outer_dims<T>();
      auto ms_flat = ms.flat_outer_dims<T>();
      auto mg_flat = mg.flat_outer_dims<T>();
      auto mom_flat = mom.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      const T lr_scalar = lr.scalar<T>()();
      const T rho_scalar = rho.scalar<T>()();
      const T epsilon_scalar = epsilon.scalar<T>()();
      const T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);

        auto ms_ = ms_flat.template chip<0>(index);
        auto mom_ = mom_flat.template chip<0>(index);
        auto grad_ = grad_flat.template chip<0>(i);

        ms_ = ms_ * ms_.constant(rho_scalar) +
              grad_.square() * grad_.constant(T(1) - rho_scalar);

        auto mg_ = mg_flat.template chip<0>(index);
        mg_ = mg_ * mg_.constant(rho_scalar) +
              grad_ * grad_.constant(T(1) - rho_scalar);
        auto denom_ = ms_ + ms_.constant(epsilon_scalar) - mg_.square();
        mom_ = mom_ * mom_.constant(momentum_scalar) +
               denom_.rsqrt() * ms_.constant(lr_scalar) * grad_;
        auto v = var_flat.template chip<0>(index);
        v -= mom_;
      }
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                 \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyRMSProp")                  \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          SparseApplyRMSPropOp<T, Tindices>);         \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyCenteredRMSProp")          \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          SparseApplyCenteredRMSPropOp<T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyRMSProp")          \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          SparseApplyRMSPropOp<T, Tindices>);         \
  REGISTER_KERNEL_BUILDER(Name("ResourceSparseApplyCenteredRMSProp")  \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T>("T")                 \
                              .TypeConstraint<Tindices>("Tindices"),  \
                          SparseApplyCenteredRMSPropOp<T, Tindices>);

REGISTER_KERNELS(Eigen::half, int32);
REGISTER_KERNELS(Eigen::half, int64);
REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
REGISTER_KERNELS(complex64, int32);
REGISTER_KERNELS(complex64, int64);
REGISTER_KERNELS(complex128, int32);
REGISTER_KERNELS(complex128, int64);

#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAddSignOp : public OpKernel {
 public:
  explicit ApplyAddSignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& alpha = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& sign_decay = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(alpha.shape()),
                errors::InvalidArgument("sign_decay is not a scalar: ",
                                        sign_decay.shape().DebugString()));
    const Tensor& beta = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta.shape()),
                errors::InvalidArgument("beta is not a scalar: ",
                                        beta.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyAddSign<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), lr.scalar<T>(), alpha.scalar<T>(),
        sign_decay.scalar<T>(), beta.scalar<T>(), grad.flat<T>());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyAddSign").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAddSignOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyAddSign")                \
                              .Device(DEVICE_##D)                     \
                              .HostMemory("var")                      \
                              .HostMemory("m")                        \
                              .TypeConstraint<T>("T"),                \
                          ApplyAddSignOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                           \
  template <>                                                         \
  void ApplyAddSign<GPUDevice, T>::operator()(                        \
      const GPUDevice& d, typename TTypes<T>::Flat var,               \
      typename TTypes<T>::Flat m, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstScalar alpha,                          \
      typename TTypes<T>::ConstScalar sign_decay,                     \
      typename TTypes<T>::ConstScalar beta,                           \
      typename TTypes<T>::ConstFlat grad);                            \
  extern template struct ApplyAddSign<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyPowerSignOp : public OpKernel {
 public:
  explicit ApplyPowerSignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<Device, T>(
        ctx, use_exclusive_lock_, sparse, {0, 1});

    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<Device, T>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& logbase = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(logbase.shape()),
                errors::InvalidArgument("logbase is not a scalar: ",
                                        logbase.shape().DebugString()));
    const Tensor& sign_decay = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(logbase.shape()),
                errors::InvalidArgument("sign_decay is not a scalar: ",
                                        sign_decay.shape().DebugString()));
    const Tensor& beta = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta.shape()),
                errors::InvalidArgument("beta is not a scalar: ",
                                        beta.shape().DebugString()));
    const Tensor& grad = ctx->input(6);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();
    functor::ApplyPowerSign<Device, T>()(
        device, var.flat<T>(), m.flat<T>(), lr.scalar<T>(), logbase.scalar<T>(),
        sign_decay.scalar<T>(), beta.scalar<T>(), grad.flat<T>());
    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(D, T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ApplyPowerSign").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyPowerSignOp<D##Device, T>);                                  \
  REGISTER_KERNEL_BUILDER(Name("ResourceApplyPowerSign")                \
                              .Device(DEVICE_##D)                       \
                              .HostMemory("var")                        \
                              .HostMemory("m")                          \
                              .TypeConstraint<T>("T"),                  \
                          ApplyPowerSignOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);

TF_CALL_half(REGISTER_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_CPU_KERNELS);
TF_CALL_float(REGISTER_CPU_KERNELS);
TF_CALL_double(REGISTER_CPU_KERNELS);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                           \
  template <>                                                         \
  void ApplyPowerSign<GPUDevice, T>::operator()(                      \
      const GPUDevice& d, typename TTypes<T>::Flat var,               \
      typename TTypes<T>::Flat m, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstScalar logbase,                        \
      typename TTypes<T>::ConstScalar sign_decay,                     \
      typename TTypes<T>::ConstScalar beta,                           \
      typename TTypes<T>::ConstFlat grad);                            \
  extern template struct ApplyPowerSign<GPUDevice, T>;
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_CPU_KERNELS
#undef REGISTER_KERNELS

}  // namespace tensorflow
