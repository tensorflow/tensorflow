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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/training_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
__global__ void ApplyAdamKernel(int32 data_dim, T* var, T* m, T* v,
                                const T* const beta1_power_,
                                const T* const beta2_power_, const T* const lr_,
                                const T* const beta1_, const T* const beta2_,
                                const T* const epsilon_, const T* grad,
                                bool use_nesterov) {
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(gridDim.y == 1);
  eigen_assert(gridDim.z == 1);

  const T mul_factor = (*lr_) * sqrt(static_cast<T>(1.0) - (*beta2_power_)) /
                       (static_cast<T>(1.0) - (*beta1_power_));
  const T epsilon = (*epsilon_);
  const T beta1 = (*beta1_);
  const T one_minus_beta1 = static_cast<T>(1.0) - (beta1);
  const T one_minus_beta2 = static_cast<T>(1.0) - (*beta2_);
  const int32 stripe = gridDim.x * blockDim.x;

  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < data_dim;
       i += stripe) {
    auto m_i = m[i];
    auto g_i = grad[i];
    auto v_i = v[i];

    m_i += one_minus_beta1 * (g_i - m_i);
    v_i += one_minus_beta2 * (g_i * g_i - v_i);
    if (use_nesterov) {
      var[i] -= mul_factor * (m_i * beta1 + one_minus_beta1 * g_i) /
                (epsilon + sqrt(v_i));
    } else {
      var[i] -= mul_factor * m_i / (epsilon + sqrt(v_i));
    }

    m[i] = m_i;
    v[i] = v_i;
  }
}

template <typename T, typename Tindex>
__global__ void SparseApplyKerasMomentumKernel(
    T* var, T* accum, const T* lr, const T* grad, const Tindex* indices,
    const T* momentum, bool use_nesterov, Tindex param_rows,
    Tindex updates_size, Tindex indices_size) {
  Tindex col_size = updates_size / indices_size;
  GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
    Tindex indices_row = grad_index / col_size;
    Tindex param_row = indices[indices_row];
    if (param_row < 0 || param_row >= param_rows) {
      // Ignore indices that are out of range.
      continue;
    }

    // Compute the index of var and accum.
    Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var[param_index];
    T accum_i = accum[param_index];
    T grad_i = grad[grad_index];
    const T momentum_t = *momentum;
    const T lr_t = *lr;

    // Variable update computation.
    accum_i = momentum_t * accum_i - lr_t * grad_i;
    // static branching in cuda does not impact performance.
    if (use_nesterov) {
      var_i += (momentum_t * accum_i - lr_t * grad_i);
    } else {
      var_i += accum_i;
    }

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
  }
}

template <typename T>
struct ApplyGradientDescent<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad;
  }
};

template <typename T>
struct ApplyAdagrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad * accum.rsqrt();
  }
};

template <typename T>
struct ApplyAdagradV2<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    const auto update =
        grad / (accum.sqrt() + epsilon.reshape(single).broadcast(bcast));
    var.device(d) -= lr.reshape(single).broadcast(bcast) * update;
  }
};

template <typename T>
struct ApplyAdadelta<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat accum_update,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    accum.device(d) = accum * rho.reshape(single).broadcast(bcast) +
                      grad.square() * (grad.constant(T(1)) -
                                       rho.reshape(single).broadcast(bcast));
    const auto update =
        (accum_update + epsilon.reshape(single).broadcast(bcast)).sqrt() *
        (accum + epsilon.reshape(single).broadcast(bcast)).rsqrt() * grad;
    var.device(d) -= update * lr.reshape(single).broadcast(bcast);
    accum_update.device(d) =
        accum_update * rho.reshape(single).broadcast(bcast) +
        update.square() *
            (grad.constant(T(1)) - rho.reshape(single).broadcast(bcast));
  }
};

template <typename T>
struct ApplyFtrl<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar lr_power) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    auto l1_bcast = l1.reshape(single).broadcast(bcast);
    auto l2_bcast = l2.reshape(single).broadcast(bcast);
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto lr_power_bcast = -lr_power.reshape(single).broadcast(bcast);
    const auto two = static_cast<T>(2.0);

    auto new_accum = accum + grad.square();
    auto accum_power = accum.binaryExpr(lr_power_bcast,
                                        Eigen::internal::scalar_pow_op<T, T>());
    auto new_accum_power = new_accum.binaryExpr(
        lr_power_bcast, Eigen::internal::scalar_pow_op<T, T>());
    linear.device(d) += grad - (new_accum_power - accum_power) * var / lr_bcast;
    auto x = (l1_bcast * linear.sign() - linear);
    auto y = (new_accum_power / lr_bcast) + linear.constant(two) * l2_bcast;
    auto pre_shrink = x / y;
    var.device(d) = (linear.abs() > l1_bcast)
                        .select(pre_shrink, var.constant(static_cast<T>(0)));
    accum.device(d) += grad.square();
  }
};

template <typename T>
struct ApplyFtrlV2<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::Flat linear,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstScalar l2_shrinkage,
                  typename TTypes<T>::ConstScalar lr_power) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    auto l1_bcast = l1.reshape(single).broadcast(bcast);
    auto l2_bcast = l2.reshape(single).broadcast(bcast);
    auto l2_shrinkage_bcast = l2_shrinkage.reshape(single).broadcast(bcast);
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto lr_power_bcast = -lr_power.reshape(single).broadcast(bcast);
    const auto two = static_cast<T>(2.0);

    auto new_accum = accum + grad.square();
    auto accum_power = accum.binaryExpr(lr_power_bcast,
                                        Eigen::internal::scalar_pow_op<T, T>());
    auto new_accum_power = new_accum.binaryExpr(
        lr_power_bcast, Eigen::internal::scalar_pow_op<T, T>());
    auto grad_with_shrinkage =
        grad + (var.constant(two) * l2_shrinkage_bcast * var);
    linear.device(d) +=
        grad_with_shrinkage - (new_accum_power - accum_power) * var / lr_bcast;
    auto x = (l1_bcast * linear.sign() - linear);
    auto y = (new_accum_power / lr_bcast) + linear.constant(two) * l2_bcast;
    auto pre_shrink = x / y;
    var.device(d) = (linear.abs() > l1_bcast)
                        .select(pre_shrink, var.constant(static_cast<T>(0)));
    accum.device(d) += grad.square();
  }
};

template <typename T>
struct ApplyMomentum<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    accum.device(d) = accum * momentum.reshape(single).broadcast(bcast) + grad;
    if (use_nesterov) {
      var.device(d) -= grad * lr.reshape(single).broadcast(bcast) +
                       accum * momentum.reshape(single).broadcast(bcast) *
                           lr.reshape(single).broadcast(bcast);
    } else {
      var.device(d) -= lr.reshape(single).broadcast(bcast) * accum;
    }
  }
};

template <typename T>
struct ApplyKerasMomentum<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum, bool use_nesterov) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    accum.device(d) = (accum * momentum.reshape(single).broadcast(bcast) -
                       grad * lr.reshape(single).broadcast(bcast));
    if (use_nesterov) {
      var.device(d) += (accum * momentum.reshape(single).broadcast(bcast) -
                        grad * lr.reshape(single).broadcast(bcast));
    } else {
      var.device(d) += accum;
    }
  }
};

template <typename T, typename Tindex>
struct SparseApplyKerasMomentum<GPUDevice, T, Tindex> {
  Tindex operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices,
                    typename TTypes<T>::ConstScalar momentum,
                    bool use_nesterov) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    TF_CHECK_OK(GpuLaunchKernel(
        SparseApplyKerasMomentumKernel<T, Tindex>, config.block_count,
        config.thread_per_block, 0, d.stream(), var.data(), accum.data(),
        lr.data(), grad.data(), indices.data(), momentum.data(), use_nesterov,
        first_dim_size, grad_size, indices_size));
    return static_cast<Tindex>(-1);
  }
};

template <typename T>
struct ApplyAdam<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
    int32 data_dim = grad.dimension(0);
    GpuLaunchConfig config = GetGpuLaunchConfig(data_dim, d);
    eigen_assert(static_cast<int64>(grad.dimension(0)) +
                     static_cast<int64>(config.block_count) *
                         static_cast<int64>(config.thread_per_block) <
                 std::numeric_limits<int32>::max());

    TF_CHECK_OK(GpuLaunchKernel(
        ApplyAdamKernel<T>, config.block_count, config.thread_per_block, 0,
        d.stream(), data_dim, var.data(), m.data(), v.data(),
        beta1_power.data(), beta2_power.data(), lr.data(), beta1.data(),
        beta2.data(), epsilon.data(), grad.data(), use_nesterov));
  }
};

template <typename T>
struct ApplyAdamWithAmsgrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::Flat vhat,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) =
        m + (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
                (grad - m);
    v.device(d) =
        v + (beta2.constant(one) - beta2).reshape(single).broadcast(bcast) *
                (grad.square() - v);
    vhat.device(d) = vhat.cwiseMax(v);

    var.device(d) -= (lr * (beta2_power.constant(one) - beta2_power).sqrt() /
                      (beta1_power.constant(one) - beta1_power))
                         .reshape(single)
                         .broadcast(bcast) *
                     m /
                     (epsilon.reshape(single).broadcast(bcast) + vhat.sqrt());
  }
};

template <typename T>
struct ApplyAdaMax<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    m.device(d) +=
        (beta1.constant(one) - beta1).reshape(single).broadcast(bcast) *
        (grad - m);
    v.device(d) =
        (beta2.reshape(single).broadcast(bcast) * v).cwiseMax(grad.abs());
    var.device(d) -= lr.reshape(single).broadcast(bcast) /
                     (beta1_power.constant(one) - beta1_power)
                         .reshape(single)
                         .broadcast(bcast) *
                     (m / (v + epsilon.reshape(single).broadcast(bcast)));
  }
};

template <typename T>
struct ApplyRMSProp<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    ms.device(d) =
        ms + (rho.constant(one) - rho).reshape(single).broadcast(bcast) *
                 (grad.square() - ms);
    mom.device(d) =
        mom * momentum.reshape(single).broadcast(bcast) +
        lr.reshape(single).broadcast(bcast) * grad /
            ((epsilon.reshape(single).broadcast(bcast) + ms).sqrt());
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyCenteredRMSProp<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat mg, typename TTypes<T>::Flat ms,
                  typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    const auto one = static_cast<T>(1.0);
    const auto one_minus_rho =
        (rho.constant(one) - rho).reshape(single).broadcast(bcast);
    ms.device(d) = ms + one_minus_rho * (grad.square() - ms);
    mg.device(d) = mg + one_minus_rho * (grad - mg);
    auto denom = (ms - mg.square()) + epsilon.reshape(single).broadcast(bcast);
    mom.device(d) = mom * momentum.reshape(single).broadcast(bcast) +
                    lr.reshape(single).broadcast(bcast) * grad / denom.sqrt();
    var.device(d) -= mom;
  }
};

template <typename T>
struct ApplyAddSign<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar alpha,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    // The following is the GPU equivalent of the CPU version:
    // m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    const auto one = static_cast<T>(1.0);
    auto beta_bcast = beta.reshape(single).broadcast(bcast);
    auto one_minus_beta =
        (beta.constant(one) - beta).reshape(single).broadcast(bcast);
    m.device(d) = m * beta_bcast + grad * one_minus_beta;

    // The following is the GPU equivalent of the CPU version:
    // var.device(d) -= lr() * (alpha() + sign_decay() * sign_gm) * grad;
    auto sign_gm = grad.sign() * m.sign();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto alpha_bcast = alpha.reshape(single).broadcast(bcast);
    auto sign_decay_bcast = sign_decay.reshape(single).broadcast(bcast);
    var.device(d) -=
        lr_bcast * (alpha_bcast + sign_decay_bcast * sign_gm) * grad;
  }
};

template <typename T>
struct ApplyPowerSign<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar logbase,
                  typename TTypes<T>::ConstScalar sign_decay,
                  typename TTypes<T>::ConstScalar beta,
                  typename TTypes<T>::ConstFlat grad) {
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;

    // The following is the GPU equivalent of the CPU version:
    // m.device(d) = m * beta() + grad * (static_cast<T>(1) - beta());
    const auto one = static_cast<T>(1.0);
    auto beta_bcast = beta.reshape(single).broadcast(bcast);
    auto one_minus_beta =
        (beta.constant(one) - beta).reshape(single).broadcast(bcast);
    m.device(d) = m * beta_bcast + grad * one_minus_beta;

    // The following is the GPU equivalent of the CPU version:
    // auto grad_scale = (logbase() * sign_decay() * sign_gm).exp();
    // var.device(d) -= lr() * grad_scale * grad;
    auto sign_gm = grad.sign() * m.sign();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto logbase_bcast = logbase.reshape(single).broadcast(bcast);
    auto sign_decay_bcast = sign_decay.reshape(single).broadcast(bcast);
    auto grad_scale = (logbase_bcast * sign_decay_bcast * sign_gm).exp();
    var.device(d) -= lr_bcast * grad_scale * grad;
  }
};

}  // namespace functor

template struct functor::ApplyGradientDescent<GPUDevice, Eigen::half>;
template struct functor::ApplyGradientDescent<GPUDevice, float>;
template struct functor::ApplyGradientDescent<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyGradientDescent<GPUDevice, complex64>;
template struct functor::ApplyGradientDescent<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyAdagrad<GPUDevice, Eigen::half>;
template struct functor::ApplyAdagrad<GPUDevice, float>;
template struct functor::ApplyAdagrad<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyAdagrad<GPUDevice, complex64>;
template struct functor::ApplyAdagrad<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyAdagradV2<GPUDevice, Eigen::half>;
template struct functor::ApplyAdagradV2<GPUDevice, float>;
template struct functor::ApplyAdagradV2<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyAdagradV2<GPUDevice, complex64>;
template struct functor::ApplyAdagradV2<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyAdadelta<GPUDevice, Eigen::half>;
template struct functor::ApplyAdadelta<GPUDevice, float>;
template struct functor::ApplyAdadelta<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyAdadelta<GPUDevice, complex64>;
template struct functor::ApplyAdadelta<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyFtrl<GPUDevice, Eigen::half>;
template struct functor::ApplyFtrl<GPUDevice, float>;
template struct functor::ApplyFtrl<GPUDevice, double>;

template struct functor::ApplyFtrlV2<GPUDevice, Eigen::half>;
template struct functor::ApplyFtrlV2<GPUDevice, float>;
template struct functor::ApplyFtrlV2<GPUDevice, double>;

template struct functor::ApplyMomentum<GPUDevice, Eigen::half>;
template struct functor::ApplyMomentum<GPUDevice, float>;
template struct functor::ApplyMomentum<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyMomentum<GPUDevice, complex64>;
template struct functor::ApplyMomentum<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyKerasMomentum<GPUDevice, Eigen::half>;
template struct functor::ApplyKerasMomentum<GPUDevice, float>;
template struct functor::ApplyKerasMomentum<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyKerasMomentum<GPUDevice, complex64>;
template struct functor::ApplyKerasMomentum<GPUDevice, complex128>;
#endif
#endif

template struct functor::SparseApplyKerasMomentum<GPUDevice, Eigen::half,
                                                  int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, Eigen::half,
                                                  int64>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, float, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, float, int64>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, double, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, double, int64>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex64, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex64, int64>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex128, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex128, int64>;
#endif
#endif

template struct functor::ApplyAdam<GPUDevice, Eigen::half>;
template struct functor::ApplyAdam<GPUDevice, float>;
template struct functor::ApplyAdam<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyAdam<GPUDevice, complex64>;
template struct functor::ApplyAdam<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyAdamWithAmsgrad<GPUDevice, Eigen::half>;
template struct functor::ApplyAdamWithAmsgrad<GPUDevice, float>;
template struct functor::ApplyAdamWithAmsgrad<GPUDevice, double>;

template struct functor::ApplyAdaMax<GPUDevice, Eigen::half>;
template struct functor::ApplyAdaMax<GPUDevice, float>;
template struct functor::ApplyAdaMax<GPUDevice, double>;

template struct functor::ApplyRMSProp<GPUDevice, Eigen::half>;
template struct functor::ApplyRMSProp<GPUDevice, float>;
template struct functor::ApplyRMSProp<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyRMSProp<GPUDevice, complex64>;
template struct functor::ApplyRMSProp<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyCenteredRMSProp<GPUDevice, Eigen::half>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, float>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support complex sqrt
#ifndef PLATFORM_WINDOWS
template struct functor::ApplyCenteredRMSProp<GPUDevice, complex64>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, complex128>;
#endif
#endif

template struct functor::ApplyAddSign<GPUDevice, Eigen::half>;
template struct functor::ApplyAddSign<GPUDevice, float>;
template struct functor::ApplyAddSign<GPUDevice, double>;

template struct functor::ApplyPowerSign<GPUDevice, Eigen::half>;
template struct functor::ApplyPowerSign<GPUDevice, float>;
template struct functor::ApplyPowerSign<GPUDevice, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
