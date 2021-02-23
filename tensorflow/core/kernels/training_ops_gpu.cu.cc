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

#if TENSORFLOW_USE_ROCM

#include "rocm/include/hip/hip_complex.h"

#endif  // TENSORFLOW_USE_ROCM

// if any kernels involving complex sqrt/rsqrt are compiled with ROCm, build
// process completes without errors,but the resulting executable ends up
// unusable (throwing errors "no device code available for function" for
/// completely unrelated kernels.)
// We also can't cast to hipFloatComplex etc. because (as of 2020-01) HIP does
// not provide sqrt for complex.
// We have no choice but to implement sqrt and rsqrt by hand
template <typename T>
__device__ T impl_sqrt(T x) {
  return sqrt(x);
}
template <typename T>
__device__ T impl_rsqrt(T x) {
  return rsqrt(x);
}
template <>
__device__ Eigen::half impl_sqrt(Eigen::half x) {
  return __float2half(sqrt(__half2float(x)));
}
template <>
__device__ Eigen::half impl_rsqrt(Eigen::half x) {
  return __float2half(rsqrt(__half2float(x)));
}

template <class T>
__device__ std::complex<T> impl_sqrt(std::complex<T> x) {
  T re = x.real(), im = x.imag();
  T mod_x = sqrt(re * re + im * im);
  const T root2 = 0.7071067811865475;
  // We pick the root with the same sign of the imaginary component as
  // the input.
  T root[2] = {T(sqrt(mod_x + re) * root2),
               T(sqrt(mod_x - re) * root2 * (im >= 0 ? 1. : -1.))};
  // hcc/clang is really weird with its support of complex in device code;
  // for some reason it does not permit a 2-argument constructor
  return *(reinterpret_cast<std::complex<T>*>(&root));
}

template <class T>
__device__ T rsqrt_helper(T x) {
  return 0.5 * x + 0.125 * x * x + 0.0625 * x * x * x;
}

template <class T>
__device__ std::complex<T> impl_rsqrt(std::complex<T> x) {
  T re = x.real(), im = x.imag();
  T r = rsqrt(re * re + im * im);
  T ar2 = re * r * r;
  const T root2 = 0.7071067811865475;
  T root[2];
  // With float, calculating 1+re*r and 1-re*r may result in excessive errors
  // due to subtraction of two close values. We have to get fancy
  root[0] = sqrt(r * ((std::is_same<T, float>::value && re * r < -0.98)
                          ? rsqrt_helper(im * im * r * r)
                          : max(T(0.0), 1 + re * r))) *
            root2;
  root[1] = sqrt(r * ((std::is_same<T, float>::value && re * r > 0.98)
                          ? rsqrt_helper(im * im * r * r)
                          : max(T(0.0), 1 - re * r))) *
            root2 * (im >= 0 ? -1. : 1.);
  return *(reinterpret_cast<std::complex<T>*>(&root));
}

template <typename T>
__device__ T impl_fabs(T x) {
  return fabs(x);
}
template <>
__device__ Eigen::half impl_fabs(Eigen::half x) {
  return __float2half(fabs(__half2float(x)));
}

template <typename T>
__device__ T impl_sign(T x) {
  return x == T(0) ? T(0) : x < T(0) ? T(-1) : T(1);
}

template <typename T, typename Tindex, bool has_epsilon>
__global__ __launch_bounds__(1024) void SparseApplyAdagradKernel(
    T* var, T* accum, const T* lr, const T* epsilon, const T* grad,
    const Tindex* indices, Tindex param_rows, Tindex updates_size,
    Tindex indices_size, bool update_slots) {
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
    const T lr_t = *lr;
    const T epsilon_t = *epsilon;

    if (update_slots) {
      accum_i += grad_i * grad_i;
    }
    if (has_epsilon) {
      var_i -= lr_t * grad_i / (sqrt(accum_i) + epsilon_t);
    } else {
      var_i -= lr_t * grad_i * impl_rsqrt(accum_i);
    }

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
  }
}

template <typename T, typename Tindex>
__global__ __launch_bounds__(1024) void SparseApplyProximalAdagradKernel(
    T* var, T* accum, const T* lr, const T* l1, const T* l2, const T* grad,
    const Tindex* indices, Tindex param_rows, Tindex updates_size,
    Tindex indices_size) {
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
    const T lr_t = *lr;
    const T l1_t = *l1;
    const T l2_t = *l2;

    accum_i += grad_i * grad_i;
    T learning_rate = lr_t * impl_rsqrt(accum_i);
    // compute v = w - lr * grad.
    T prox_var_i = var_i - grad_i * learning_rate;
    // compute sign(v) * max(|v| - lr * max(l1, 0), 0)
    var_i = (prox_var_i >= 0 ? T(1.) : T(-1.)) *
            max(abs(prox_var_i) - learning_rate * max(l1_t, T(0)), T(0)) /
            (T(1.) + l2_t * learning_rate);

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
  }
}

template <typename T, typename Tindex, bool has_l2_shrinkage>
__global__ void SparseApplyFtrlKernel(T* var, T* accum, T* linear, const T* lr,
                                      const T* l1, const T* l2,
                                      const T* l2_shrinkage, const T* lr_power,
                                      const T* grad, const Tindex* indices,
                                      Tindex param_rows, Tindex updates_size,
                                      Tindex indices_size,
                                      bool multiply_linear_by_lr) {
  const Tindex col_size = updates_size / indices_size;
  GPU_1D_KERNEL_LOOP(grad_index, updates_size) {
    const Tindex indices_row = grad_index / col_size;
    const Tindex param_row = indices[indices_row];
    if (param_row < 0 || param_row >= param_rows) {
      // Ignore indices that are out of range.
      continue;
    }

    // Compute the index of var and accum.
    const Tindex param_index = param_row * col_size + (grad_index % col_size);

    // Read variables.
    T var_i = var[param_index];
    T accum_i = accum[param_index];
    T linear_i = linear[param_index];
    const T grad_i = grad[grad_index];
    const T lr_t = *lr;
    const T l1_t = *l1;
    const T l2_t = *l2;
    const T lr_power_t = *lr_power;

    const T grad_shr_i =
        has_l2_shrinkage ? grad_i + static_cast<T>(2) * (*l2_shrinkage) * var_i
                         : grad_i;
    const T new_accum_i = accum_i + grad_i * grad_i;
    const bool lr_power_is_neg_half = lr_power_t == static_cast<T>(-0.5);
    const T pow_new_accum = lr_power_is_neg_half
                                ? sqrt(new_accum_i)
                                : pow(new_accum_i, -lr_power_t);
    const T pow_accum =
        lr_power_is_neg_half ? sqrt(accum_i) : pow(accum_i, -lr_power_t);
    T linear_change = grad_shr_i * lr_t - (pow_new_accum - pow_accum) * var_i;
    if (!multiply_linear_by_lr) {
      linear_change /= lr_t;
    }
    linear_i += linear_change;

    T l1_mult = l1_t;
    if (multiply_linear_by_lr) {
      l1_mult *= lr_t;
    }
    const T l1_reg_adjust = max(min(linear_i, l1_mult), -l1_mult);
    const T x = l1_reg_adjust - linear_i;
    T y = pow_new_accum + static_cast<T>(2) * l2_t * lr_t;
    if (!multiply_linear_by_lr) {
      y /= lr_t;
    }
    var_i = x / y;
    accum_i = new_accum_i;

    // Write update back to variables.
    var[param_index] = var_i;
    accum[param_index] = accum_i;
    linear[param_index] = linear_i;
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void ApplyAdamKernel(
    int32 data_dim, T* var, T* m, T* v, const T* const beta1_power_,
    const T* const beta2_power_, const T* const lr_, const T* const beta1_,
    const T* const beta2_, const T* const epsilon_, const T* grad,
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
__global__ __launch_bounds__(1024) void SparseApplyKerasMomentumKernel(
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
__global__ __launch_bounds__(1024) void ApplyAdagradKernel(GpuLaunchConfig cfg,
                                                           T* var, T* accum,
                                                           const T* lr,
                                                           const T* grad,
                                                           bool update_slots) {
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    if (update_slots) accum[i] += grad[i] * grad[i];
    var[i] -= lr[0] * grad[i] * impl_rsqrt(accum[i]);
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void ApplyAdagradV2Kernel(
    GpuLaunchConfig cfg, T* var, T* accum, const T* lr, const T* epsilon,
    const T* grad, bool update_slots) {
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    if (update_slots) accum[i] += grad[i] * grad[i];
    T update = grad[i] / (impl_sqrt(accum[i]) + epsilon[0]);
    var[i] -= lr[0] * update;
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void ApplyProximalAdagradKernel(
    GpuLaunchConfig cfg, T* var, T* accum, const T* lr, const T* l1,
    const T* l2, const T* grad) {
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    accum[i] += grad[i] * grad[i];
    T lr_scaled = lr[0] * impl_rsqrt(accum[i]);
    T prox_var = var[i] - grad[i] * lr_scaled;
    var[i] = impl_sign(prox_var) *
             max(impl_fabs(prox_var) - lr_scaled * max(l1[0], T(0.f)), T(0.f)) /
             (T(1.f) + l2[0] * lr_scaled);
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void ApplyAdadeltaKernel(
    GpuLaunchConfig cfg, T* var, T* accum, T* accum_update, const T* plr,
    const T* prho, const T* peps, const T* grad) {
  T rho = prho[0];
  T eps = peps[0];
  T lr = plr[0];
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    accum[i] = accum[i] * rho + grad[i] * grad[i] * (T(1.0) - rho);
    T update =
        impl_sqrt(accum_update[i] + eps) * grad[i] * impl_rsqrt(accum[i] + eps);
    var[i] -= update * lr;
    accum_update[i] = accum_update[i] * rho + update * update * (T(1.0) - rho);
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void ApplyRMSPropKernel(
    GpuLaunchConfig cfg, T* var, T* ms, T* mom, const T* plr, const T* prho,
    const T* pmomentum, const T* peps, const T* grad) {
  T rho = prho[0];
  T eps = peps[0];
  T lr = plr[0];
  T momentum = pmomentum[0];
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    ms[i] += (T(1.0) - rho) * (grad[i] * grad[i] - ms[i]);
    mom[i] = mom[i] * momentum + lr * grad[i] * impl_rsqrt(eps + ms[i]);
    var[i] -= mom[i];
  }
}

template <typename T>
__global__ __launch_bounds__(1024) void ApplyCenteredRMSPropKernel(
    GpuLaunchConfig cfg, T* var, T* mg, T* ms, T* mom, const T* plr,
    const T* prho, const T* pmomentum, const T* peps, const T* grad) {
  T rho = prho[0];
  T eps = peps[0];
  T lr = plr[0];
  T momentum = pmomentum[0];
  T one_minus_rho = T(1.0) - rho;
  GPU_1D_KERNEL_LOOP(i, cfg.virtual_thread_count) {
    ms[i] += one_minus_rho * (grad[i] * grad[i] - ms[i]);
    mg[i] += one_minus_rho * (grad[i] - mg[i]);
    T denom = (ms[i] - mg[i] * mg[i]) + eps;
    mom[i] = mom[i] * momentum + lr * grad[i] * impl_rsqrt(denom);
    var[i] -= mom[i];
  }
}

namespace kernel_forward {
bool to_pointers(bool x) { return x; }
template <class T>
typename T::PointerType to_pointers(T& x) {
  return x.data();
}
template <class T>
typename T::ConstPointerType to_pointers(const T& x) {
  return x.data();
}

template <typename T, typename... CallerArgs, typename... KernelArgs>
void wrap_kernel_call(void (*func)(KernelArgs...), const GPUDevice& d, T var,
                      CallerArgs... args) {
  int32 data_dim = var.dimension(0);
  auto config = GetGpuLaunchConfig(data_dim, d);
  TF_CHECK_OK(GpuLaunchKernel(func, config.block_count, config.thread_per_block,
                              0, d.stream(), config, var.data(),
                              to_pointers(args)...));
}
};  // namespace kernel_forward

using kernel_forward::wrap_kernel_call;

template <typename T>
struct ApplyAdagrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
#if TENSORFLOW_USE_ROCM
    wrap_kernel_call(ApplyAdagradKernel<T>, d, var, accum, lr, grad,
                     update_slots);
#else
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    var.device(d) -= lr.reshape(single).broadcast(bcast) * grad * accum.rsqrt();
#endif
  }
};

template <typename T>
struct ApplyAdagradV2<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool update_slots) {
#if TENSORFLOW_USE_ROCM
    wrap_kernel_call(ApplyAdagradV2Kernel<T>, d, var, accum, lr, epsilon, grad,
                     update_slots);
#else
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    if (update_slots) {
      accum.device(d) += grad.square();
    }
    const auto update =
        grad / (accum.sqrt() + epsilon.reshape(single).broadcast(bcast));
    var.device(d) -= lr.reshape(single).broadcast(bcast) * update;
#endif
  }
};

template <typename T, typename Tindex, bool has_epsilon>
struct SparseApplyAdagrad<GPUDevice, T, Tindex, has_epsilon> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar epsilon,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                    bool update_slots) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    if (grad_size == 0) {
      return Status::OK();
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(
        SparseApplyAdagradKernel<T, Tindex, has_epsilon>, config.block_count,
        config.thread_per_block, 0, d.stream(), var.data(), accum.data(),
        lr.data(), epsilon.data(), grad.data(), indices.data(), first_dim_size,
        grad_size, indices_size, update_slots);
  }
};

template <typename T>
struct ApplyProximalAdagrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar l1,
                  typename TTypes<T>::ConstScalar l2,
                  typename TTypes<T>::ConstFlat grad) {
#if TENSORFLOW_USE_ROCM
    wrap_kernel_call(ApplyProximalAdagradKernel<T>, d, var, accum, lr, l1, l2,
                     grad);
#else
    Eigen::array<typename TTypes<T>::Tensor::Index, 1> bcast;
    bcast[0] = grad.dimension(0);
    Eigen::Sizes<1> single;
    // Fobos update per paper with Adagrad learning rate.
    accum.device(d) += grad.square();
    // Adagrad learning rate.
    // The following is the GPU equivalent of the CPU version:
    // auto learning_rate = accum.constant(lr()) * accum.rsqrt();
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto l1_bcast = l1.reshape(single).broadcast(bcast);
    auto l2_bcast = l2.reshape(single).broadcast(bcast);
    auto learning_rate = lr_bcast * accum.rsqrt();
    auto prox_var = var;
    // compute v = w - lr * grad.
    prox_var.device(d) -= grad * learning_rate;
    // compute sign(v) * max(|v| - lr * max(l1, 0), 0)
    var.device(d) = prox_var.sign() *
                    (prox_var.abs() - learning_rate * l1_bcast.cwiseMax(T(0.f)))
                        .cwiseMax(T(0.f)) /
                    (var.constant(T(1.f)) + l2_bcast * learning_rate);
#endif
  }
};

template <typename T, typename Tindex>
struct SparseApplyProximalAdagrad<GPUDevice, T, Tindex> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar l1,
                    typename TTypes<T>::ConstScalar l2,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices,
                    int64 inner_dim) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    if (grad_size == 0) {
      return Status::OK();
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
    return GpuLaunchKernel(SparseApplyProximalAdagradKernel<T, Tindex>,
                           config.block_count, config.thread_per_block, 0,
                           d.stream(), var.data(), accum.data(), lr.data(),
                           l1.data(), l2.data(), grad.data(), indices.data(),
                           first_dim_size, grad_size, indices_size);
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
#if TENSORFLOW_USE_ROCM
    wrap_kernel_call(ApplyAdadeltaKernel<T>, d, var, accum, accum_update, lr,
                     rho, epsilon, grad);
#else
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
#endif
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
struct ApplyFtrlMultiplyLinearByLr<GPUDevice, T> {
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

    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto l1_lr_bcast = (l1 * lr).reshape(single).broadcast(bcast);
    auto l2_lr_bcast = (l2 * lr).reshape(single).broadcast(bcast);
    auto lr_power_bcast = -lr_power.reshape(single).broadcast(bcast);
    const auto two = static_cast<T>(2.0);

    auto new_accum = accum + grad.square();
    auto accum_power = accum.binaryExpr(lr_power_bcast,
                                        Eigen::internal::scalar_pow_op<T, T>());
    auto new_accum_power = new_accum.binaryExpr(
        lr_power_bcast, Eigen::internal::scalar_pow_op<T, T>());
    linear.device(d) += grad * lr_bcast - (new_accum_power - accum_power) * var;
    auto x = (l1_lr_bcast * linear.sign() - linear);
    auto y = new_accum_power + linear.constant(two) * l2_lr_bcast;
    auto pre_shrink = x / y;
    var.device(d) = (linear.abs() > l1_lr_bcast)
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
struct ApplyFtrlV2MultiplyLinearByLr<GPUDevice, T> {
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

    auto l2_shrinkage_bcast = l2_shrinkage.reshape(single).broadcast(bcast);
    auto lr_bcast = lr.reshape(single).broadcast(bcast);
    auto l1_lr_bcast = (l1 * lr).reshape(single).broadcast(bcast);
    auto l2_lr_bcast = (l2 * lr).reshape(single).broadcast(bcast);
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
        grad_with_shrinkage * lr_bcast - (new_accum_power - accum_power) * var;
    auto x = (l1_lr_bcast * linear.sign() - linear);
    auto y = new_accum_power + linear.constant(two) * l2_lr_bcast;
    auto pre_shrink = x / y;
    var.device(d) = (linear.abs() > l1_lr_bcast)
                        .select(pre_shrink, var.constant(static_cast<T>(0)));
    accum.device(d) += grad.square();
  }
};

template <typename T, typename Tindex, bool has_l2_shrinkage>
struct SparseApplyFtrl<GPUDevice, T, Tindex, has_l2_shrinkage> {
  Status operator()(const GPUDevice& d, typename TTypes<T>::Matrix var,
                    typename TTypes<T>::Matrix accum,
                    typename TTypes<T>::Matrix linear,
                    typename TTypes<T>::ConstScalar lr,
                    typename TTypes<T>::ConstScalar l1,
                    typename TTypes<T>::ConstScalar l2,
                    typename TTypes<T>::ConstScalar l2_shrinkage,
                    typename TTypes<T>::ConstScalar lr_power,
                    typename TTypes<T>::ConstMatrix grad,
                    typename TTypes<Tindex>::ConstVec indices, int64 inner_dim,
                    bool multiply_linear_by_lr) {
    const Tindex first_dim_size = var.dimension(0);
    const Tindex grad_size = grad.size();
    const Tindex indices_size = indices.size();
    if (grad_size == 0) {
      return Status::OK();
    }
    // The simpler overload of GetGpuLaunchConfig() would result in a "too many
    // resources requested for launch" error.
    auto* device_func = SparseApplyFtrlKernel<T, Tindex, has_l2_shrinkage>;
    GpuLaunchConfig config =
        GetGpuLaunchConfig(grad_size, d, device_func, 0, 0);
    return GpuLaunchKernel(
        device_func, config.block_count, config.thread_per_block, 0, d.stream(),
        /*var=*/var.data(),
        /*accum=*/accum.data(),
        /*linear=*/linear.data(), /*lr=*/lr.data(), /*l1=*/l1.data(),
        /*l2=*/l2.data(), /*l2_shrinkage=*/l2_shrinkage.data(),
        /*lr_power=*/lr_power.data(), /*grad=*/grad.data(),
        /*indices=*/indices.data(), /*param_rows=*/first_dim_size,
        /*updates_size=*/grad_size,
        /*indices_size=*/indices_size,
        /*multiply_linear_by_lr=*/multiply_linear_by_lr);
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
    if (grad_size != 0) {
      GpuLaunchConfig config = GetGpuLaunchConfig(grad_size, d);
      TF_CHECK_OK(GpuLaunchKernel(
          SparseApplyKerasMomentumKernel<T, Tindex>, config.block_count,
          config.thread_per_block, 0, d.stream(), var.data(), accum.data(),
          lr.data(), grad.data(), indices.data(), momentum.data(), use_nesterov,
          first_dim_size, grad_size, indices_size));
    }
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
#if TENSORFLOW_USE_ROCM
    wrap_kernel_call(ApplyRMSPropKernel<T>, d, var, ms, mom, lr, rho, momentum,
                     epsilon, grad);
#else
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
#endif
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
#if TENSORFLOW_USE_ROCM
    wrap_kernel_call(ApplyCenteredRMSPropKernel<T>, d, var, mg, ms, mom, lr,
                     rho, momentum, epsilon, grad);
#else
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
#endif
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
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support
                             // complex sqrt
template struct functor::ApplyGradientDescent<GPUDevice, complex64>;
template struct functor::ApplyGradientDescent<GPUDevice, complex128>;
#endif

template struct functor::ApplyAdagrad<GPUDevice, Eigen::half>;
template struct functor::ApplyAdagrad<GPUDevice, float>;
template struct functor::ApplyAdagrad<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support
                             // complex sqrt
template struct functor::ApplyAdagrad<GPUDevice, complex64>;
template struct functor::ApplyAdagrad<GPUDevice, complex128>;
#endif

template struct functor::ApplyAdagradV2<GPUDevice, Eigen::half>;
template struct functor::ApplyAdagradV2<GPUDevice, float>;
template struct functor::ApplyAdagradV2<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support
                             // complex sqrt
template struct functor::ApplyAdagradV2<GPUDevice, complex64>;
template struct functor::ApplyAdagradV2<GPUDevice, complex128>;
#endif

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                             \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int32,    \
                                              /*has_epsilon=*/false>; \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int64,    \
                                              /*has_epsilon=*/false>; \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int32,    \
                                              /*has_epsilon=*/true>;  \
  template struct functor::SparseApplyAdagrad<GPUDevice, T, int64,    \
                                              /*has_epsilon=*/true>
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

template struct functor::ApplyProximalAdagrad<GPUDevice, Eigen::half>;
template struct functor::ApplyProximalAdagrad<GPUDevice, float>;
template struct functor::ApplyProximalAdagrad<GPUDevice, double>;

template struct functor::SparseApplyProximalAdagrad<GPUDevice, Eigen::half,
                                                    int32>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, Eigen::half,
                                                    int64>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, float, int32>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, float, int64>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, double, int32>;
template struct functor::SparseApplyProximalAdagrad<GPUDevice, double, int64>;

template struct functor::ApplyAdadelta<GPUDevice, Eigen::half>;
template struct functor::ApplyAdadelta<GPUDevice, float>;
template struct functor::ApplyAdadelta<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support
                             // complex sqrt
template struct functor::ApplyAdadelta<GPUDevice, complex64>;
template struct functor::ApplyAdadelta<GPUDevice, complex128>;
#endif

template struct functor::ApplyFtrl<GPUDevice, Eigen::half>;
template struct functor::ApplyFtrl<GPUDevice, float>;
template struct functor::ApplyFtrl<GPUDevice, double>;

template struct functor::ApplyFtrlMultiplyLinearByLr<GPUDevice, Eigen::half>;
template struct functor::ApplyFtrlMultiplyLinearByLr<GPUDevice, float>;
template struct functor::ApplyFtrlMultiplyLinearByLr<GPUDevice, double>;

template struct functor::ApplyFtrlV2<GPUDevice, Eigen::half>;
template struct functor::ApplyFtrlV2<GPUDevice, float>;
template struct functor::ApplyFtrlV2<GPUDevice, double>;

template struct functor::ApplyFtrlV2MultiplyLinearByLr<GPUDevice, Eigen::half>;
template struct functor::ApplyFtrlV2MultiplyLinearByLr<GPUDevice, float>;
template struct functor::ApplyFtrlV2MultiplyLinearByLr<GPUDevice, double>;

#define EXPLICITLY_INSTANTIATE_FUNCTOR(T)                               \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int32,         \
                                           /*has_l2_shrinkage=*/false>; \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int64,         \
                                           /*has_l2_shrinkage=*/false>; \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int32,         \
                                           /*has_l2_shrinkage=*/true>;  \
  template struct functor::SparseApplyFtrl<GPUDevice, T, int64,         \
                                           /*has_l2_shrinkage=*/true>
EXPLICITLY_INSTANTIATE_FUNCTOR(Eigen::half);
EXPLICITLY_INSTANTIATE_FUNCTOR(float);
EXPLICITLY_INSTANTIATE_FUNCTOR(double);
#undef EXPLICITLY_INSTANTIATE_FUNCTOR

template struct functor::ApplyMomentum<GPUDevice, Eigen::half>;
template struct functor::ApplyMomentum<GPUDevice, float>;
template struct functor::ApplyMomentum<GPUDevice, double>;
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
template struct functor::ApplyMomentum<GPUDevice, complex64>;
template struct functor::ApplyMomentum<GPUDevice, complex128>;
#endif

template struct functor::ApplyKerasMomentum<GPUDevice, Eigen::half>;
template struct functor::ApplyKerasMomentum<GPUDevice, float>;
template struct functor::ApplyKerasMomentum<GPUDevice, double>;
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
template struct functor::ApplyKerasMomentum<GPUDevice, complex64>;
template struct functor::ApplyKerasMomentum<GPUDevice, complex128>;
#endif

template struct functor::SparseApplyKerasMomentum<GPUDevice, Eigen::half,
                                                  int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, Eigen::half,
                                                  int64>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, float, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, float, int64>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, double, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, double, int64>;
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex64, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex64, int64>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex128, int32>;
template struct functor::SparseApplyKerasMomentum<GPUDevice, complex128, int64>;
#endif

template struct functor::ApplyAdam<GPUDevice, Eigen::half>;
template struct functor::ApplyAdam<GPUDevice, float>;
template struct functor::ApplyAdam<GPUDevice, double>;
#if !defined(TENSORFLOW_USE_NVCC) && \
    !defined(TENSORFLOW_USE_ROCM)  // TODO(b/143684500): Eigen to support
                                   // complex sqrt
template struct functor::ApplyAdam<GPUDevice, complex64>;
template struct functor::ApplyAdam<GPUDevice, complex128>;
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
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support
                             // complex sqrt
template struct functor::ApplyRMSProp<GPUDevice, complex64>;
template struct functor::ApplyRMSProp<GPUDevice, complex128>;
#endif

template struct functor::ApplyCenteredRMSProp<GPUDevice, Eigen::half>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, float>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, double>;
#ifndef TENSORFLOW_USE_NVCC  // TODO(b/143684500): Eigen to support
                             // complex sqrt
template struct functor::ApplyCenteredRMSProp<GPUDevice, complex64>;
template struct functor::ApplyCenteredRMSProp<GPUDevice, complex128>;
#endif

template struct functor::ApplyAddSign<GPUDevice, Eigen::half>;
template struct functor::ApplyAddSign<GPUDevice, float>;
template struct functor::ApplyAddSign<GPUDevice, double>;

template struct functor::ApplyPowerSign<GPUDevice, Eigen::half>;
template struct functor::ApplyPowerSign<GPUDevice, float>;
template struct functor::ApplyPowerSign<GPUDevice, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
