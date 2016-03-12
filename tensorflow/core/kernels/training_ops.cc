/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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
struct ApplyAdagrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad) {
    accum.device(d) += grad.square();
    var.device(d) -= grad * lr() * accum.rsqrt();
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
    if (lr_power() == -0.5) {
      linear.device(d) += grad - (new_accum.sqrt() - accum.sqrt()) / lr() * var;
    } else {
      linear.device(d) +=
          grad -
          (new_accum.pow(-lr_power()) - accum.pow(-lr_power())) / lr() * var;
    }
    auto x = (linear.constant(l1()) * linear.sign() - linear);
    if (lr_power() == -0.5) {
      auto y = new_accum.sqrt() / new_accum.constant(lr()) +
               linear.constant(2 * l2());
      var.device(d) = x / y;
    } else {
      auto y = new_accum.pow(-lr_power()) / new_accum.constant(lr()) +
               linear.constant(2 * l2());
      var.device(d) = x / y;
    }
    var.device(d) =
        (linear.abs() > linear.constant(l1())).select(var, var.constant(0));
    accum.device(d) += grad.square();
  }
};

template <typename T>
struct ApplyMomentum<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat accum,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstFlat grad,
                  typename TTypes<T>::ConstScalar momentum) {
    accum.device(d) = accum * momentum() + grad;
    var.device(d) -= accum * lr();
  }
};

template <typename T>
struct ApplyAdam<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat m, typename TTypes<T>::Flat v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    const T alpha = lr() * std::sqrt(1 - beta2_power()) / (1 - beta1_power());
    m.device(d) += (grad - m) * (1 - beta1());
    v.device(d) += (grad.square() - v) * (1 - beta2());
    var.device(d) -= (m * alpha) / (v.sqrt() + epsilon());
  }
};

template <typename T>
struct ApplyRMSProp<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat var,
                  typename TTypes<T>::Flat ms, typename TTypes<T>::Flat mom,
                  typename TTypes<T>::ConstScalar lr,
                  typename TTypes<T>::ConstScalar rho,
                  typename TTypes<T>::ConstScalar momentum,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad) {
    ms.device(d) += (grad.square() - ms) * (1 - rho());
    mom.device(d) =
        mom * momentum() + (grad * lr()) / ((ms + epsilon()).sqrt());
    var.device(d) -= mom;
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
    if (use_exclusive_lock_) {
      mutex_lock l(*ctx->input_ref_mutex(0));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    const Tensor& alpha = ctx->input(1);
    OP_REQUIRES(ctx, IsLegacyScalar(alpha.shape()),
                errors::InvalidArgument("alpha is not a scalar: ",
                                        alpha.shape().DebugString()));
    const Tensor& delta = ctx->input(2);
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(delta.shape()),
        errors::InvalidArgument("var and delta do not have the same shape",
                                var.shape().DebugString(), " ",
                                delta.shape().DebugString()));
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    const Tensor& alpha = ctx->input(1);
    const Tensor& delta = ctx->input(2);
    functor::ApplyGradientDescent<Device, T>()(
        device, var.flat<T>(), alpha.scalar<T>(), delta.flat<T>());
  }
};

#define REGISTER_KERNELS(D, T)                                                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ApplyGradientDescent").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyGradientDescentOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                             \
  template <>                                           \
  void ApplyGradientDescent<GPUDevice, T>::operator()(  \
      const GPUDevice& d, typename TTypes<T>::Flat var, \
      typename TTypes<T>::ConstScalar alpha,            \
      typename TTypes<T>::ConstFlat delta);             \
  extern template struct ApplyGradientDescent<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdagradOp : public OpKernel {
 public:
  explicit ApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (use_exclusive_lock_) {
      mutex_lock l1(*ctx->input_ref_mutex(0));
      // Don't try to acquire a lock on the second ref as they share the same
      // mutex.
      //
      // mutex_lock l2(*ctx->input_ref_mutex(1));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
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
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    const Tensor& lr = ctx->input(2);
    const Tensor& grad = ctx->input(3);
    functor::ApplyAdagrad<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                       lr.scalar<T>(), grad.flat<T>());
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyAdagrad").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdagradOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyAdagrad<GPUDevice, T>::operator()(                            \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad);                                \
  extern template struct ApplyAdagrad<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_KERNELS

namespace {
template <class T>
inline T sgn(const T x) {
  return (x == 0 ? 0 : (x < 0 ? -1 : 1));
}

template <typename T>
inline T FtrlCompute(const T& accum, const T& linear, const T& lr, const T& l1,
                     const T& l2, const T& lr_power) {
  T quadratic;
  if (lr_power == -0.5) {
    quadratic = std::sqrt(accum) / lr + 2 * l2;
  } else {
    quadratic = pow(accum, -lr_power) / lr + 2 * l2;
  }
  if (std::abs(linear) > l1) {
    return (l1 * sgn(linear) - linear) / quadratic;
  } else {
    return static_cast<T>(0.0);
  }
}
}  // namespace

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyAdagradOp : public OpKernel {
 public:
  explicit SparseApplyAdagradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    mutex* mu_var = ctx->input_ref_mutex(0);
    // mu_accum is actually the same mutex as mu_var since currently we use a
    // global mutex.
    //
    // mutex* mu_accum = ctx->input_ref_mutex(1);
    if (use_exclusive_lock_) {
      mu_var->lock();
    }
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(accum.shape()),
        errors::InvalidArgument("var and accum do not have the same shape",
                                var.shape().DebugString(), " ",
                                accum.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(var.shape()),
                errors::InvalidArgument("var must be at least 1 dimensional"));

    const Tensor& lr = ctx->input(2);
    OP_REQUIRES(ctx, IsLegacyScalar(lr.shape()),
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

    if (N > 0) {
      if (inner_dim > 1) {
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
        auto accum_flat = accum.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();

        // Note(yonghui): It might be worth multi-threading square() and
        // rsqrt().
        for (Tindex i = 0; i < N; i++) {
          const Tindex index = indices_vec(i);
          auto a = accum_flat.template chip<0>(index);
          auto g = grad_flat.template chip<0>(i);
          auto v = var_flat.template chip<0>(index);
          a += g.square();
          v -= g.constant(lr_scalar) * g * a.rsqrt();
        }
      } else {
        CHECK_EQ(1, inner_dim);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = indices_vec(i);
          T& a = accum_flat(index);
          const T& g = grad_flat(i);
          a += g * g;
          var_flat(index) -= lr_scalar * g / std::sqrt(a);
        }
      }
    }
    if (use_exclusive_lock_) {
      mu_var->unlock();
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyAdagrad")                 \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyAdagradOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyFtrlOp : public OpKernel {
 public:
  explicit ApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (use_exclusive_lock_) {
      mutex_lock l1(*ctx->input_ref_mutex(0));
      // Don't try to acquire a lock on the second ref as they share the same
      // mutex.
      //
      // mutex_lock l2(*ctx->input_ref_mutex(1));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor linear = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

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
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) && lr.scalar<T>()() > 0,
                errors::InvalidArgument("lr is not a scalar or <= 0.0: ",
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
    const Tensor& lr_power = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power.shape()),
                errors::InvalidArgument("lr power is not a scalar: ",
                                        lr_power.shape().DebugString()));
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor linear = ctx->mutable_input(2, use_exclusive_lock_);
    const Tensor& grad = ctx->input(3);
    const Tensor& lr = ctx->input(4);
    const Tensor& l1 = ctx->input(5);
    const Tensor& l2 = ctx->input(6);
    const Tensor& lr_power = ctx->input(7);
    functor::ApplyFtrl<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                    linear.flat<T>(), grad.flat<T>(),
                                    lr.scalar<T>(), l1.scalar<T>(),
                                    l2.scalar<T>(), lr_power.scalar<T>());
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyFtrl").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyFtrlOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename Device, typename T, typename Tindex>
class SparseApplyFtrlOp : public OpKernel {
 public:
  explicit SparseApplyFtrlOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    mutex* mu_var = ctx->input_ref_mutex(0);
    // mu_accum is actually the same mutex as mu_var since currently we use a
    // global mutex.
    //
    // mutex* mu_accum = ctx->input_ref_mutex(1);
    if (use_exclusive_lock_) {
      mu_var->lock();
    }
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor linear = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, linear.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));
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

    const Tensor& lr = ctx->input(5);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(lr.shape()) && lr.scalar<T>()() > 0,
                errors::InvalidArgument("lr is not a scalar or <= 0.0: ",
                                        lr.shape().DebugString()));

    const Tensor& l1 = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l1.shape()),
                errors::InvalidArgument("l1 is not a scalar: ",
                                        l1.shape().DebugString()));
    const Tensor& l2 = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(l2.shape()),
                errors::InvalidArgument("l2 is not a scalar: ",
                                        l2.shape().DebugString()));
    const Tensor& lr_power = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_power.shape()),
                errors::InvalidArgument("lr_power is not a scalar: ",
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

    if (N > 0) {
      if (inner_dim > 1) {
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
        auto accum_flat = accum.flat_outer_dims<T>();
        auto linear_flat = linear.flat_outer_dims<T>();
        auto grad_flat = grad.flat_outer_dims<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T lr_power_scalar = lr_power.scalar<T>()();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = indices_vec(i);
          auto accum = accum_flat.template chip<0>(index);
          auto linear = linear_flat.template chip<0>(index);
          auto grad = grad_flat.template chip<0>(i);
          auto var = var_flat.template chip<0>(index);

          auto new_accum = accum + grad.square();
          if (lr_power_scalar == -0.5) {
            linear +=
                grad - (new_accum.sqrt() - accum.sqrt()) / lr_scalar * var;
          } else {
            linear += grad -
                      (new_accum.pow(-lr_power_scalar) -
                       accum.pow(-lr_power_scalar)) /
                          lr_scalar * var;
          }
          auto x = (linear.constant(l1_scalar) * linear.sign() - linear);
          if (lr_power_scalar == -0.5) {
            auto y = new_accum.sqrt() / new_accum.constant(lr_scalar) +
                     linear.constant(2 * l2_scalar);
            var = x / y;
          } else {
            auto y = new_accum.pow(-lr_power_scalar) /
                         new_accum.constant(lr_scalar) +
                     linear.constant(2 * l2_scalar);
            var = x / y;
          }
          var = (linear.abs() > linear.constant(l1_scalar))
                    .select(var, var.constant(0));
          accum += grad.square();
        }
      } else {
        CHECK_EQ(1, inner_dim);
        auto indices_vec = indices.vec<Tindex>();
        auto var_flat = var.flat<T>();
        auto accum_flat = accum.flat<T>();
        auto linear_flat = linear.flat<T>();
        auto grad_flat = grad.flat<T>();
        T lr_scalar = lr.scalar<T>()();
        T l1_scalar = l1.scalar<T>()();
        T l2_scalar = l2.scalar<T>()();
        T lr_power_scalar = lr_power.scalar<T>()();

        for (Tindex i = 0; i < N; i++) {
          const Tindex index = indices_vec(i);
          T& a = accum_flat(index);
          T& l = linear_flat(index);
          T& v = var_flat(index);
          const T& g = grad_flat(i);

          T updated_a = a + g * g;
          T sigma = pow(updated_a, -lr_power_scalar) - pow(a, -lr_power_scalar);
          sigma /= lr_scalar;
          T updated_l = l + g - sigma * v;
          v = FtrlCompute(updated_a, updated_l, lr_scalar, l1_scalar, l2_scalar,
                          lr_power_scalar);
          a = updated_a;
          l = updated_l;
        }
      }
    }
    if (use_exclusive_lock_) {
      mu_var->unlock();
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyFtrl")                    \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyFtrlOp<CPUDevice, T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyMomentumOp : public OpKernel {
 public:
  explicit ApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (use_exclusive_lock_) {
      mutex_lock l1(*ctx->input_ref_mutex(0));
      // Don't try to acquire a lock on the second ref as they share the same
      // mutex.
      //
      // mutex_lock l2(*ctx->input_ref_mutex(1));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
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
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    const Tensor& lr = ctx->input(2);
    const Tensor& grad = ctx->input(3);
    const Tensor& momentum = ctx->input(4);
    functor::ApplyMomentum<Device, T>()(device, var.flat<T>(), accum.flat<T>(),
                                        lr.scalar<T>(), grad.flat<T>(),
                                        momentum.scalar<T>());
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(D, T)                                         \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ApplyMomentum").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyMomentumOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                               \
  template <>                                                             \
  void ApplyMomentum<GPUDevice, T>::operator()(                           \
      const GPUDevice& d, typename TTypes<T>::Flat var,                   \
      typename TTypes<T>::Flat accum, typename TTypes<T>::ConstScalar lr, \
      typename TTypes<T>::ConstFlat grad,                                 \
      typename TTypes<T>::ConstScalar momentum);                          \
  extern template struct ApplyMomentum<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_KERNELS

// Note, this op works on cpu only.
template <typename T, typename Tindex>
class SparseApplyMomentumOp : public OpKernel {
 public:
  explicit SparseApplyMomentumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override NO_THREAD_SAFETY_ANALYSIS {
    mutex* mu_var = ctx->input_ref_mutex(0);
    // mu_accum is actually the same mutex as mu_var since currently we use a
    // global mutex.
    //
    // mutex* mu_accum = ctx->input_ref_mutex(1);
    if (use_exclusive_lock_) {
      mu_var->lock();
    }
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor accum = ctx->mutable_input(1, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, accum.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
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
      auto accum_flat = accum.flat_outer_dims<T>();
      auto grad_flat = grad.flat_outer_dims<T>();
      T lr_scalar = lr.scalar<T>()();
      T momentum_scalar = momentum.scalar<T>()();

      for (Tindex i = 0; i < N; i++) {
        const Tindex index = indices_vec(i);
        auto a = accum_flat.template chip<0>(index);
        auto g = grad_flat.template chip<0>(i);
        auto v = var_flat.template chip<0>(index);
        a = a * a.constant(momentum_scalar) + g;
        v -= a.constant(lr_scalar) * a;
      }
    }
    if (use_exclusive_lock_) {
      mu_var->unlock();
    }

    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;
};

#define REGISTER_KERNELS(T, Tindices)                                \
  REGISTER_KERNEL_BUILDER(Name("SparseApplyMomentum")                \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          SparseApplyMomentumOp<T, Tindices>);

REGISTER_KERNELS(float, int32);
REGISTER_KERNELS(float, int64);
REGISTER_KERNELS(double, int32);
REGISTER_KERNELS(double, int64);
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyAdamOp : public OpKernel {
 public:
  explicit ApplyAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (use_exclusive_lock_) {
      // all input refs share the same mutex
      mutex_lock l1(*ctx->input_ref_mutex(0));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor m = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor v = ctx->mutable_input(2, use_exclusive_lock_);
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

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
                errors::InvalidArgument("lr is not a scalar: ",
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
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor m = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor v = ctx->mutable_input(2, use_exclusive_lock_);
    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& lr = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);
    const Tensor& grad = ctx->input(9);

    functor::ApplyAdam<Device, T>()(device, var.flat<T>(), m.flat<T>(),
                                    v.flat<T>(), beta1_power.scalar<T>(),
                                    beta2_power.scalar<T>(), lr.scalar<T>(),
                                    beta1.scalar<T>(), beta2.scalar<T>(),
                                    epsilon.scalar<T>(), grad.flat<T>());
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(D, T)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("ApplyAdam").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyAdamOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA
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
      typename TTypes<T>::ConstFlat grad);                    \
  extern template struct ApplyAdam<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_KERNELS

template <typename Device, typename T>
class ApplyRMSPropOp : public OpKernel {
 public:
  explicit ApplyRMSPropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
  }

  void Compute(OpKernelContext* ctx) override {
    if (use_exclusive_lock_) {
      // all input refs share the same mutex
      mutex_lock l1(*ctx->input_ref_mutex(0));
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    } else {
      DoValidate(ctx);
      if (!ctx->status().ok()) return;
      DoCompute(ctx);
    }
    ctx->forward_ref_input_to_ref_output(0, 0);
  }

 private:
  bool use_exclusive_lock_;

  void DoValidate(OpKernelContext* ctx) {
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(2, use_exclusive_lock_);

    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(0)));
    OP_REQUIRES(
        ctx, ms.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(1)));
    OP_REQUIRES(
        ctx, mom.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", def().input(2)));

    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);

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

    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));
  }

  void DoCompute(OpKernelContext* ctx) {
    const Device& device = ctx->template eigen_device<Device>();
    Tensor var = ctx->mutable_input(0, use_exclusive_lock_);
    Tensor ms = ctx->mutable_input(1, use_exclusive_lock_);
    Tensor mom = ctx->mutable_input(2, use_exclusive_lock_);
    const Tensor& lr = ctx->input(3);
    const Tensor& rho = ctx->input(4);
    const Tensor& momentum = ctx->input(5);
    const Tensor& epsilon = ctx->input(6);
    const Tensor& grad = ctx->input(7);

    functor::ApplyRMSProp<Device, T>()(device, var.flat<T>(), ms.flat<T>(),
                                       mom.flat<T>(), lr.scalar<T>(),
                                       rho.scalar<T>(), momentum.scalar<T>(),
                                       epsilon.scalar<T>(), grad.flat<T>());
  }
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(D, T)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("ApplyRMSProp").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ApplyRMSPropOp<D##Device, T>);

REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA
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
  extern template struct ApplyRMSProp<GPUDevice, T>;
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#endif
#undef REGISTER_KERNELS

}  // namespace tensorflow
