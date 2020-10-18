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

#ifndef TENSORFLOW_CORE_KERNELS_CWISE_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_CWISE_OPS_COMMON_H_

// See docs in ../ops/math_ops.cc.
#define _USE_MATH_DEFINES
#include <cmath>

#define EIGEN_USE_THREADS

#include "tensorflow/core/platform/bfloat16.h"


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_gradients.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class BinaryOpShared : public OpKernel {
 public:
  explicit BinaryOpShared(OpKernelConstruction* ctx, DataType out, DataType in);

 protected:
  struct BinaryOpState {
    // Sets up bcast with the shape of in0 and in1, ensures that the bcast
    // is valid, and if so, set out, either by allocating a new buffer using
    // ctx->output(...) or by creating an alias for an owned input buffer for
    // in-place computation.
    // Caller must check ctx->status() upon return for non-ok status.
    // If ctx->status().ok() is true, then out is guaranteed to be allocated.
    explicit BinaryOpState(OpKernelContext* ctx);

    const Tensor& in0;
    const Tensor& in1;

    BCast bcast;
    Tensor* out = nullptr;
    int64 out_num_elements;

    int64 in0_num_elements;
    int64 in1_num_elements;

    int ndims;
    bool result;
  };

  void SetUnimplementedError(OpKernelContext* ctx);
  void SetComputeError(OpKernelContext* ctx);
};

// Coefficient-wise binary operations:
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined in cwise_ops.h. E.g., functor::add.
template <typename Device, typename Functor>
class BinaryOp : public BinaryOpShared {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.

  explicit BinaryOp(OpKernelConstruction* ctx)
      : BinaryOpShared(ctx, DataTypeToEnum<Tout>::v(),
                       DataTypeToEnum<Tin>::v()) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_0 = ctx->input(0);
    const Tensor& input_1 = ctx->input(1);
    const Device& eigen_device = ctx->eigen_device<Device>();
    bool error = false;
    bool* const error_ptr = Functor::has_errors ? &error : nullptr;

    // NOTE: Handle three simple cases before building the BinaryOpState, which
    // is relatively expensive for small operations.
    if (input_0.shape() == input_1.shape()) {
      // tensor op tensor with no broadcasting.
      Tensor* out;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0, 1}, 0, input_0.shape(), &out));
      functor::BinaryFunctor<Device, Functor, 1>()(
          eigen_device, out->template flat<Tout>(),
          input_0.template flat<Tin>(), input_1.template flat<Tin>(),
          error_ptr);
      if (Functor::has_errors && error) {
        SetComputeError(ctx);
      }
      return;
    } else if (input_0.shape().dims() == 0) {
      // scalar op tensor.
      Tensor* out;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {1}, 0, input_1.shape(), &out));

      functor::BinaryFunctor<Device, Functor, 1>().Left(
          eigen_device, out->template flat<Tout>(),
          input_0.template scalar<Tin>(), input_1.template flat<Tin>(),
          error_ptr);
      if (Functor::has_errors && error) {
        SetComputeError(ctx);
      }
      return;
    } else if (input_1.shape().dims() == 0) {
      // tensor op scalar.
      Tensor* out;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, input_0.shape(), &out));
      functor::BinaryFunctor<Device, Functor, 1>().Right(
          eigen_device, out->template flat<Tout>(),
          input_0.template flat<Tin>(), input_1.template scalar<Tin>(),
          error_ptr);
      if (Functor::has_errors && error) {
        SetComputeError(ctx);
      }
      return;
    }

    // 'state': Shared helper not dependent on T to reduce code size
    BinaryOpState state(ctx);
    if (ctx->status().code() == error::RESOURCE_EXHAUSTED) {
      // Stop when BinaryOpState's constructor failed due to OOM.
      return;
    }
    auto& bcast = state.bcast;
    Tensor* out = state.out;
    if (!bcast.IsValid()) {
      if (ctx->status().ok()) {
        if (state.result) {
          functor::SetOneFunctor<Device, bool>()(eigen_device,
                                                 out->flat<bool>());
        } else {
          functor::SetZeroFunctor<Device, bool>()(eigen_device,
                                                  out->flat<bool>());
        }
      }
      return;
    }

    auto& in0 = state.in0;
    auto& in1 = state.in1;
    if (state.out_num_elements == 0) {
      return;
    }

    const int ndims = state.ndims;
    if (ndims <= 1) {
      auto out_flat = out->flat<Tout>();
      if (state.in1_num_elements == 1) {
        // tensor op scalar
        functor::BinaryFunctor<Device, Functor, 1>().Right(
            eigen_device, out_flat, in0.template flat<Tin>(),
            in1.template scalar<Tin>(), error_ptr);
      } else if (state.in0_num_elements == 1) {
        // scalar op tensor
        functor::BinaryFunctor<Device, Functor, 1>().Left(
            eigen_device, out_flat, in0.template scalar<Tin>(),
            in1.template flat<Tin>(), error_ptr);
      } else {
        functor::BinaryFunctor<Device, Functor, 1>()(
            eigen_device, out_flat, in0.template flat<Tin>(),
            in1.template flat<Tin>(), error_ptr);
      }
    } else if (ndims == 2) {
      functor::BinaryFunctor<Device, Functor, 2>().BCast(
          eigen_device, out->shaped<Tout, 2>(bcast.result_shape()),
          in0.template shaped<Tin, 2>(bcast.x_reshape()),
          BCast::ToIndexArray<2>(bcast.x_bcast()),
          in1.template shaped<Tin, 2>(bcast.y_reshape()),
          BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);
    } else if (ndims == 3) {
      functor::BinaryFunctor<Device, Functor, 3>().BCast(
          eigen_device, out->shaped<Tout, 3>(bcast.result_shape()),
          in0.template shaped<Tin, 3>(bcast.x_reshape()),
          BCast::ToIndexArray<3>(bcast.x_bcast()),
          in1.template shaped<Tin, 3>(bcast.y_reshape()),
          BCast::ToIndexArray<3>(bcast.y_bcast()), error_ptr);
    } else if (ndims == 4) {
      functor::BinaryFunctor<Device, Functor, 4>().BCast(
          eigen_device, out->shaped<Tout, 4>(bcast.result_shape()),
          in0.template shaped<Tin, 4>(bcast.x_reshape()),
          BCast::ToIndexArray<4>(bcast.x_bcast()),
          in1.template shaped<Tin, 4>(bcast.y_reshape()),
          BCast::ToIndexArray<4>(bcast.y_bcast()), error_ptr);
    } else if (ndims == 5) {
      functor::BinaryFunctor<Device, Functor, 5>().BCast(
          eigen_device, out->shaped<Tout, 5>(bcast.result_shape()),
          in0.template shaped<Tin, 5>(bcast.x_reshape()),
          BCast::ToIndexArray<5>(bcast.x_bcast()),
          in1.template shaped<Tin, 5>(bcast.y_reshape()),
          BCast::ToIndexArray<5>(bcast.y_bcast()), error_ptr);
    } else {
      SetUnimplementedError(ctx);
    }
    if (Functor::has_errors && error) {
      SetComputeError(ctx);
    }
  }
};

template <typename Device, typename T>
class ApproximateEqualOp : public OpKernel {
 public:
  explicit ApproximateEqualOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float tolerance;
    OP_REQUIRES_OK(context, context->GetAttr("tolerance", &tolerance));
    tolerance_ = T(tolerance);
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& x_input = context->input(0);
    const Tensor& y_input = context->input(1);
    OP_REQUIRES(
        context, x_input.shape() == y_input.shape(),
        errors::InvalidArgument("x and y must be of the same shape. ",
                                "x shape: ", x_input.shape().DebugString(),
                                ". y shape: ", y_input.shape().DebugString()));
    Tensor* z_output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, x_input.shape(), &z_output));
    const Device& d = context->eigen_device<Device>();
    typename TTypes<T>::ConstFlat x(x_input.flat<T>());
    typename TTypes<T>::ConstFlat y(y_input.flat<T>());
    typename TTypes<bool>::Flat z(z_output->flat<bool>());
    functor::ApproximateEqual<Device, T>()(d, x, y, tolerance_, z);
  }

 private:
  T tolerance_;
};

// Basic coefficient-wise binary operations that are known to not require
// any broadcasting. This is the case for example of the gradients of
// unary operations.
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined above. E.g., functor::tanh_grad.
template <typename Device, typename Functor>
class SimpleBinaryOp : public OpKernel {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.

  explicit SimpleBinaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    auto in0_flat = in0.flat<Tin>();
    auto in1_flat = in1.flat<Tin>();
    const Device& eigen_device = ctx->eigen_device<Device>();

    Tensor* out = nullptr;
    if (std::is_same<Tin, Tout>::value) {
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0, 1}, 0, in0.shape(), &out));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &out));
    }
    auto out_flat = out->flat<Tout>();
    functor::SimpleBinaryFunctor<Device, Functor>()(eigen_device, out_flat,
                                                    in0_flat, in1_flat);
  }
};

// Coefficient-wise unary operations:
//   Device: E.g., CPUDevice, GPUDevice.
//   Functor: defined in cwise_ops.h. E.g., functor::sqrt.
template <typename Device, typename Functor>
class UnaryOp : public OpKernel {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.
  // Tin may be different from Tout. E.g., abs: complex64 -> float

  explicit UnaryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto in = DataTypeToEnum<Tin>::v();
    auto out = DataTypeToEnum<Tout>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({in}, {out}));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    Tensor* out = nullptr;
    if (std::is_same<Tin, Tout>::value) {
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, inp.shape(), &out));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &out));
    }
    functor::UnaryFunctor<Device, Functor>()(
        ctx->eigen_device<Device>(), out->flat<Tout>(), inp.flat<Tin>());
  }
};

template <typename Device, VariantUnaryOp OpEnum>
class UnaryVariantOp : public OpKernel {
 public:
  explicit UnaryVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(inp.shape()),
        errors::InvalidArgument("Non-scalar variants are not supported."));
    const Variant& v = inp.scalar<Variant>()();
    Variant v_out;
    OP_REQUIRES_OK(ctx, UnaryOpVariant<Device>(ctx, OpEnum, v, &v_out));
    int numa_node = ctx->device()->NumaNode();
    Tensor out(cpu_allocator(numa_node), DT_VARIANT, TensorShape());
    out.scalar<Variant>()() = std::move(v_out);
    ctx->set_output(0, std::move(out));
  }
};

namespace functor {

template <typename D, typename Out, typename Rhs>
void Assign(const D& d, Out out, Rhs rhs) {
  out.device(d) = rhs;
}

// Partial specialization of BinaryFunctor<Device=CPUDevice, Functor, NDIMS>
// for functors with no error checking.
template <typename Functor, int NDIMS>
struct BinaryFunctor<CPUDevice, Functor, NDIMS, false> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    Assign(d, out, in0.binaryExpr(in1, typename Functor::func()));
  }

  void Left(const CPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef
        typename Eigen::internal::scalar_left<Tout, Tin, Binary,
                                              /*is_scalar_in_host_memory=*/true>
            Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

  void Right(const CPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<
        Tout, Tin, Binary, /*is_scalar_in_host_memory=*/true>
        Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

  void BCast(const CPUDevice& dev,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typename Functor::func func;
    if (AllOne<NDIMS>(bcast0) && AllOne<NDIMS>(bcast1)) {
      Assign(dev, out, in0.binaryExpr(in1, func));
    } else if (AllOne<NDIMS>(bcast0)) {
      auto rhs = in1.broadcast(bcast1);
      Assign(dev, out, in0.binaryExpr(rhs, func));
    } else if (AllOne<NDIMS>(bcast1)) {
      auto lhs = in0.broadcast(bcast0);
      Assign(dev, out, lhs.binaryExpr(in1, func));
    } else {
      auto lhs = in0.broadcast(bcast0);
      auto rhs = in1.broadcast(bcast1);
      Assign(dev, out, lhs.binaryExpr(rhs, func));
    }
  }
};

// Partial specialization of BinaryFunctor<Device=CPUDevice, Functor, 2>
// for functors with no error checking.
template <typename Functor>
struct BinaryFunctor<CPUDevice, Functor, 2, false> {
  enum { NDIMS = 2 };

  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    Assign(d, out, in0.binaryExpr(in1, typename Functor::func()));
  }

  void Left(const CPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef
        typename Eigen::internal::scalar_left<Tout, Tin, Binary,
                                              /*is_scalar_in_host_memory=*/true>
            Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

  void Right(const CPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<
        Tout, Tin, Binary, /*is_scalar_in_host_memory=*/true>
        Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data())));
  }

#if !defined(EIGEN_HAS_INDEX_LIST)
  inline Eigen::DSizes<int, 2> NByOne(int n) {
    return Eigen::DSizes<int, 2>(n, 1);
  }
  inline Eigen::DSizes<int, 2> OneByM(int m) {
    return Eigen::DSizes<int, 2>(1, m);
  }
#else
  inline Eigen::IndexList<int, Eigen::type2index<1>> NByOne(int n) {
    Eigen::IndexList<int, Eigen::type2index<1>> ret;
    ret.set(0, n);
    return ret;
  }
  inline Eigen::IndexList<Eigen::type2index<1>, int> OneByM(int m) {
    Eigen::IndexList<Eigen::type2index<1>, int> ret;
    ret.set(1, m);
    return ret;
  }
#endif

  void BCast(const CPUDevice& dev,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typedef typename Functor::in_type T;
    typename Functor::func func;
    if (Functor::use_bcast_optimization && use_bcast_optimization<T>::value) {
      // Optimize for speed by using Eigen::type2index and avoid
      // .broadcast() when we know it's a no-op.
      //
      // Here, we need to handle 6 cases depending on how many "1"
      // exist in in0 and in1's shapes (4 numbers in total). It's not
      // possible that two shapes have more than 2 1s because those
      // are simplified to NDIMS==1 case.
      //
      // Because this optimization increases the binary size for each
      // Functor (+, -, *, /, <, <=, etc.), type and ndim combination.
      // we only apply such optimization for selected ops/types/ndims.
      //
      // Because NDIMS, Functor::use_broadcast_optimization and
      // use_broadcast_optimization<T> are compile-time constant, gcc
      // does a decent job avoiding generating code when conditions
      // are not met.
      const int a = in0.dimension(0);  // in0 is shape [a, b]
      const int b = in0.dimension(1);
      const int c = in1.dimension(0);  // in1 is shape [c, d]
      const int d = in1.dimension(1);
      if ((a == 1) && (d == 1)) {
        auto lhs = in0.reshape(OneByM(b)).broadcast(NByOne(c));
        auto rhs = in1.reshape(NByOne(c)).broadcast(OneByM(b));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if ((b == 1) && (c == 1)) {
        auto lhs = in0.reshape(NByOne(a)).broadcast(OneByM(d));
        auto rhs = in1.reshape(OneByM(d)).broadcast(NByOne(a));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (a == 1) {
        auto lhs = in0.reshape(OneByM(b)).broadcast(NByOne(c));
        auto rhs = in1;
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (b == 1) {
        auto lhs = in0.reshape(NByOne(a)).broadcast(OneByM(d));
        auto rhs = in1;
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (c == 1) {
        auto lhs = in0;
        auto rhs = in1.reshape(OneByM(d)).broadcast(NByOne(a));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
      if (d == 1) {
        auto lhs = in0;
        auto rhs = in1.reshape(NByOne(c)).broadcast(OneByM(b));
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }

      const bool bcast0_all_one = AllOne<NDIMS>(bcast0);
      const bool bcast1_all_one = AllOne<NDIMS>(bcast1);
      if (bcast0_all_one && !bcast1_all_one) {
        auto lhs = in0;  // No need to do broadcast for in0
        auto rhs = in1.broadcast(bcast1);
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }

      if (!bcast0_all_one && bcast1_all_one) {
        auto lhs = in0.broadcast(bcast0);
        auto rhs = in1;  // No need to do broadcast for in1
        Assign(dev, out, lhs.binaryExpr(rhs, func));
        return;
      }
    }

    // Fallback path. Always works and probably slower.
    auto lhs = in0.broadcast(bcast0);
    auto rhs = in1.broadcast(bcast1);
    Assign(dev, out, lhs.binaryExpr(rhs, func));
  }
};

// Version of BinaryFunctor with error handling.
template <typename Functor, int NDIMS>
struct BinaryFunctor<CPUDevice, Functor, NDIMS, true> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in0,
                  typename Functor::tin_type in1, bool* error) {
    Assign(d, out, in0.binaryExpr(in1, typename Functor::func(error)));
  }

  void Left(const CPUDevice& d, typename Functor::tout_type out,
            typename Functor::tscalar_type scalar,
            typename Functor::tin_type in, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef
        typename Eigen::internal::scalar_left<Tout, Tin, Binary,
                                              /*is_scalar_in_host_memory=*/true>
            Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data(), error)));
  }

  void Right(const CPUDevice& d, typename Functor::tout_type out,
             typename Functor::tin_type in,
             typename Functor::tscalar_type scalar, bool* error) {
    typedef typename Functor::out_type Tout;
    typedef typename Functor::in_type Tin;
    typedef typename Functor::func Binary;
    typedef typename Eigen::internal::scalar_right<
        Tout, Tin, Binary, /*is_scalar_in_host_memory=*/true>
        Unary;
    Assign(d, out, in.unaryExpr(Unary(scalar.data(), error)));
  }

  void BCast(const CPUDevice& dev,
             typename TTypes<typename Functor::out_type, NDIMS>::Tensor out,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in0,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast0,
             typename TTypes<typename Functor::in_type, NDIMS>::ConstTensor in1,
             typename Eigen::array<Eigen::DenseIndex, NDIMS> bcast1,
             bool* error) {
    typename Functor::func func(error);
    auto lhs = in0.broadcast(bcast0);
    auto rhs = in1.broadcast(bcast1);
    Assign(dev, out, lhs.binaryExpr(rhs, func));
  }
};

// Partial specialization of UnaryFunctor<Device=CPUDevice, Functor>.
template <typename Functor>
struct UnaryFunctor<CPUDevice, Functor> {
  void operator()(const CPUDevice& d, typename Functor::tout_type out,
                  typename Functor::tin_type in) {
    Assign(d, out, in.unaryExpr(typename Functor::func()));
  }
};

// Partial specialization of ApproximateEqual<Device=CPUDevice, T>.
template <typename T>
struct ApproximateEqual<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat x,
                  typename TTypes<T>::ConstFlat y, T tolerance,
                  typename TTypes<bool>::Flat z) {
    auto diff = x - y;
    z.device(d) = diff.abs() <= tolerance;
  }
};

}  // end namespace functor

#define REGISTER(OP, D, N, F, T)                                             \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_##D).TypeConstraint<T>("T"), \
                          OP<D##Device, F<T>>);

#define REGISTER_VARIANT(OP, D, N, ENUM)                       \
  REGISTER_KERNEL_BUILDER(                                     \
      Name(N).Device(DEVICE_##D).TypeConstraint<Variant>("T"), \
      OP<D##Device, ENUM>);

// Macros to register kernels for multiple types (T0, T1, etc.)  on
// device type "D" (CPU or GPU) for operation "N" (e.g., sqrt) using
// the functor "F" (e.g., functor::sqrt).

#if defined(__ANDROID_TYPES_SLIM__)
// Note that __ANDROID_TYPES_SLIM__ is also checked in the cwise_ops*.cc files.
// Normally Android TensorFlow is built with a reduced number of types (float).
// Override on the command-line using "--copt=-D__ANDROID_TYPES_FULL__"
// to generate a library with full type support with a consequent increase in
// code size.
#define REGISTER2(OP, D, N, F, T0, T1) REGISTER(OP, D, N, F, T0)
#define REGISTER3(OP, D, N, F, T0, T1, T2) REGISTER(OP, D, N, F, T0)
#define REGISTER4(OP, D, N, F, T0, T1, T2, T3) REGISTER(OP, D, N, F, T0)
#define REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4) REGISTER(OP, D, N, F, T0)
#define REGISTER6(OP, D, N, F, T0, T1, T2, T3, T4, T5) REGISTER(OP, D, N, F, T0)
#define REGISTER7(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6) \
  REGISTER(OP, D, N, F, T0)
#define REGISTER8(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7) \
  REGISTER(OP, D, N, F, T0)
#define REGISTER9(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  REGISTER(OP, D, N, F, T0)
#else  // !defined(__ANDROID_TYPES_SLIM__)
#define REGISTER2(OP, D, N, F, T0, T1) \
  REGISTER(OP, D, N, F, T0)            \
  REGISTER(OP, D, N, F, T1)
#define REGISTER3(OP, D, N, F, T0, T1, T2) \
  REGISTER2(OP, D, N, F, T0, T1)           \
  REGISTER(OP, D, N, F, T2)
#define REGISTER4(OP, D, N, F, T0, T1, T2, T3) \
  REGISTER2(OP, D, N, F, T0, T1)               \
  REGISTER2(OP, D, N, F, T2, T3)
#define REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4) \
  REGISTER3(OP, D, N, F, T0, T1, T2)               \
  REGISTER2(OP, D, N, F, T3, T4)
#define REGISTER6(OP, D, N, F, T0, T1, T2, T3, T4, T5) \
  REGISTER3(OP, D, N, F, T0, T1, T2)                   \
  REGISTER3(OP, D, N, F, T3, T4, T5)
#define REGISTER7(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6) \
  REGISTER4(OP, D, N, F, T0, T1, T2, T3)                   \
  REGISTER3(OP, D, N, F, T4, T5, T6)
#define REGISTER8(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7) \
  REGISTER4(OP, D, N, F, T0, T1, T2, T3)                       \
  REGISTER4(OP, D, N, F, T4, T5, T6, T7)
#define REGISTER9(OP, D, N, F, T0, T1, T2, T3, T4, T5, T6, T7, T8) \
  REGISTER5(OP, D, N, F, T0, T1, T2, T3, T4)                       \
  REGISTER4(OP, D, N, F, T5, T6, T7, T8)

// Instead of adding REGISTER10, etc., shard the .cc files - see
// cwise_op_equal_to_*.cc for an example.

#endif  // defined(__ANDROID_TYPES_SLIM__)

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CWISE_OPS_COMMON_H_
