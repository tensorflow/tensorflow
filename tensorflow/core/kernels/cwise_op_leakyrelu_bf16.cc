/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <functional>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

// Keeping all new leakyrelu changes in 1 file.
// This is similar to changes in cwise_ops.h
namespace Eigen {
namespace internal {

template <typename Scalar>
struct leakyrelu_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit leakyrelu_op(float val = 0.2f)
      EIGEN_NO_THROW {
    m_alpha = Scalar(val);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    return x > Scalar(0) ? x : x * Scalar(m_alpha);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    Packet alpha = pset1<Packet>(m_alpha);
    return pselect(pcmp_le(x, pzero(x)), pmul(x, alpha), x);
  }
  Scalar m_alpha;
};

template <typename Scalar>
struct functor_traits<leakyrelu_op<Scalar>> {
  enum {
    Cost =
        Eigen::NumTraits<Scalar>::AddCost + Eigen::NumTraits<Scalar>::MulCost,
    PacketAccess =
        packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasCmp,
  };
};

}  // namespace internal
}  // namespace Eigen

namespace tensorflow {

namespace functor {
template <typename T>
struct leakyrelu : base<T, Eigen::internal::leakyrelu_op<T>> {};
}  // namespace functor

template <typename Device, typename Functor>
class LeakyReluOp : public OpKernel {
 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.
  // Tin may be different from Tout. E.g., abs: complex64 -> float

  explicit LeakyReluOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    auto in = DataTypeToEnum<Tin>::v();
    auto out = DataTypeToEnum<Tout>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({in}, {out}));

    float alpha;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha));
    alpha_ = alpha;
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
    functor::UnaryFunctorWithArg<Device, Functor, float>()(
        ctx->eigen_device<Device>(), out->flat<Tout>(), inp.flat<Tin>(),
        alpha_);
  }

 private:
  float alpha_;
};

REGISTER(LeakyReluOp, CPU, "LeakyRelu", functor::leakyrelu, bfloat16);
}  // namespace tensorflow
