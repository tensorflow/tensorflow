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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS
// TODO(b/31098934): Figure out why this is necessary here but not in
// any other place, e.g., the cwise lgamma ops.
#define EIGEN_HAS_C99_MATH 1

#include "tensorflow/core/kernels/betainc_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BetaincOp : public OpKernel {
 public:
  explicit BetaincOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& x = ctx->input(2);

    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& x_shape = x.shape();
    if (a_shape.dims() > 0 && b_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == b_shape,
                  errors::InvalidArgument(
                      "Shapes of a and b are inconsistent: ",
                      a_shape.DebugString(), " vs. ", b_shape.DebugString()));
    }
    if (a_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, a_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of a and x are inconsistent: ",
                      a_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }
    if (b_shape.dims() > 0 && x_shape.dims() > 0) {
      OP_REQUIRES(ctx, b_shape == x_shape,
                  errors::InvalidArgument(
                      "Shapes of b and x are inconsistent: ",
                      b_shape.DebugString(), " vs. ", x_shape.DebugString()));
    }

    TensorShape merged_shape(a_shape);
    if (b_shape.dims() > 0) merged_shape = b_shape;
    if (x_shape.dims() > 0) merged_shape = x_shape;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, merged_shape, &output));

    if (a_shape == b_shape && a_shape == x_shape) {
      functor::Betainc<Device, T, 1> functor;
      functor(ctx->eigen_device<Device>(), a.flat<T>(), b.flat<T>(),
              x.flat<T>(), output->flat<T>());
      return;
    }

    auto merged_shape_vec = BCast::FromShape(merged_shape);
    BCast a_shaper(BCast::FromShape(a_shape), merged_shape_vec);
    BCast b_shaper(BCast::FromShape(b_shape), merged_shape_vec);
    BCast x_shaper(BCast::FromShape(x_shape), merged_shape_vec);

    int ndims = static_cast<int>(a_shaper.x_reshape().size());

    switch (ndims) {
#define CASE(NDIM)                                                        \
  case NDIM: {                                                            \
    functor::Betainc<Device, T, NDIM> functor;                            \
    auto a_value = a.shaped<T, NDIM>(a_shaper.x_reshape());               \
    auto b_value = b.shaped<T, NDIM>(b_shaper.x_reshape());               \
    auto x_value = x.shaped<T, NDIM>(x_shaper.x_reshape());               \
    functor.BCast(ctx->eigen_device<Device>(), a_value,                   \
                  BCast::ToIndexArray<NDIM>(a_shaper.x_bcast()), b_value, \
                  BCast::ToIndexArray<NDIM>(b_shaper.x_bcast()), x_value, \
                  BCast::ToIndexArray<NDIM>(x_shaper.x_bcast()),          \
                  output->shaped<T, NDIM>(a_shaper.y_reshape()));         \
    return;                                                               \
  }

      CASE(1);
      CASE(2);
      default: {
        ctx->SetStatus(errors::InvalidArgument(
            "Broadcasting rank not supported: ", ndims));
        return;
      }
    }
  }
};

#define REGISTER_KERNELS(type)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Betainc").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BetaincOp<CPUDevice, type>);

REGISTER_KERNELS(float);
REGISTER_KERNELS(double);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC_NDIM(T, NDIM)                               \
  template <>                                                        \
  void Betainc<GPUDevice, T, NDIM>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, NDIM>::ConstTensor a,   \
      typename TTypes<T, NDIM>::ConstTensor b,                       \
      typename TTypes<T, NDIM>::ConstTensor x,                       \
      typename TTypes<T, NDIM>::Tensor output);                      \
  template <>                                                        \
  void Betainc<GPUDevice, T, NDIM>::BCast(                           \
      const GPUDevice& d, typename TTypes<T, NDIM>::ConstTensor a,   \
      const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_a, \
      typename TTypes<T, NDIM>::ConstTensor b,                       \
      const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_b, \
      typename TTypes<T, NDIM>::ConstTensor x,                       \
      const typename Eigen::array<Eigen::DenseIndex, NDIM>& bcast_x, \
      typename TTypes<T, NDIM>::Tensor output);                      \
  extern template struct Betainc<GPUDevice, T, NDIM>;

#define DECLARE_GPU_SPEC(T)   \
  DECLARE_GPU_SPEC_NDIM(T, 1) \
  DECLARE_GPU_SPEC_NDIM(T, 2)

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);

#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_NDIM
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Betainc").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BetaincOp<GPUDevice, type>);

REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
