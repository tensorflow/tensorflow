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

// See docs in ../ops/array_ops.cc

#define EIGEN_USE_THREADS

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/one_hot_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename TI>
class OneHotOp : public OpKernel {
 public:
  explicit OneHotOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& indices = ctx->input(0);
    const Tensor& depth = ctx->input(1);
    const Tensor& on_value = ctx->input(2);
    const Tensor& off_value = ctx->input(3);
    const TensorShape& indices_shape = indices.shape();

    const int indices_dims = indices_shape.dims();
    const int output_dims = indices_dims + 1;

    // Preliminary validation of sizes.
    OP_REQUIRES(
        ctx, axis_ == -1 || (axis_ >= 0 && axis_ < output_dims),
        errors::InvalidArgument("Expected axis to be -1 or between [0, ",
                                output_dims, ").  But received: ", axis_));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(depth.shape()),
                errors::InvalidArgument("depth must be a scalar, but got: ",
                                        depth.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(on_value.shape()),
                errors::InvalidArgument("on_value must be a scalar, but got: ",
                                        on_value.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(off_value.shape()),
                errors::InvalidArgument("off_value must be a scalar, but got: ",
                                        off_value.shape().DebugString()));

    const int axis = (axis_ == -1) ? indices_dims : axis_;

    // The one-hot dimension.
    const int32_t depth_v = depth.scalar<int32>()();
    OP_REQUIRES(
        ctx, depth_v >= 0,
        errors::InvalidArgument("depth must be non-negative, got: ", depth_v));
    OP_REQUIRES(
        ctx,
        MultiplyWithoutOverflow(indices_shape.num_elements(), depth_v) >= 0,
        errors::InvalidArgument("OneHot result would have shape ",
                                indices_shape.DebugString(), " + [", depth_v,
                                "], which exceeds 2**63 - 1 elements"));

    TensorShape output_shape = indices_shape;
    output_shape.InsertDim(axis, depth_v);

    auto on_value_t = on_value.scalar<T>();
    auto off_value_t = off_value.scalar<T>();

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output_shape.num_elements() > 0) {
      // prefix_dim_size == # of elements before the axis
      // depth_v == # of elements per axis
      // suffix_dim_size == # of elements after the axis
      int64_t prefix_dim_size = 1;
      for (int i = 0; i < axis; ++i) {
        prefix_dim_size *= indices_shape.dim_size(i);
      }
      int64_t suffix_dim_size = indices_shape.num_elements() / prefix_dim_size;

      // Split indices into matrix of size prefix_dim_size x suffix_dim_size
      auto indices_t =
          indices.shaped<TI, 2>({prefix_dim_size, suffix_dim_size});
      // Split output into 3-Tensor of size:
      //   prefix_dim_size x depth x suffix_dim_size.
      auto output_t =
          output->shaped<T, 3>({prefix_dim_size, depth_v, suffix_dim_size});

      functor::OneHot<Device, T, TI>::Compute(ctx->eigen_device<Device>(),
                                              indices_t, on_value_t,
                                              off_value_t, &output_t);
    }
  }

 private:
  int32 axis_;

  TF_DISALLOW_COPY_AND_ASSIGN(OneHotOp);
};

#define REGISTER_ONE_HOT_INDEX(type, index_type)                \
  REGISTER_KERNEL_BUILDER(Name("OneHot")                        \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotOp<CPUDevice, type, index_type>);

#define REGISTER_ONE_HOT(type)         \
  REGISTER_ONE_HOT_INDEX(type, uint8); \
  REGISTER_ONE_HOT_INDEX(type, int8);  \
  REGISTER_ONE_HOT_INDEX(type, int32); \
  REGISTER_ONE_HOT_INDEX(type, int64_t)

TF_CALL_ALL_TYPES(REGISTER_ONE_HOT);

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC_INDEX(T, TI)                                      \
  template <>                                                              \
  void OneHot<GPUDevice, T, TI>::Compute(                                  \
      const GPUDevice& d, const typename TTypes<TI>::ConstMatrix& indices, \
      const typename TTypes<T>::ConstScalar& on_value,                     \
      const typename TTypes<T>::ConstScalar& off_value,                    \
      typename TTypes<T, 3>::Tensor* output);                              \
  extern template struct OneHot<GPUDevice, T, TI>;

#define DECLARE_GPU_SPEC(T)         \
  DECLARE_GPU_SPEC_INDEX(T, uint8); \
  DECLARE_GPU_SPEC_INDEX(T, int8);  \
  DECLARE_GPU_SPEC_INDEX(T, int32); \
  DECLARE_GPU_SPEC_INDEX(T, int64_t);

TF_CALL_int8(DECLARE_GPU_SPEC);
TF_CALL_int32(DECLARE_GPU_SPEC);
TF_CALL_int64(DECLARE_GPU_SPEC);
TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC_INDEX
#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_ONE_HOT_GPU_INDEX(type, index_type)            \
  REGISTER_KERNEL_BUILDER(Name("OneHot")                        \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<index_type>("TI") \
                              .TypeConstraint<type>("T")        \
                              .HostMemory("depth"),             \
                          OneHotOp<GPUDevice, type, index_type>);

#define REGISTER_ONE_HOT_GPU(type)         \
  REGISTER_ONE_HOT_GPU_INDEX(type, uint8); \
  REGISTER_ONE_HOT_GPU_INDEX(type, int8);  \
  REGISTER_ONE_HOT_GPU_INDEX(type, int32); \
  REGISTER_ONE_HOT_GPU_INDEX(type, int64_t);

TF_CALL_int8(REGISTER_ONE_HOT_GPU);
TF_CALL_int32(REGISTER_ONE_HOT_GPU);
TF_CALL_int64(REGISTER_ONE_HOT_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_ONE_HOT_GPU);

#undef REGISTER_ONE_HOT_GPU_INDEX
#undef REGISTER_ONE_HOT_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
