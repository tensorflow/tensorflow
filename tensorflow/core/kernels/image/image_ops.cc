/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/image/image_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

// Explicit instantiation of the CPU functor.
typedef Eigen::ThreadPoolDevice CPUDevice;

template struct FillProjectiveTransform<CPUDevice, uint8>;
template struct FillProjectiveTransform<CPUDevice, int32>;
template struct FillProjectiveTransform<CPUDevice, int64_t>;
template struct FillProjectiveTransform<CPUDevice, Eigen::half>;
template struct FillProjectiveTransform<CPUDevice, float>;
template struct FillProjectiveTransform<CPUDevice, double>;

}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::FillProjectiveTransform;
using generator::Interpolation;
using generator::Mode;

template <typename Device, typename T>
void DoImageProjectiveTransformOp(OpKernelContext* ctx,
                                  const Interpolation& interpolation,
                                  const Mode& fill_mode) {
  const Tensor& images_t = ctx->input(0);
  const Tensor& transform_t = ctx->input(1);
  OP_REQUIRES(ctx, images_t.shape().dims() == 4,
              errors::InvalidArgument("Input images must have rank 4"));
  OP_REQUIRES(ctx,
              (TensorShapeUtils::IsMatrix(transform_t.shape()) &&
               (transform_t.dim_size(0) == images_t.dim_size(0) ||
                transform_t.dim_size(0) == 1) &&
               transform_t.dim_size(1) == 8),
              errors::InvalidArgument(
                  "Input transform should be num_images x 8 or 1 x 8"));

  int32_t out_height, out_width;
  // Kernel is shared by legacy "ImageProjectiveTransform" op with 2 args.
  if (ctx->num_inputs() >= 3) {
    const Tensor& shape_t = ctx->input(2);
    OP_REQUIRES(ctx, shape_t.dims() == 1,
                errors::InvalidArgument("output shape must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(ctx, shape_t.NumElements() == 2,
                errors::InvalidArgument("output shape must have two elements",
                                        shape_t.shape().DebugString()));
    auto shape_vec = shape_t.vec<int32>();
    out_height = shape_vec(0);
    out_width = shape_vec(1);
    OP_REQUIRES(ctx, out_height > 0 && out_width > 0,
                errors::InvalidArgument("output dimensions must be positive"));
  } else {
    // Shape is N (batch size), H (height), W (width), C (channels).
    out_height = images_t.shape().dim_size(1);
    out_width = images_t.shape().dim_size(2);
  }

  T fill_value(0);
  // Kernel is shared by "ImageProjectiveTransformV2" with 3 args.
  if (ctx->num_inputs() >= 4) {
    const Tensor& fill_value_t = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(fill_value_t.shape()),
                errors::InvalidArgument("fill_value must be a scalar",
                                        fill_value_t.shape().DebugString()));
    fill_value = static_cast<T>(*(fill_value_t.scalar<float>().data()));
  }

  Tensor* output_t;
  TensorShape output_shape;
  OP_REQUIRES_OK(
      ctx, TensorShape::BuildTensorShape({images_t.dim_size(0), out_height,
                                          out_width, images_t.dim_size(3)},
                                         &output_shape));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_t));
  auto output = output_t->tensor<T, 4>();
  auto images = images_t.tensor<T, 4>();
  auto transform = transform_t.matrix<float>();

  (FillProjectiveTransform<Device, T>(interpolation))(
      ctx->eigen_device<Device>(), &output, images, transform, fill_mode,
      fill_value);
}

template <typename Device, typename T>
class ImageProjectiveTransformV2 : public OpKernel {
 private:
  Interpolation interpolation_;
  Mode fill_mode_;

 public:
  explicit ImageProjectiveTransformV2(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string interpolation_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "NEAREST") {
      interpolation_ = Interpolation::NEAREST;
    } else if (interpolation_str == "BILINEAR") {
      interpolation_ = Interpolation::BILINEAR;
    } else {
      LOG(ERROR) << "Invalid interpolation " << interpolation_str
                 << ". Supported types: NEAREST, BILINEAR";
    }
    string mode_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_mode", &mode_str));
    if (mode_str == "REFLECT") {
      fill_mode_ = Mode::FILL_REFLECT;
    } else if (mode_str == "WRAP") {
      fill_mode_ = Mode::FILL_WRAP;
    } else if (mode_str == "CONSTANT") {
      fill_mode_ = Mode::FILL_CONSTANT;
    } else if (mode_str == "NEAREST") {
      fill_mode_ = Mode::FILL_NEAREST;
    } else {
      LOG(ERROR) << "Invalid mode " << mode_str
                 << ". Supported types: REFLECT, WRAP, CONSTANT, NEAREST";
    }
  }

  void Compute(OpKernelContext* ctx) override {
    DoImageProjectiveTransformOp<Device, T>(ctx, interpolation_, fill_mode_);
  }
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV2")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageProjectiveTransformV2<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

template <typename Device, typename T>
class ImageProjectiveTransformV3
    : public ImageProjectiveTransformV2<Device, T> {
 public:
  explicit ImageProjectiveTransformV3(OpKernelConstruction* ctx)
      : ImageProjectiveTransformV2<Device, T>(ctx) {}
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV3")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageProjectiveTransformV3<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;
typedef generator::Mode Mode;

namespace functor {

// NOTE(ringwalt): We get an undefined symbol error if we don't explicitly
// instantiate the operator() in GCC'd code.
#define DECLARE_PROJECT_FUNCTOR(TYPE)                                       \
  template <>                                                               \
  void FillProjectiveTransform<GPUDevice, TYPE>::operator()(                \
      const GPUDevice& device, OutputType* output, const InputType& images, \
      const TransformsType& transform, const Mode fill_mode,                \
      const TYPE fill_value) const;                                         \
  extern template struct FillProjectiveTransform<GPUDevice, TYPE>

TF_CALL_uint8(DECLARE_PROJECT_FUNCTOR);
TF_CALL_int32(DECLARE_PROJECT_FUNCTOR);
TF_CALL_int64(DECLARE_PROJECT_FUNCTOR);
TF_CALL_half(DECLARE_PROJECT_FUNCTOR);
TF_CALL_float(DECLARE_PROJECT_FUNCTOR);
TF_CALL_double(DECLARE_PROJECT_FUNCTOR);

}  // end namespace functor

namespace generator {

#define DECLARE_MAP_FUNCTOR(Mode)                                         \
  template <>                                                             \
  float MapCoordinate<GPUDevice, Mode>::operator()(const float out_coord, \
                                                   const DenseIndex len); \
  extern template struct MapCoordinate<GPUDevice, Mode>

DECLARE_MAP_FUNCTOR(Mode::FILL_REFLECT);
DECLARE_MAP_FUNCTOR(Mode::FILL_WRAP);
DECLARE_MAP_FUNCTOR(Mode::FILL_CONSTANT);
DECLARE_MAP_FUNCTOR(Mode::FILL_NEAREST);

}  // end namespace generator

#define REGISTER(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV2") \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("dtype") \
                              .HostMemory("output_shape"),   \
                          ImageProjectiveTransformV2<GPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#define REGISTER(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransformV3") \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<TYPE>("dtype") \
                              .HostMemory("output_shape")    \
                              .HostMemory("fill_value"),     \
                          ImageProjectiveTransformV3<GPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
