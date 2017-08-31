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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/image/kernels/image_ops.h"
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
template struct FillProjectiveTransform<CPUDevice, int64>;
template struct FillProjectiveTransform<CPUDevice, float>;
template struct FillProjectiveTransform<CPUDevice, double>;

}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::FillProjectiveTransform;
using generator::INTERPOLATION_BILINEAR;
using generator::INTERPOLATION_NEAREST;
using generator::Interpolation;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
class ImageProjectiveTransform : public OpKernel {
 private:
  Interpolation interpolation_;

 public:
  explicit ImageProjectiveTransform(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string interpolation_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("interpolation", &interpolation_str));
    if (interpolation_str == "NEAREST") {
      interpolation_ = INTERPOLATION_NEAREST;
    } else if (interpolation_str == "BILINEAR") {
      interpolation_ = INTERPOLATION_BILINEAR;
    } else {
      LOG(FATAL) << "Invalid interpolation " << interpolation_str
                 << ". Supported types: NEAREST, BILINEAR";
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& images_t = ctx->input(0);
    const Tensor& transform_t = ctx->input(1);
    OP_REQUIRES(ctx, images_t.shape().dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));
    OP_REQUIRES(ctx, (TensorShapeUtils::IsMatrix(transform_t.shape()) &&
                      (transform_t.dim_size(0) == images_t.dim_size(0) ||
                       transform_t.dim_size(0) == 1) &&
                      transform_t.dim_size(1) ==
                          ProjectiveGenerator<Device, T>::kNumParameters),
                errors::InvalidArgument(
                    "Input transform should be num_images x 8 or 1 x 8"));
    auto images = images_t.tensor<T, 4>();
    auto transform = transform_t.matrix<float>();
    Tensor* output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, images_t.shape(), &output_t));
    auto output = output_t->tensor<T, 4>();
    (FillProjectiveTransform<Device, T>(interpolation_))(
        ctx->eigen_device<Device>(), &output, images, transform);
  }
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransform")    \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageProjectiveTransform<CPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// NOTE(ringwalt): We get an undefined symbol error if we don't explicitly
// instantiate the operator() in GCC'd code.
#define DECLARE_FUNCTOR(TYPE)                                               \
  template <>                                                               \
  void FillProjectiveTransform<GPUDevice, TYPE>::operator()(                \
      const GPUDevice& device, OutputType* output, const InputType& images, \
      const TransformsType& transform) const;                               \
  extern template struct FillProjectiveTransform<GPUDevice, TYPE>

TF_CALL_uint8(DECLARE_FUNCTOR);
TF_CALL_int32(DECLARE_FUNCTOR);
TF_CALL_int64(DECLARE_FUNCTOR);
TF_CALL_float(DECLARE_FUNCTOR);
TF_CALL_double(DECLARE_FUNCTOR);

}  // end namespace functor

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ImageProjectiveTransform")    \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ImageProjectiveTransform<GPUDevice, TYPE>)

TF_CALL_uint8(REGISTER);
TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
