/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "dropout_op.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct ApplyDropout<CPUDevice, T> {
  void operator()(const CPUDevice& d, T* out, uint8* mask, const T* in,
                  const float* rng_data, float rate, uint64 num_elements,
                  random::PhiloxRandom gen, bool seeded);
};

template <typename T>
struct ApplyDropoutGrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, T* outgrads, const T* grads, const uint8* mask,
                  float rate, uint64 num_elements);
};

template <typename T>
void ApplyDropout<CPUDevice, T>::operator()(const CPUDevice& d, T* out, uint8* mask, 
                                            const T* in, const float* rng_data,
                                            float rate, uint64 num_elements,
                                            random::PhiloxRandom gen,
                                            bool seeded) {
  T scale = T(1. / (1. - rate));
  for (uint64 i = 0; i < num_elements; i++) {
    bool b = (rng_data[i] > rate);
    out[i] = b ? in[i] * scale : T(0.0);
    mask[i] = (uint8)b;
  }
}

template <typename T>
void ApplyDropoutGrad<CPUDevice, T>::operator()(const CPUDevice& d, T* outgrads,
                                                const T* grads, const uint8* mask,
                                                float rate,
                                                uint64 num_elements) {
  T scale = T(1. / (1 - rate));
  for (uint64 i = 0; i < num_elements; i++) {
    outgrads[i] = mask[i] ? (grads[i] * scale) : T(0);
  }
};

template <typename Device, typename T>
class DropoutOp : public OpKernel {
 private:
  // todo: may be sufficient to use random::PhiloxRandom, since we don't
  // require Compute() to be reentrant
  GuardedPhiloxRandom generator_;

 public:
  explicit DropoutOp(OpKernelConstruction* context) : OpKernel(context) {
    generator_.Init(0, 0);
  }

  ~DropoutOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);

    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dtype() == in1.dtype(),
                errors::InvalidArgument(
                    "Dropout rate must be same type as input tensor."));
    OP_REQUIRES(
        ctx, in1.dims() == 0,
        errors::InvalidArgument("Dropout rate must be a scalar tensor."));
    float rate = static_cast<float>(in1.scalar<T>()());

    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx, in0.dims() == in2.shape().num_elements(),
                errors::InvalidArgument("MIOpen only supports input dimensions "
                                        "to match noise dimensions."));
    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &output));
    if (output->NumElements() == 0) return;

    Tensor* mask;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, in0.shape(), &mask));

    const Tensor& in3 = ctx->input(3);
    OP_REQUIRES(
        ctx, in3.dims() == 0,
        errors::InvalidArgument("Dropout seed must be a scalar tensor."));
    int64 seed = 0;
    if (in3.dtype() == DT_INT32)
      seed = in3.scalar<int32>()();
    else
      seed = in3.scalar<int64>()();
    // don't reset the seed for every call unless it is explicitly non-0
    if (seed != 0) 
      generator_.ResetSeeds(seed, 0);
    else
      generator_.ResetSeeds(random::New64(), 0);

    typedef random::UniformDistribution<random::PhiloxRandom, float>
        Distribution;
    Distribution dist;
    random::PhiloxRandom gen =
        generator_.ReserveRandomOutputs(in0.NumElements(), 256);

    if (std::is_same<Device, CPUDevice>::value) {
      Tensor rng_data;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape(in0.shape()), &rng_data));
      auto rng_flat = rng_data.flat<float>();

      functor::FillPhiloxRandom<Device, Distribution>()(
          ctx, ctx->eigen_device<Device>(), /*key=*/nullptr,
	  /*counter=*/nullptr, gen, rng_flat.data(), rng_flat.size(), dist);
      ApplyDropout<Device, T>()(ctx->eigen_device<Device>(),
                                output->flat<T>().data(),
                                mask->flat<uint8>().data(),
                                in0.flat<T>().data(),
                                rng_flat.data(), rate, in0.NumElements(), gen,
                                seed != 0);
    } else {
      ApplyDropout<Device, T>()(ctx->eigen_device<Device>(),
                                output->flat<T>().data(),
                                mask->flat<uint8>().data(),
                                in0.flat<T>().data(),
                                nullptr, rate, in0.NumElements(), gen,
                                seed != 0);
    }
  }
};

#define REGISTER_DROPOUT(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Dropout").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      DropoutOp<CPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT);
TF_CALL_float(REGISTER_DROPOUT);
TF_CALL_half(REGISTER_DROPOUT);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_DROPOUT_GPU(TYPE)                        \
  REGISTER_KERNEL_BUILDER(Name("Dropout")                 \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T")  \
                              .HostMemory("rate")         \
                              .HostMemory("seed")         \
                              .HostMemory("noise_shape"), \
                          DropoutOp<GPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT_GPU);
TF_CALL_float(REGISTER_DROPOUT_GPU);
TF_CALL_half(REGISTER_DROPOUT_GPU);
#endif

template <typename Device, typename T>
class DropoutGradOp : public OpKernel {
 public:
  explicit DropoutGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  ~DropoutGradOp() override {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, grad.dtype() == in1.dtype(),
                errors::InvalidArgument(
                    "Dropout rate must be same type as input tensor."));
    OP_REQUIRES(
        ctx, in1.dims() == 0,
        errors::InvalidArgument("Dropout rate must be a scalar tensor."));
    float rate = static_cast<float>(in1.scalar<T>()());

    const Tensor& mask = ctx->input(2);
    OP_REQUIRES(ctx, grad.NumElements() == mask.NumElements(),
                errors::InvalidArgument("ROCm DropoutGrad dim mismatch"));

    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad.shape(), &output));
    if (output->NumElements() == 0) return;

    ApplyDropoutGrad<Device, T>()(
        ctx->eigen_device<Device>(), output->flat<T>().data(),
        grad.flat<T>().data(),
        mask.flat<uint8>().data(),
        rate,
        grad.NumElements());
  }
};

#define REGISTER_DROPOUT_GRAD_CPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DropoutGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      DropoutGradOp<CPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT_GRAD_CPU);
TF_CALL_float(REGISTER_DROPOUT_GRAD_CPU);
TF_CALL_half(REGISTER_DROPOUT_GRAD_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_DROPOUT_GRAD_GPU(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("DropoutGrad")             \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T")  \
                              .HostMemory("rate"),        \
                          DropoutGradOp<GPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT_GRAD_GPU);
TF_CALL_float(REGISTER_DROPOUT_GRAD_GPU);
TF_CALL_half(REGISTER_DROPOUT_GRAD_GPU);
#endif

}  // namespace tensorflow
