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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#define FAKE_QUANT_NO_DEBUG

#include "tensorflow/core/kernels/fake_quant_ops_functor.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::BinaryElementWiseOp;
using tensorflow::DEVICE_CPU;
#if GOOGLE_CUDA
using tensorflow::DEVICE_GPU;
#endif
using tensorflow::DT_BOOL;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::PersistentTensor;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;  // NOLINT This is needed in CUDA mode, do not remove.
using tensorflow::UnaryElementWiseOp;
using tensorflow::errors::InvalidArgument;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxArgsOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxArgsOp
    : public UnaryElementWiseOp<float, FakeQuantWithMinMaxArgsOp<Device>> {
 public:
  typedef UnaryElementWiseOp<float, FakeQuantWithMinMaxArgsOp<Device>> Base;
  explicit FakeQuantWithMinMaxArgsOp(OpKernelConstruction* context)
      : Base::UnaryElementWiseOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
    OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
    OP_REQUIRES(context, min_ < max_,
                InvalidArgument("min has to be smaller than max, was: ", min_,
                                " >= ", max_));
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    FakeQuantWithMinMaxArgsFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(), min_, max_,
            output->flat<float>());
  }
 private:
  float min_;
  float max_;
};

// Implementation of FakeQuantWithMinMaxArgsGradientOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxArgsGradientOp
    : public BinaryElementWiseOp<float,
                                 FakeQuantWithMinMaxArgsGradientOp<Device>> {
 public:
  typedef BinaryElementWiseOp<float, FakeQuantWithMinMaxArgsGradientOp<Device>>
      Base;
  explicit FakeQuantWithMinMaxArgsGradientOp(OpKernelConstruction* context)
      : Base::BinaryElementWiseOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("min", &min_));
    OP_REQUIRES_OK(context, context->GetAttr("max", &max_));
    OP_REQUIRES(context, min_ < max_,
                InvalidArgument("min has to be smaller than max, was: ", min_,
                                " >= ", max_));
  }

  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& gradient,
               const Tensor& input, Tensor* output) {
    OperateNoTemplate(context, gradient, input, output);
  }

  void OperateNoTemplate(OpKernelContext* context, const Tensor& gradient,
                         const Tensor& input, Tensor* output) {
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    FakeQuantWithMinMaxArgsGradientFunctor<Device> functor;
    functor(context->eigen_device<Device>(), gradient.flat<float>(),
            input.flat<float>(), min_, max_, output->flat<float>());
  }
 private:
  float min_;
  float max_;
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxArgs").Device(DEVICE_CPU),
                        FakeQuantWithMinMaxArgsOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxArgsGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxArgsGradientOp<CPUDevice>);

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;

// Forward declarations for functor specializations for GPU.
template <>
void FakeQuantWithMinMaxArgsFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    typename TTypes<float>::ConstFlat inputs,
    const float min, const float max,
    typename TTypes<float>::Flat outputs);
extern template struct FakeQuantWithMinMaxArgsFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxArgs").Device(DEVICE_GPU),
                        FakeQuantWithMinMaxArgsOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxArgsGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs,
    const float min, const float max,
    typename TTypes<float>::Flat backprops);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxArgsGradient").Device(DEVICE_GPU),
    FakeQuantWithMinMaxArgsGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxVarsOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
#ifndef FAKE_QUANT_NO_DEBUG
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_BOOL, {},
                                                &check_min_max_handle_,
                                                nullptr));
#endif
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(3, context->num_inputs());
    const Tensor& input = context->input(0);
    const Tensor& min = context->input(1);
    const Tensor& max = context->input(2);
#ifndef FAKE_QUANT_NO_DEBUG
    Tensor* check_min_max = check_min_max_handle_.AccessTensor(context);
#endif

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    FakeQuantWithMinMaxVarsFunctor<Device> functor;
    functor(context->eigen_device<Device>(), input.flat<float>(),
            min.scalar<float>(), max.scalar<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
            check_min_max->scalar<bool>(),
#endif
            output->flat<float>());
  }

 private:
#ifndef FAKE_QUANT_NO_DEBUG
  PersistentTensor check_min_max_handle_;
#endif
};

// Implementation of FakeQuantWithMinMaxVarsGradientOp, see its documentation in
// core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsGradientOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsGradientOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
#ifndef FAKE_QUANT_NO_DEBUG
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_BOOL, {},
                                                &check_min_max_handle_,
                                                nullptr));
#endif
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(4, context->num_inputs());
    const Tensor& gradient = context->input(0);
    const Tensor& input = context->input(1);
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    const Tensor& min = context->input(2);
    const Tensor& max = context->input(3);
#ifndef FAKE_QUANT_NO_DEBUG
    Tensor* check_min_max = check_min_max_handle_.AccessTensor(context);
#endif

    Tensor* grad_wrt_input;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &grad_wrt_input));

    TensorShape scalar_shape;
    Tensor* grad_wrt_min;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, scalar_shape, &grad_wrt_min));

    Tensor* grad_wrt_max;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, scalar_shape, &grad_wrt_max));

    FakeQuantWithMinMaxVarsGradientFunctor<Device> functor;
    functor(context->eigen_device<Device>(), gradient.flat<float>(),
            input.flat<float>(), min.scalar<float>(), max.scalar<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
            check_min_max->scalar<bool>(),
#endif
            grad_wrt_input->flat<float>(), grad_wrt_min->scalar<float>(),
            grad_wrt_max->scalar<float>());
  }

 private:
#ifndef FAKE_QUANT_NO_DEBUG
  PersistentTensor check_min_max_handle_;
#endif
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVars").Device(DEVICE_CPU),
                        FakeQuantWithMinMaxVarsOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(
    Name("FakeQuantWithMinMaxVarsGradient").Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsGradientOp<CPUDevice>);

#if GOOGLE_CUDA
template <>
void FakeQuantWithMinMaxVarsFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstScalar min,
    typename TTypes<float>::ConstScalar max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Flat output);
extern template struct FakeQuantWithMinMaxVarsFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVars")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsOp<GPUDevice>);

template <>
void FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstScalar min,
    typename TTypes<float>::ConstScalar max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Flat backprops_wrt_input,
    typename TTypes<float>::Scalar backprop_wrt_min,
    typename TTypes<float>::Scalar backprop_wrt_max);
extern template struct FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA

// -----------------------------------------------------------------------------
// Implementation of FakeQuantWithMinMaxVarsPerChannelOp, see its documentation
// in core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsPerChannelOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelOp(OpKernelConstruction* context)
      : OpKernel::OpKernel(context) {
#ifndef FAKE_QUANT_NO_DEBUG
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_BOOL, {},
                                                &check_min_max_handle_,
                                                nullptr));
#endif
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(3, context->num_inputs());
    const Tensor& input = context->input(0);
    const int depth = input.dim_size(input.dims() - 1);  // last dimension size.
    const Tensor& min = context->input(1);
    OP_REQUIRES(context, min.dim_size(0) == depth,
                InvalidArgument("min has incorrect size, expected ", depth,
                                " was ", min.dim_size(0)));
    const Tensor& max = context->input(2);
    OP_REQUIRES(context, max.dim_size(0) == depth,
                InvalidArgument("max has incorrect size, expected ", depth,
                                " was ", max.dim_size(0)));
#ifndef FAKE_QUANT_NO_DEBUG
    Tensor* check_min_max = check_min_max_handle_.AccessTensor(context);
#endif

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    switch (input.dims()) {
      case 4: {
        FakeQuant4WithMinMaxVarsPerChannelFunctor<Device> functor;
        functor(context->eigen_device<Device>(), input.dim_size(0),
                input.dim_size(1), input.dim_size(2), input.dim_size(3),
                input.flat<float>(), min.vec<float>(), max.vec<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
                check_min_max->scalar<bool>(),
#endif
                output->flat<float>());
        break;
      }
      case 2: {
        FakeQuant2WithMinMaxVarsPerChannelFunctor<Device> functor;
        functor(context->eigen_device<Device>(),
                input.dim_size(0), input.dim_size(1),
                input.flat<float>(), min.vec<float>(), max.vec<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
                check_min_max->scalar<bool>(),
#endif
                output->flat<float>());
        break;
      }
      case 1: {
        FakeQuant1WithMinMaxVarsPerChannelFunctor<Device> functor;
        functor(context->eigen_device<Device>(),
                input.vec<float>(), min.vec<float>(), max.vec<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
                check_min_max->scalar<bool>(),
#endif
                output->vec<float>());
        break;
      }
      default:
        context->SetStatus(InvalidArgument("Only inputs of dimensions 1, 2 or "
                                           "4 supported, was: ", input.dims()));
        break;
    }
  }

 private:
#ifndef FAKE_QUANT_NO_DEBUG
  PersistentTensor check_min_max_handle_;
#endif
};

// Implementation of FakeQuantWithMinMaxVarsPerChannelGradientOp, see its
// documentation in core/ops/array_ops.cc.
template <typename Device>
class FakeQuantWithMinMaxVarsPerChannelGradientOp : public OpKernel {
 public:
  explicit FakeQuantWithMinMaxVarsPerChannelGradientOp(
      OpKernelConstruction* context) : OpKernel::OpKernel(context) {
#ifndef FAKE_QUANT_NO_DEBUG
    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_BOOL, {},
                                                &check_min_max_handle_,
                                                nullptr));
#endif
  }

  void Compute(OpKernelContext* context) override {
    CHECK_EQ(4, context->num_inputs());
    const Tensor& gradient = context->input(0);
    const Tensor& input = context->input(1);
    OP_REQUIRES(context, input.IsSameSize(gradient),
                InvalidArgument("gradient and input must be the same size"));
    const int depth = input.dim_size(input.dims() - 1);  // last dimension size.
    const Tensor& min = context->input(2);
    OP_REQUIRES(context, min.dim_size(0) == depth,
                InvalidArgument("min has incorrect size, expected ", depth,
                                " was ", min.dim_size(0)));
    const Tensor& max = context->input(3);
    OP_REQUIRES(context, max.dim_size(0) == depth,
                InvalidArgument("max has incorrect size, expected ", depth,
                                " was ", max.dim_size(0)));
#ifndef FAKE_QUANT_NO_DEBUG
    Tensor* check_min_max = check_min_max_handle_.AccessTensor(context);
#endif

    Tensor* grad_wrt_input;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &grad_wrt_input));

    TensorShape min_max_shape({input.dim_size(input.dims() - 1)});
    Tensor* grad_wrt_min;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, min_max_shape, &grad_wrt_min));

    Tensor* grad_wrt_max;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, min_max_shape, &grad_wrt_max));

    switch (input.dims()) {
      case 4: {
        FakeQuant4WithMinMaxVarsPerChannelGradientFunctor<Device> functor;
        functor(context->eigen_device<Device>(), input.dim_size(0),
                input.dim_size(1), input.dim_size(2), input.dim_size(3),
                gradient.flat<float>(), input.flat<float>(),
                min.vec<float>(), max.vec<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
                check_min_max->scalar<bool>(),
#endif
                grad_wrt_input->flat<float>(),
                grad_wrt_min->vec<float>(), grad_wrt_max->vec<float>());
        break;
      }
      case 2: {
        FakeQuant2WithMinMaxVarsPerChannelGradientFunctor<Device> functor;
        functor(context->eigen_device<Device>(),
                input.dim_size(0), input.dim_size(1),
                gradient.flat<float>(), input.flat<float>(),
                min.vec<float>(), max.vec<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
                check_min_max->scalar<bool>(),
#endif
                grad_wrt_input->flat<float>(),
                grad_wrt_min->vec<float>(), grad_wrt_max->vec<float>());
        break;
      }
      case 1: {
        FakeQuant1WithMinMaxVarsPerChannelGradientFunctor<Device> functor;
        functor(context->eigen_device<Device>(),
                gradient.vec<float>(), input.vec<float>(),
                min.vec<float>(), max.vec<float>(),
#ifndef FAKE_QUANT_NO_DEBUG
                check_min_max->scalar<bool>(),
#endif
                grad_wrt_input->vec<float>(),
                grad_wrt_min->vec<float>(), grad_wrt_max->vec<float>());
        break;
      }
      default:
        context->SetStatus(InvalidArgument("Only inputs of dimensions 1, 2 or "
                                           "4 supported, was: ", input.dims()));
        break;
    }
  }

 private:
#ifndef FAKE_QUANT_NO_DEBUG
  PersistentTensor check_min_max_handle_;
#endif
};

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannel")
                            .Device(DEVICE_CPU),
                        FakeQuantWithMinMaxVarsPerChannelOp<CPUDevice>);
REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannelGradient")
                            .Device(DEVICE_CPU),
    FakeQuantWithMinMaxVarsPerChannelGradientOp<CPUDevice>);

#if GOOGLE_CUDA
template <>
void FakeQuant1WithMinMaxVarsPerChannelFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    typename TTypes<float>::ConstVec inputs,
    typename TTypes<float>::ConstVec min,
    typename TTypes<float>::ConstVec max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Vec outputs);
extern template struct FakeQuant1WithMinMaxVarsPerChannelFunctor<GPUDevice>;

template <>
void FakeQuant2WithMinMaxVarsPerChannelFunctor<GPUDevice>::operator()(
    const GPUDevice& d, const Index batch_size, const Index depth,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstFlat min,
    typename TTypes<float>::ConstFlat max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Flat outputs);
extern template struct FakeQuant2WithMinMaxVarsPerChannelFunctor<GPUDevice>;

template <>
void FakeQuant4WithMinMaxVarsPerChannelFunctor<GPUDevice>::operator()(
    const GPUDevice& d, const Index batch_size, const Index height,
    const Index width, const Index depth,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstFlat min,
    typename TTypes<float>::ConstFlat max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Flat outputs);
extern template struct FakeQuant4WithMinMaxVarsPerChannelFunctor<GPUDevice>;

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannel")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsPerChannelOp<GPUDevice>);

template <>
void FakeQuant1WithMinMaxVarsPerChannelGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d,
    typename TTypes<float>::ConstVec gradients,
    typename TTypes<float>::ConstVec inputs,
    typename TTypes<float>::ConstVec min,
    typename TTypes<float>::ConstVec max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Vec backprops_wrt_input,
    typename TTypes<float>::Vec backprop_wrt_min,
    typename TTypes<float>::Vec backprop_wrt_max);
extern template struct
    FakeQuant1WithMinMaxVarsPerChannelGradientFunctor<GPUDevice>;

template <>
void FakeQuant2WithMinMaxVarsPerChannelGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, const Index batch_size, const Index depth,
    typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstVec min,
    typename TTypes<float>::ConstVec max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Flat backprops_wrt_input,
    typename TTypes<float>::Vec backprop_wrt_min,
    typename TTypes<float>::Vec backprop_wrt_max);
extern template struct
    FakeQuant2WithMinMaxVarsPerChannelGradientFunctor<GPUDevice>;

template <>
void FakeQuant4WithMinMaxVarsPerChannelGradientFunctor<GPUDevice>::operator()(
    const GPUDevice& d, const Index batch_size, const Index height,
    const Index width, const Index depth,
    typename TTypes<float>::ConstFlat gradients,
    typename TTypes<float>::ConstFlat inputs,
    typename TTypes<float>::ConstVec min,
    typename TTypes<float>::ConstVec max,
#ifndef FAKE_QUANT_NO_DEBUG
    typename TTypes<bool>::Scalar check_min_max,
#endif
    typename TTypes<float>::Flat backprops_wrt_input,
    typename TTypes<float>::Vec backprop_wrt_min,
    typename TTypes<float>::Vec backprop_wrt_max);
extern template struct
    FakeQuant4WithMinMaxVarsPerChannelGradientFunctor<GPUDevice>;

REGISTER_KERNEL_BUILDER(Name("FakeQuantWithMinMaxVarsPerChannelGradient")
                            .Device(DEVICE_GPU)
                            .HostMemory("min")
                            .HostMemory("max"),
                        FakeQuantWithMinMaxVarsPerChannelGradientOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
