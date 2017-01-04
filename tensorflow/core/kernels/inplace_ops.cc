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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class InplaceOpBase : public OpKernel {
 public:
  explicit InplaceOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto value = ctx->input(0);
    auto loc = ctx->input(1);
    auto update = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(loc.shape()),
                errors::InvalidArgument("loc must be a vector. ",
                                        loc.shape().DebugString()));
    OP_REQUIRES(
        ctx, value.dims() == update.dims(),
        errors::InvalidArgument("value and update shape doesn't match: ",
                                value.shape().DebugString(), " vs. ",
                                update.shape().DebugString()));
    for (int i = 1; i < value.dims(); ++i) {
      OP_REQUIRES(
          ctx, value.dim_size(i) == update.dim_size(i),
          errors::InvalidArgument("value and update shape doesn't match ",
                                  value.shape().DebugString(), " vs. ",
                                  update.shape().DebugString()));
    }
    OP_REQUIRES(ctx, loc.dim_size(0) == update.dim_size(0),
                errors::InvalidArgument("loc and update shape doesn't match: ",
                                        loc.shape().DebugString(), " vs. ",
                                        update.shape().DebugString()));

    Tensor output = value;  // This creates an alias intentionally.
    OP_REQUIRES_OK(ctx, DoCompute(ctx, update, loc, &output));
    ctx->set_output(0, output);
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& value,
                           const Tensor& loc, Tensor* output) = 0;
};

namespace functor {

template <typename T>
Status DoInplaceUpdate(const CPUDevice& d, InplaceOpType op,
                       const Tensor& value, const Tensor& loc, Tensor* output) {
  auto Tloc = loc.flat<int64>();
  auto Tvalue = value.flat_outer_dims<T>();
  auto Toutput = output->flat_outer_dims<T>();
  auto nrows = Toutput.dimension(0);
  for (int64 j = 0; j < Tloc.size(); ++j) {
    auto r = (Tloc(j) % nrows + nrows) % nrows;  // Guard index range.
    switch (op) {
      case I_UPDATE:
        Toutput.template chip<0>(r).device(d) = Tvalue.template chip<0>(j);
        break;
      case I_ADD:
        Toutput.template chip<0>(r).device(d) += Tvalue.template chip<0>(j);
        break;
      case I_SUB:
        Toutput.template chip<0>(r).device(d) -= Tvalue.template chip<0>(j);
        break;
      default:
        return errors::InvalidArgument("Unsupported inplace operation", op);
    }
  }
  return Status::OK();
}

template <>
Status DoInplace(const CPUDevice& d, InplaceOpType op, const Tensor& value,
                 const Tensor& loc, Tensor* output) {
  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                  \
  case DataTypeToEnum<type>::value: \
    return DoInplaceUpdate<type>(d, op, value, loc, output);
    TF_CALL_NUMBER_TYPES(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ", value.dtype());
  }
}

}  // end namespace functor

template <typename Device, functor::InplaceOpType op>
class InplaceOp : public InplaceOpBase {
 public:
  explicit InplaceOp(OpKernelConstruction* ctx) : InplaceOpBase(ctx) {}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& value, const Tensor& loc,
                   Tensor* output) override {
    const auto& d = ctx->eigen_device<Device>();
    return ::tensorflow::functor::DoInplace(d, op, value, loc, output);
  }
};

template <typename Device, typename T>
class EmptyOp : public OpKernel {
 public:
  explicit EmptyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("init", &init_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape = ctx->input(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(shape.shape()),
        errors::InvalidArgument("shape must be a vector of int32, got shape ",
                                shape.shape().DebugString()));
    auto dims = shape.flat<int32>();
    TensorShape out_shape;
    OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(
                            reinterpret_cast<const int32*>(dims.data()),
                            dims.size(), &out_shape));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (init_) {
      functor::SetZeroFunctor<Device, T>()(ctx->eigen_device<Device>(),
                                           out->flat<T>());
    }
  }

 private:
  bool init_;
};

#define REGISTER(type)                                                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("InplaceUpdate").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      InplaceOp<CPUDevice, functor::I_UPDATE>);                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("InplaceAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      InplaceOp<CPUDevice, functor::I_ADD>);                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("InplaceSubtract").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      InplaceOp<CPUDevice, functor::I_SUB>);
TF_CALL_NUMBER_TYPES(REGISTER)
#undef REGISTER

#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Empty")                       \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("shape")            \
                              .TypeConstraint<type>("dtype"), \
                          EmptyOp<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("Empty")                       \
                              .Device(DEVICE_GPU)             \
                              .HostMemory("shape")            \
                              .TypeConstraint<type>("dtype"), \
                          EmptyOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#define REGISTER(type)                                                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      InplaceOp<GPUDevice, functor::I_UPDATE>);                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("InplaceAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      InplaceOp<GPUDevice, functor::I_ADD>);                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("InplaceSubtract").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      InplaceOp<GPUDevice, functor::I_SUB>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER)
#undef REGISTER

// Register versions that operate on int32 data on the CPU even though the op
// has been placed on the GPU

REGISTER_KERNEL_BUILDER(Name("InplaceUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("loc")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_UPDATE>);

REGISTER_KERNEL_BUILDER(Name("InplaceAdd")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("loc")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_ADD>);

REGISTER_KERNEL_BUILDER(Name("InplaceSubtract")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("loc")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_SUB>);
#endif

}  // end namespace tensorflow
