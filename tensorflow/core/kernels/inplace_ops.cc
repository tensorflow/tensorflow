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

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename Device, typename T>
Status DoParallelConcatUpdate(const Device& d, const Tensor& value, int32_t loc,
                              Tensor* output) {
  auto Tvalue = value.shaped<T, 2>({1, value.NumElements()});
  auto Toutput = output->flat_outer_dims<T>();
  auto nrows = Toutput.dimension(0);
  auto r = (loc % nrows + nrows) % nrows;  // Guard index range.
  Toutput.template chip<0>(r).device(d) = Tvalue.template chip<0>(0);
  return OkStatus();
}

template <>
Status DoParallelConcat(const CPUDevice& d, const Tensor& value, int32_t loc,
                        Tensor* output) {
  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                  \
  case DataTypeToEnum<type>::value: \
    return DoParallelConcatUpdate<CPUDevice, type>(d, value, loc, output);
    TF_CALL_POD_TYPES(CASE);
    TF_CALL_tstring(CASE);
    TF_CALL_variant(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(value.dtype()));
  }
}

}  // end namespace functor

namespace {

template <typename Device>
class ParallelConcatUpdate : public OpKernel {
 public:
  explicit ParallelConcatUpdate(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("loc", &loc_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto value = ctx->input(0);
    // Value should be at least rank 1. Also the 0th dimension should be
    // at least loc_.
    OP_REQUIRES(ctx, value.dims() >= 1,
                errors::InvalidArgument("value should be at least rank 1."));
    OP_REQUIRES(
        ctx, value.dim_size(0) > loc_,
        errors::InvalidArgument("0th dimension of value = ", value.dim_size(0),
                                " must be greater than loc_ = ", loc_));

    auto update = ctx->input(1);

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
    OP_REQUIRES(ctx, 1 == update.dim_size(0),
                errors::InvalidArgument("update shape doesn't match: ",
                                        update.shape().DebugString()));

    Tensor output = value;  // This creates an alias intentionally.
    const auto& d = ctx->eigen_device<Device>();
    OP_REQUIRES_OK(
        ctx, ::tensorflow::functor::DoParallelConcat(d, update, loc_, &output));
    ctx->set_output(0, output);
  }

 private:
  int32 loc_;
};

template <typename Device, typename T>
class ParallelConcatStart : public OpKernel {
 public:
  explicit ParallelConcatStart(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor* out = nullptr;
    // We do not know whether the output will be used on GPU. Setting it to be
    // gpu-compatible for now.
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape_, &out, attr));
  }

 private:
  TensorShape shape_;
};

class FailureKernel : public OpKernel {
 public:
  explicit FailureKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   errors::Internal("Found instance of parallel_stack which "
                                    "could not be properly replaced."));
  }

  void Compute(OpKernelContext*) override {}
};

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<CPUDevice>);
TF_CALL_POD_STRING_TYPES(REGISTER)
#undef REGISTER

#define REGISTER_EMPTY(type)                                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_EMPTY)
#undef REGISTER_EMPTY

#define REGISTER_PARALLEL_CONCAT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_POD_STRING_TYPES(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_PARALLEL_CONCAT_START(type)                  \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatStart")        \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("dtype"), \
                          ParallelConcatStart<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_PARALLEL_CONCAT_START);
#undef REGISTER_PARALLEL_CONCAT_START

#define REGISTER_PARALLEL_CONCAT(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ParallelConcat").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      FailureKernel);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_PARALLEL_CONCAT);
#undef REGISTER_PARALLEL_CONCAT

#define REGISTER(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")   \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          ParallelConcatUpdate<GPUDevice>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER);
#undef REGISTER

// Register versions that operate on int32 data on the CPU even though the op
// has been placed on the GPU

REGISTER_KERNEL_BUILDER(Name("_ParallelConcatUpdate")
                            .Device(DEVICE_GPU)
                            .HostMemory("value")
                            .HostMemory("update")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        ParallelConcatUpdate<CPUDevice>);
#endif

class InplaceOpBase : public OpKernel {
 public:
  explicit InplaceOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto x = ctx->input(0);
    auto i = ctx->input(1);
    auto v = ctx->input(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(i.shape()),
                errors::InvalidArgument("i must be a vector. ",
                                        i.shape().DebugString()));
    OP_REQUIRES(ctx, x.dims() == v.dims(),
                errors::InvalidArgument(
                    "x and v shape doesn't match (ranks differ): ",
                    x.shape().DebugString(), " vs. ", v.shape().DebugString()));
    for (int i = 1; i < x.dims(); ++i) {
      OP_REQUIRES(
          ctx, x.dim_size(i) == v.dim_size(i),
          errors::InvalidArgument("x and v shape doesn't match at index ", i,
                                  " : ", x.shape().DebugString(), " vs. ",
                                  v.shape().DebugString()));
    }
    OP_REQUIRES(ctx, i.dim_size(0) == v.dim_size(0),
                errors::InvalidArgument(
                    "i and x shape doesn't match at index 0: ",
                    i.shape().DebugString(), " vs. ", v.shape().DebugString()));

    Tensor y = x;  // This creates an alias intentionally.
    // Skip processing if tensors are empty.
    if (x.NumElements() > 0 && v.NumElements() > 0) {
      OP_REQUIRES_OK(ctx, DoCompute(ctx, i, v, &y));
    }
    ctx->set_output(0, y);
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& i,
                           const Tensor& v, Tensor* y) = 0;
};

}  // end namespace

namespace functor {

template <typename T>
void DoInplaceOp(const CPUDevice& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  auto Ti = i.flat<int32>();
  auto Tv = v.flat_outer_dims<T>();
  auto Ty = y->flat_outer_dims<T>();
  auto nrows = Ty.dimension(0);
  for (int64_t j = 0; j < Ti.size(); ++j) {
    auto r = (Ti(j) % nrows + nrows) % nrows;  // Guard index range.
    switch (op) {
      case I_UPDATE:
        Ty.template chip<0>(r).device(d) = Tv.template chip<0>(j);
        break;
      case I_ADD:
        Ty.template chip<0>(r).device(d) += Tv.template chip<0>(j);
        break;
      case I_SUB:
        Ty.template chip<0>(r).device(d) -= Tv.template chip<0>(j);
        break;
    }
  }
}

// String type only supports inplace update.
void DoInplaceStringUpdateOp(const CPUDevice& d, const Tensor& i,
                             const Tensor& v, Tensor* y) {
  auto Ti = i.flat<int32>();
  auto Tv = v.flat_outer_dims<tstring>();
  auto Ty = y->flat_outer_dims<tstring>();
  auto nrows = Ty.dimension(0);
  for (int64_t j = 0; j < Ti.size(); ++j) {
    auto r = (Ti(j) % nrows + nrows) % nrows;  // Guard index range.
    Ty.template chip<0>(r).device(d) = Tv.template chip<0>(j);
  }
}

template <>
Status DoInplace(const CPUDevice& device, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  CHECK_EQ(v.dtype(), y->dtype());
  if (op == I_UPDATE) {
    if (v.dtype() == DT_STRING) {
      DoInplaceStringUpdateOp(device, i, v, y);
      return OkStatus();
    } else if (v.dtype() == DT_BOOL) {
      DoInplaceOp<bool>(device, op, i, v, y);
      return OkStatus();
    }
  }
  switch (v.dtype()) {
#define CASE(type)                          \
  case DataTypeToEnum<type>::value:         \
    DoInplaceOp<type>(device, op, i, v, y); \
    break;
    TF_CALL_NUMBER_TYPES(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(v.dtype()));
  }
  return OkStatus();
}

}  // end namespace functor

namespace {
template <typename Device, functor::InplaceOpType op>
class InplaceOp : public InplaceOpBase {
 public:
  explicit InplaceOp(OpKernelConstruction* ctx) : InplaceOpBase(ctx) {}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& i, const Tensor& v,
                   Tensor* y) override {
    const auto& d = ctx->eigen_device<Device>();
    return ::tensorflow::functor::DoInplace(d, op, i, v, y);
  }
};

class CopyOpBase : public OpKernel {
 public:
  explicit CopyOpBase(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto x = ctx->input(0);
    Tensor* y;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, DoCompute(ctx, x, y));
  }

 protected:
  virtual Status DoCompute(OpKernelContext* ctx, const Tensor& x,
                           Tensor* y) = 0;
};

template <typename Device>
class CopyOp : public CopyOpBase {
 public:
  explicit CopyOp(OpKernelConstruction* ctx) : CopyOpBase(ctx) {}

 protected:
  Status DoCompute(OpKernelContext* ctx, const Tensor& x, Tensor* y) override {
    const auto& d = ctx->eigen_device<Device>();
    return ::tensorflow::functor::DoCopy(d, x, y);
  }
};

}  // end namespace

namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <>
Status DoCopy(const CPUDevice& device, const Tensor& x, Tensor* y) {
  CHECK_EQ(x.dtype(), y->dtype());
  switch (x.dtype()) {
#define CASE(type)                                   \
  case DataTypeToEnum<type>::value:                  \
    y->flat<type>().device(device) = x.flat<type>(); \
    break;

    TF_CALL_NUMBER_TYPES(CASE);
    TF_CALL_bool(CASE);
    TF_CALL_tstring(CASE);
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(x.dtype()));
  }
  return OkStatus();
}

}  // end namespace functor

namespace {
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

REGISTER_KERNEL_BUILDER(Name("InplaceUpdate").Device(DEVICE_CPU),
                        InplaceOp<CPUDevice, functor::I_UPDATE>);
REGISTER_KERNEL_BUILDER(Name("InplaceAdd").Device(DEVICE_CPU),
                        InplaceOp<CPUDevice, functor::I_ADD>);
REGISTER_KERNEL_BUILDER(Name("InplaceSub").Device(DEVICE_CPU),
                        InplaceOp<CPUDevice, functor::I_SUB>);
REGISTER_KERNEL_BUILDER(Name("DeepCopy").Device(DEVICE_CPU), CopyOp<CPUDevice>);

#define REGISTER_EMPTY(type, dev)                             \
  REGISTER_KERNEL_BUILDER(Name("Empty")                       \
                              .Device(DEVICE_##dev)           \
                              .HostMemory("shape")            \
                              .TypeConstraint<type>("dtype"), \
                          EmptyOp<dev##Device, type>)

REGISTER_EMPTY(float, CPU)
REGISTER_EMPTY(bfloat16, CPU)
REGISTER_EMPTY(double, CPU)
REGISTER_EMPTY(Eigen::half, CPU)
REGISTER_EMPTY(tstring, CPU)
REGISTER_EMPTY(int32, CPU)
REGISTER_EMPTY(int64_t, CPU)
REGISTER_EMPTY(bool, CPU)
REGISTER_EMPTY(uint8, CPU)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER(TYPE)                                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      InplaceOp<GPUDevice, functor::I_UPDATE>);                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceAdd").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      InplaceOp<GPUDevice, functor::I_ADD>);                              \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InplaceSub").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),    \
      InplaceOp<GPUDevice, functor::I_SUB>);                              \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DeepCopy").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),      \
      CopyOp<GPUDevice>);

REGISTER_KERNEL_BUILDER(
    Name("InplaceUpdate").Device(DEVICE_GPU).TypeConstraint<bool>("T"),
    InplaceOp<GPUDevice, functor::I_UPDATE>);
REGISTER(float);
REGISTER(double);
REGISTER(Eigen::half);
REGISTER(Eigen::bfloat16);
REGISTER(int64_t);

REGISTER_EMPTY(float, GPU);
REGISTER_EMPTY(double, GPU);
REGISTER_EMPTY(Eigen::half, GPU);
REGISTER_EMPTY(Eigen::bfloat16, GPU);
REGISTER_EMPTY(int64_t, GPU);
REGISTER_EMPTY(int32, GPU);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER_KERNEL_BUILDER(Name("InplaceUpdate")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("i")
                            .HostMemory("v")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_UPDATE>);
REGISTER_KERNEL_BUILDER(Name("InplaceAdd")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("i")
                            .HostMemory("v")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_ADD>);
REGISTER_KERNEL_BUILDER(Name("InplaceSub")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("i")
                            .HostMemory("v")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        InplaceOp<CPUDevice, functor::I_SUB>);

REGISTER_KERNEL_BUILDER(Name("DeepCopy")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        CopyOp<CPUDevice>);

}  // end namespace
}  // end namespace tensorflow
