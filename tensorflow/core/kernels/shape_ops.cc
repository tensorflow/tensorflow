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

// See docs in ../ops/array_ops.cc.

#include <unordered_set>

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

template <typename OutType>
class ShapeOp : public OpKernel {
 public:
  explicit ShapeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    const int rank = inp.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rank}), &out));
    auto vec = out->vec<OutType>();
    for (int i = 0; i < rank; ++i) {
      int64 dim_size = inp.dim_size(i);
      if (out->dtype() == DT_INT32) {
        OP_REQUIRES(
            ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
            errors::InvalidArgument("Shape output type is 32-bit ", " but dim ",
                                    i, " is ", dim_size));
      }
      vec(i) = static_cast<OutType>(dim_size);
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int32>);                       \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_SYCL)               \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int64>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_SYCL_KERNEL);
#undef REGISTER_SYCL_KERNEL

REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);
#endif // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int32>);                       \
  REGISTER_KERNEL_BUILDER(Name("Shape")                          \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeOp<int64>);

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Shape")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeOp<int64>);
#endif

template <typename OutType>
class ShapeNOp : public OpKernel {
 public:
  explicit ShapeNOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      auto shape = ctx->input(i).shape();
      const int dims = shape.dims();
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, {dims}, &out));
      auto vec = out->vec<OutType>();

      for (int j = 0; j < dims; ++j) {
        int64 dim_size = shape.dim_size(j);
        if (out->dtype() == DT_INT32) {
          OP_REQUIRES(
              ctx, FastBoundsCheck(dim_size, std::numeric_limits<int32>::max()),
              errors::InvalidArgument("ShapeN output type is 32-bit but shape ",
                                      i, " dim ", j, " is ", dim_size));
        }
        vec(j) = static_cast<OutType>(dim_size);
      }
    }
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int32>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int32>);                      \
  REGISTER_KERNEL_BUILDER(Name("ShapeN")                         \
                              .Device(DEVICE_GPU)                \
                              .HostMemory("output")              \
                              .TypeConstraint<int64>("out_type") \
                              .TypeConstraint<type>("T"),        \
                          ShapeNOp<int64>)

TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type"),
                        ShapeNOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ShapeN")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type"),
                        ShapeNOp<int64>);
#endif

class RankOp : public OpKernel {
 public:
  explicit RankOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    const int rank = inp.dims();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    out->scalar<int32>()() = rank;
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_CPU).HostMemory("output"),
                        RankOp);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Rank")                    \
                              .Device(DEVICE_SYCL)        \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("output"),      \
                          RankOp);
REGISTER_SYCL_KERNEL(float);
#undef REGISTER_SYCL_KERNEL

// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_SYCL)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
#endif // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                        \
  REGISTER_KERNEL_BUILDER(Name("Rank")                   \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("output"),     \
                          RankOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32 and bool.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);

REGISTER_KERNEL_BUILDER(Name("Rank")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<bool>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        RankOp);
#endif

template <typename OutType>
class SizeOp : public OpKernel {
 public:
  explicit SizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& inp = ctx->input(0);
    const int64 size = inp.NumElements();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
    if (out->dtype() == DT_INT32) {
      OP_REQUIRES(
          ctx, FastBoundsCheck(size, std::numeric_limits<int32>::max()),
          errors::InvalidArgument("Number of elements was larger than "
                                  "representable by 32-bit output type"));
    }
    out->scalar<OutType>()() = static_cast<OutType>(size);
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int32>("out_type"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_CPU)
                            .HostMemory("output")
                            .TypeConstraint<int64>("out_type"),
                        SizeOp<int64>);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int32>);                        \
  REGISTER_KERNEL_BUILDER(Name("Size")                           \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_type") \
                              .HostMemory("output"),             \
                          SizeOp<int64>);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Size")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int64>("out_type")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SizeOp<int64>);
#endif

class ExpandDimsOp : public OpKernel {
 public:
  explicit ExpandDimsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    int32 dim = ctx->input(1).flat<int32>()(0);
    OP_REQUIRES(
        ctx, (dim >= -1 - ctx->input(0).dims() && dim <= ctx->input(0).dims()),
        errors::InvalidArgument("Tried to expand dim index ", dim,
                                " for tensor with ", ctx->input(0).dims(),
                                " dimensions."));

    auto existing_dims = ctx->input(0).shape().dim_sizes();
    // Safe - # elements in tensor dims bounded.
    const int existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64> new_shape(existing_dims_size);
    for (size_t i = 0; i < new_shape.size(); ++i) {
      new_shape[i] = existing_dims[i];
    }

    // We emulate numpy's interpretation of the dim axis when
    // -input.dims() >= dim <= input.dims().
    if (dim < 0) {
      dim += existing_dims.size() + 1;
    }

    // Clamp to the end if needed.
    dim = std::min<int32>(dim, existing_dims_size);
    new_shape.emplace(new_shape.begin() + dim, 1);
    const TensorShape output_shape(new_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and output
      // should always be the same (we only expand the dimension with 1).
      ctx->SetStatus(
          errors::Internal("Could not expand dimension with input shape ",
                           ctx->input(0).shape().DebugString(),
                           " and output shape ", output_shape.DebugString()));
    }
  }

  bool IsExpensive() override { return false; }
};
REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_CPU)
                            .HostMemory("dim")
                            .TypeConstraint<int32>("Tdim"),
                        ExpandDimsOp);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("ExpandDims")                 \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<type>("T")     \
                              .TypeConstraint<int32>("Tdim") \
                              .HostMemory("dim"),            \
                          ExpandDimsOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

REGISTER_KERNEL_BUILDER(Name("ExpandDims")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .TypeConstraint<int32>("Tdim")
                            .HostMemory("input")
                            .HostMemory("dim")
                            .HostMemory("output"),
                        ExpandDimsOp);
#endif

class SqueezeOp : public OpKernel {
 public:
  explicit SqueezeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    std::vector<int32> squeeze_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("squeeze_dims", &squeeze_dims));
    squeeze_dims_.insert(squeeze_dims.begin(), squeeze_dims.end());
  }

  void Compute(OpKernelContext* ctx) override {
    auto existing_dims = ctx->input(0).shape().dim_sizes();
    const int existing_dims_size = static_cast<int>(existing_dims.size());
    std::vector<int64> new_shape;

    std::unordered_set<int32> wrapped_squeeze_dims;
    wrapped_squeeze_dims.reserve(squeeze_dims_.size());
    // Validate squeeze dims against the input.
    for (int32 dim : squeeze_dims_) {
      OP_REQUIRES(
          ctx, (dim >= -ctx->input(0).dims() && dim < ctx->input(0).dims()),
          errors::InvalidArgument("Tried to squeeze dim index ", dim,
                                  " for tensor with ", ctx->input(0).dims(),
                                  " dimensions."));
      // If dim is < 0, we wrap around (-1 means the last element).
      if (dim < 0) {
        dim = existing_dims_size + dim;
      }

      wrapped_squeeze_dims.insert(dim);
    }

    for (int i = 0; i < existing_dims_size; ++i) {
      auto existing_dim = existing_dims[i];

      // If squeeze_set is non-empty, only squeeze those dimensions.
      if (!wrapped_squeeze_dims.empty()) {
        if (wrapped_squeeze_dims.count(i) > 0) {
          OP_REQUIRES(ctx, existing_dim == 1,
                      errors::InvalidArgument("Tried to explicitly squeeze "
                                              "dimension ",
                                              i, " but dimension was not 1: ",
                                              existing_dim));
        } else {
          // This dimension is not being squeezed.
          new_shape.push_back(existing_dim);
        }
      } else {
        // Copy over all non-1-length dimensions.
        if (existing_dim != 1) {
          new_shape.push_back(existing_dim);
        }
      }
    }

    const TensorShape output_shape(new_shape);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {0}, &output));
    if (!output->CopyFrom(ctx->input(0), output_shape)) {
      // This should never happen, since the sizes of the input and
      // output should always be the same.
      ctx->SetStatus(errors::Internal("Could not squeeze input with shape ",
                                      ctx->input(0).shape().DebugString(),
                                      " and output shape ",
                                      output_shape.DebugString()));
    }
  }

  bool IsExpensive() override { return false; }

 private:
  std::unordered_set<int32> squeeze_dims_;
};

REGISTER_KERNEL_BUILDER(Name("Squeeze").Device(DEVICE_CPU), SqueezeOp);

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(type)                                   \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Squeeze").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SqueezeOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Squeeze")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("output"),
                        SqueezeOp);
#endif

}  // namespace tensorflow
