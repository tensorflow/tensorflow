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

#include <algorithm>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/kernels/light_outside_compilation.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

// Sample kernels for the light outside compilation test.

namespace tensorflow {
namespace {

// Just copy the input.
REGISTER_OP("TestStaticTf")
    .Input("input: float")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class TestStaticTfOp : public OpKernel {
 public:
  explicit TestStaticTfOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* out_tensor = nullptr;
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", ctx->input(0).shape(),
                                             &out_tensor));

    // Just pass the value through.
    uint64_t size = input.AllocatedBytes();
    se::DeviceMemoryBase gpu_dst{out_tensor->data(), size};
    se::Stream* stream = ctx->op_device_context()->stream();

    OP_REQUIRES_OK(ctx,
                   stream->MemcpyD2D(
                       /*gpu_dst=*/&gpu_dst,
                       /*gpu_src=*/se::DeviceMemoryBase{input.data(), size},
                       /*size=*/input.AllocatedBytes()));
  }
};

REGISTER_KERNEL_BUILDER(Name("TestStaticTf").Device(DEVICE_GPU),
                        TestStaticTfOp);
REGISTER_XLA_OP(Name("TestStaticTf").Device(DEVICE_GPU_XLA_JIT),
                LightOutsideCompilationOp)

REGISTER_OP("TestStaticMultipleOutputTf")
    .Input("input: float")
    .Output("output1: float")
    .Output("output2: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(0));
      return absl::OkStatus();
    });

class TestStaticMultipleOutputTfOp : public OpKernel {
 public:
  explicit TestStaticMultipleOutputTfOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* out_tensor1 = nullptr;
    Tensor* out_tensor2 = nullptr;
    const Tensor& input = ctx->input(0);
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output1", ctx->input(0).shape(),
                                             &out_tensor1));
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output2", ctx->input(0).shape(),
                                             &out_tensor2));

    // Just pass the value through.
    uint64_t size = input.AllocatedBytes();
    se::DeviceMemoryBase gpu_dst1{out_tensor1->data(), size};
    se::DeviceMemoryBase gpu_dst2{out_tensor2->data(), size};
    se::Stream* stream =
        ctx->device()->tensorflow_accelerator_device_info()->stream;

    OP_REQUIRES_OK(ctx,
                   stream->MemcpyD2D(
                       /*gpu_dst=*/&gpu_dst1,
                       /*gpu_src=*/se::DeviceMemoryBase{input.data(), size},
                       /*size=*/input.AllocatedBytes()));
    OP_REQUIRES_OK(ctx,
                   stream->MemcpyD2D(
                       /*gpu_dst=*/&gpu_dst2,
                       /*gpu_src=*/se::DeviceMemoryBase{input.data(), size},
                       /*size=*/input.AllocatedBytes()));
  }
};

REGISTER_KERNEL_BUILDER(Name("TestStaticMultipleOutputTf").Device(DEVICE_GPU),
                        TestStaticMultipleOutputTfOp);
REGISTER_XLA_OP(Name("TestStaticMultipleOutputTf").Device(DEVICE_GPU_XLA_JIT),
                LightOutsideCompilationOp)

// Copy the input up to `max_size`.
REGISTER_OP("TestDynamicTf")
    .Input("input: float")
    .Attr("max_size: int")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(0))));
      return absl::OkStatus();
    });

// Same as TestStaticTfOp, but only copies up to `max_size` attribute.
class TestDynamicTfOp : public OpKernel {
 public:
  explicit TestDynamicTfOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_size", &max_size_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    // Pass through the part of the value specified by the `max_size` attribute.
    int64_t size = input.AllocatedBytes();
    CHECK_LE(max_size_, size);
    uint64_t size_to_cpy = std::min(size, max_size_) / 2;

    TensorShape allocated_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(
                       absl::Span<const int>{static_cast<int>(size_to_cpy)},
                       &allocated_shape));

    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("output", allocated_shape, &out_tensor));

    se::Stream* stream =
        ctx->device()->tensorflow_accelerator_device_info()->stream;

    se::DeviceMemoryBase gpu_dst{out_tensor->data(), size_to_cpy};
    OP_REQUIRES_OK(ctx, stream->MemcpyD2D(
                            /*gpu_dst=*/&gpu_dst,
                            /*gpu_src=*/
                            se::DeviceMemoryBase{input.data(),
                                                 static_cast<uint64_t>(size)},
                            /*size=*/size_to_cpy));
  }

 private:
  int64_t max_size_;
};
REGISTER_KERNEL_BUILDER(Name("TestDynamicTf").Device(DEVICE_GPU),
                        TestDynamicTfOp);

class TestDynamicTfXlaOp : public LightOutsideCompilationOp {
 public:
  explicit TestDynamicTfXlaOp(OpKernelConstruction* context)
      : LightOutsideCompilationOp(context) {}
  absl::StatusOr<OutputDimensionBoundsMap> DynamicOutputDimensions(
      const NodeDef& ndef, XlaOpKernelContext* ctx) const override {
    OutputDimensionBoundsMap out;
    TF_ASSIGN_OR_RETURN(auto max_bound, GetNodeAttr<int64_t>(ndef, "max_size"));
    out[0][0] = max_bound;
    return out;
  }
};

REGISTER_XLA_OP(Name("TestDynamicTf").Device(DEVICE_GPU_XLA_JIT),
                TestDynamicTfXlaOp);

REGISTER_OP("DynamicMultidim")
    .Input("output_shape: int32")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(5));
      return absl::OkStatus();
    });

// Just fill in the data with ones for a given shape.
class DynamicMultidimOp : public OpKernel {
 public:
  explicit DynamicMultidimOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    TensorShape output_shape;
    auto vec = ctx->input(0).flat<int32>();
    for (int i = 0; i < vec.size(); i++) {
      OP_REQUIRES_OK(ctx, output_shape.AddDimWithStatus(vec(i)));
    }
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("output", output_shape, &out_tensor));

    // Fill in the value with ones.
    int32_t num_elements = output_shape.num_elements();
    std::vector<float> host_data(num_elements);
    for (int i = 0; i < output_shape.num_elements(); i++) {
      host_data[i] = 1.0;
    }
    se::DeviceMemoryBase gpu_dst{out_tensor->data(),
                                 static_cast<uint64_t>(num_elements)};

    se::Stream* stream =
        ctx->device()->tensorflow_accelerator_device_info()->stream;
    OP_REQUIRES_OK(ctx, stream->Memcpy(
                            /*gpu_dst=*/&gpu_dst, /*host_src=*/host_data.data(),
                            /*size=*/num_elements * sizeof(float)));
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DynamicMultidim").Device(DEVICE_GPU).HostMemory("output_shape"),
    DynamicMultidimOp);

class DynamicMultidimXlaOp : public LightOutsideCompilationOp {
 public:
  explicit DynamicMultidimXlaOp(OpKernelConstruction* context)
      : LightOutsideCompilationOp(context) {}
  absl::StatusOr<OutputDimensionBoundsMap> DynamicOutputDimensions(
      const NodeDef& ndef, XlaOpKernelContext* ctx) const override {
    OutputDimensionBoundsMap out;
    for (int i = 0; i < 5; i++) {
      out[0][i] = 20;
    }
    return out;
  }
};

REGISTER_XLA_OP(Name("DynamicMultidim")
                    .Device(DEVICE_GPU_XLA_JIT)
                    .CompileTimeConstantInput("output_shape"),
                DynamicMultidimXlaOp);

REGISTER_OP("DynamicUnranked")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return absl::OkStatus();
    });

REGISTER_XLA_OP(Name("DynamicUnranked").Device(DEVICE_GPU_XLA_JIT),
                LightOutsideCompilationOp);

// Copies up to `to_copy_bytes` from the input: tests constant storage.
REGISTER_OP("TestTfMustBeConstant")
    .Input("input: float")
    .Input("constant_to_add: int32")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class TestTfMustBeConstantOp : public OpKernel {
 public:
  explicit TestTfMustBeConstantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(ctx->input(1).shape()),
        errors::InvalidArgument("Expected argument 0 to be a scalar. Received",
                                ctx->input(1).DebugString()));
    int constant_to_add = ctx->input(1).scalar<int>()();
    size_t allocated_size = input.AllocatedBytes();

    se::Stream* stream =
        ctx->device()->tensorflow_accelerator_device_info()->stream;

    Tensor tmp;
    AllocatorAttributes pinned_alloc_attrs;
    pinned_alloc_attrs.set_on_host(true);
    pinned_alloc_attrs.set_gpu_compatible(true);
    TF_CHECK_OK(ctx->allocate_temp(input.dtype(), input.shape(), &tmp,
                                   pinned_alloc_attrs));

    OP_REQUIRES_OK(
        ctx, stream->Memcpy(tmp.data(),
                            se::DeviceMemoryBase{input.data(), allocated_size},
                            allocated_size));

    OP_REQUIRES_OK(ctx, stream->BlockHostUntilDone());

    for (int i = 0; i < input.NumElements(); i++) {
      tmp.flat<float>().data()[i] += constant_to_add;
    }

    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", ctx->input(0).shape(),
                                             &out_tensor));
    se::DeviceMemoryBase gpu_dst{out_tensor->data(),
                                 static_cast<uint64_t>(allocated_size)};
    OP_REQUIRES_OK(ctx, stream->Memcpy(&gpu_dst, tmp.data(), allocated_size));
  }
};

REGISTER_KERNEL_BUILDER(Name("TestTfMustBeConstant").Device(DEVICE_GPU),
                        TestTfMustBeConstantOp);

REGISTER_XLA_OP(Name("TestTfMustBeConstant")
                    .Device(DEVICE_GPU_XLA_JIT)
                    .CompileTimeConstantInput("constant_to_add"),
                LightOutsideCompilationOp)

REGISTER_OP("TestDynamicTfWithBound")
    .Input("input: float")
    .Attr("max_size: int")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class TestDynamicTfWithBoundOp : public OpKernel {
 public:
  explicit TestDynamicTfWithBoundOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_size", &max_size_));
  }
  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    uint64_t size_to_cpy =
        std::min(input.AllocatedBytes(), static_cast<size_t>(max_size_));

    TensorShape allocated_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(
                       absl::Span<const int>{static_cast<int>(size_to_cpy)},
                       &allocated_shape));

    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("output", allocated_shape, &out_tensor));

    se::Stream* stream =
        ctx->device()->tensorflow_accelerator_device_info()->stream;
    se::DeviceMemoryBase gpu_dst{out_tensor->data(), size_to_cpy};
    OP_REQUIRES_OK(
        ctx, stream->MemcpyD2D(
                 /*gpu_dst=*/&gpu_dst,
                 /*gpu_src=*/se::DeviceMemoryBase{input.data(), size_to_cpy},
                 /*size=*/size_to_cpy));
  }

 private:
  int64_t max_size_;
};

REGISTER_KERNEL_BUILDER(Name("TestDynamicTfWithBound").Device(DEVICE_GPU),
                        TestDynamicTfWithBoundOp);

class TestDynamicTfWithBoundXlaOp : public LightOutsideCompilationOp {
 public:
  explicit TestDynamicTfWithBoundXlaOp(OpKernelConstruction* context)
      : LightOutsideCompilationOp(context) {}

  absl::StatusOr<OutputDimensionBoundsMap> DynamicOutputDimensions(
      const NodeDef& ndef, XlaOpKernelContext* ctx) const override {
    OutputDimensionBoundsMap out;
    TF_ASSIGN_OR_RETURN(auto max_bound, GetNodeAttr<int64_t>(ndef, "max_size"));
    out[0][0] = max_bound;
    return out;
  }
};

REGISTER_XLA_OP(Name("TestDynamicTfWithBound").Device(DEVICE_GPU_XLA_JIT),
                TestDynamicTfWithBoundXlaOp);

}  // namespace
}  // namespace tensorflow
