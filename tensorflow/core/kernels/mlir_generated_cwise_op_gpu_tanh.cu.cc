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

#include <memory>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cubin_headers/tanh_f32_kernel.h"
#include "tensorflow/core/kernels/cubin_headers/tanh_f64_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
namespace {
Status CreateKernel(absl::string_view kernel_name, uint64_t num_args,
                    absl::string_view ptx, absl::Span<const uint8_t> cubin_data,
                    se::StreamExecutor* stream_exec,
                    std::unique_ptr<se::KernelBase>& kernel_base) {
  se::MultiKernelLoaderSpec loader_spec(num_args);

  if (!cubin_data.empty()) {
    loader_spec.AddCudaCubinInMemory(
        reinterpret_cast<const char*>(cubin_data.data()), kernel_name);
  }

  kernel_base.reset(new se::KernelBase(stream_exec));
  return stream_exec->GetKernel(loader_spec, kernel_base.get());
}

class MlirGenerateTanhOp : public OpKernel {
 public:
  explicit MlirGenerateTanhOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto* stream = ctx->op_device_context()->stream();
    se::KernelBase* kernel;
    {
      std::lock_guard<std::mutex> l(mu_);
      if (!kernel_) {
        OP_REQUIRES_OK(ctx, CreateKernel("tanh_kernel", 10, "", cubin_data_,
                                         stream->parent(), kernel_));
      }
      kernel = kernel_.get();
    }

    const Tensor& inp = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->forward_input_or_allocate_output({0}, 0, inp.shape(), &out));

    if (inp.NumElements() == 0) {
      return;
    }

    se::KernelArgsArray<10> args;

    args.add_device_memory_argument(
        stream_executor::DeviceMemoryBase(inp.data(), inp.TotalBytes()));
    args.add_device_memory_argument(
        stream_executor::DeviceMemoryBase(inp.data(), inp.TotalBytes()));
    args.add_argument<int64_t>(0);
    args.add_argument<int64_t>(inp.NumElements());
    args.add_argument<int64_t>(1);

    args.add_device_memory_argument(
        stream_executor::DeviceMemoryBase(out->data(), out->TotalBytes()));
    args.add_device_memory_argument(
        stream_executor::DeviceMemoryBase(out->data(), out->TotalBytes()));
    args.add_argument<int64_t>(0);
    args.add_argument<int64_t>(inp.NumElements());
    args.add_argument<int64_t>(1);

    // TODO(b/158649746): Choose block size and thread dim according to the
    // number of input elements. For now, this supports at most 1024 elements.
    OP_REQUIRES_OK(
        ctx, stream->parent()->Launch(stream, se::ThreadDim(inp.NumElements()),
                                      se::BlockDim(1), *kernel, args));
  }

 protected:
  absl::Span<const uint8_t> cubin_data_;

 private:
  std::unique_ptr<se::KernelBase> kernel_;
  std::mutex mu_;
};

class MlirGenerateTanhF32Op : public MlirGenerateTanhOp {
 public:
  explicit MlirGenerateTanhF32Op(OpKernelConstruction* ctx)
      : MlirGenerateTanhOp(ctx) {
    cubin_data_ = kTanhF32Kernel;
  }
};

class MlirGenerateTanhF64Op : public MlirGenerateTanhOp {
 public:
  explicit MlirGenerateTanhF64Op(OpKernelConstruction* ctx)
      : MlirGenerateTanhOp(ctx) {
    cubin_data_ = kTanhF64Kernel;
  }
};
}  // namespace

REGISTER_KERNEL_BUILDER(
    Name("Tanh").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    MlirGenerateTanhF32Op);
REGISTER_KERNEL_BUILDER(
    Name("Tanh").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    MlirGenerateTanhF64Op);
}  // namespace tensorflow
