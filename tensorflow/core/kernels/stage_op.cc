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

#include <deque>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

class Buffer : public ResourceBase {
 public:
  explicit Buffer() {}

  typedef std::vector<Tensor> Tuple;

  // the Buffer takes ownership of the Tuple
  void Put(Tuple* tuple) {
    mutex_lock l(mu_);
    buf_.push_back(std::move(*tuple));
    non_empty_cond_var_.notify_one();  // maybe possible to optimize by reducing
                                       // how often this signal is sent
  }

  void Get(Tuple* tuple) {  // TODO(zhifengc): Support cancellation.
    mutex_lock l(mu_);
    while (buf_.empty()) {
      non_empty_cond_var_.wait(l);
    }

    *tuple = std::move(buf_.front());
    buf_.pop_front();
  }

  string DebugString() {
    mutex_lock l(mu_);
    return strings::StrCat("Staging size: ", buf_.size());
  }

 private:
  mutex mu_;
  condition_variable non_empty_cond_var_;
  std::deque<Tuple> buf_ GUARDED_BY(mu_);
};

Status CreateBuffer(Buffer** ret) {
  *ret = new Buffer;
  return Status::OK();
}

Status GetBuffer(OpKernelContext* ctx, const NodeDef& ndef, Buffer** buf) {
  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;
  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<Buffer>(cinfo.container(), cinfo.name(),
                                                buf, CreateBuffer));
  return Status::OK();
}

}  // namespace

class StageOp : public OpKernel {
 public:
  explicit StageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);
    Buffer::Tuple tuple;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      tuple.push_back(ctx->input(i));
    }
    buf->Put(&tuple);
  }
};

REGISTER_KERNEL_BUILDER(Name("Stage").Device(DEVICE_CPU), StageOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Stage").Device(DEVICE_GPU), StageOp);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Stage").Device(DEVICE_SYCL), StageOp);
#endif // TENSORFLOW_USE_SYCL

class UnstageOp : public OpKernel {
 public:
  explicit UnstageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);
    Buffer::Tuple tuple;
    buf->Get(&tuple);
    OP_REQUIRES(
        ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));
    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Unstage").Device(DEVICE_CPU), UnstageOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Unstage").Device(DEVICE_GPU), UnstageOp);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Unstage").Device(DEVICE_SYCL), UnstageOp);
#endif // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
