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
  // public types
  typedef std::vector<Tensor> Tuple;

 private:
  // private variables
  int capacity_;
  mutex mu_;
  condition_variable non_empty_cond_var_;
  condition_variable full_cond_var_;
  std::deque<Tuple> buf_ GUARDED_BY(mu_);


 private:
  // private methods

  // If the buffer is configured for bounded capacity, notify
  // waiting inserters that space is now available
  void notify_inserters_if_bounded(mutex_lock & l)
  {
    if(HasBoundedCapacity())
    {
      l.unlock();
      full_cond_var_.notify_one();
    }
  }

  bool HasBoundedCapacity() {
    return capacity_ > 0;
  }

  bool IsFull() {
    return buf_.size() >= capacity_;
  }

 public:
  // public methods
  explicit Buffer(int capacity) : capacity_(capacity) {}

  // the Buffer takes ownership of the Tuple
  void Put(Tuple* tuple) {
    mutex_lock l(mu_);

    // If buffer capacity is bounded wait until elements have been removed
    if(HasBoundedCapacity()) {
      full_cond_var_.wait(l, [this]() {
        return !this->IsFull();
      });
    }

    buf_.push_back(std::move(*tuple));

    l.unlock();
    // maybe possible to optimize by reducing
    // how often this signal is sent
    non_empty_cond_var_.notify_one();
  }

  // Get tuple at front of the buffer
  void Get(Tuple* tuple) {  // TODO(zhifengc): Support cancellation.
    mutex_lock l(mu_);

    // Wait for data if the buffer is empty
    non_empty_cond_var_.wait(l, [this]() {
      return !buf_.empty();
    });

    // Move data into the output tuple
    *tuple = std::move(buf_.front());
    buf_.pop_front();

    notify_inserters_if_bounded(l);
  }

  // Return tuple at index
  Status Peek(int index, Tuple* tuple) {
    mutex_lock l(mu_);

    // Wait if the requested index is not available
    non_empty_cond_var_.wait(l, [&, this]() {
      return index < this->buf_.size();
    });

    // Place tensors in the output tuple
    for(const auto & tensor: buf_[index]) {
      tuple->push_back(tensor);
    }

    return Status::OK();
  }

  // Buffer size
  size_t Size() {
    mutex_lock l(mu_);
    return buf_.size();
  }

  void Clear() {
    mutex_lock l(mu_);
    buf_.clear();
  }

  string DebugString() {
    mutex_lock l(mu_);
    return strings::StrCat("Staging size: ", buf_.size());
  }

};

Status GetBuffer(OpKernelContext* ctx, const NodeDef& ndef, Buffer** buf) {
  auto rm = ctx->resource_manager();
  ContainerInfo cinfo;

  // Lambda for creating the Staging Area
  auto create_fn = [&ndef](Buffer** ret) -> Status
  {
    int capacity;
    TF_RETURN_IF_ERROR(GetNodeAttr(ndef, "capacity", &capacity));
    *ret = new Buffer(capacity);
    return Status::OK();
  };


  TF_RETURN_IF_ERROR(cinfo.Init(rm, ndef, true /* use name() */));
  TF_RETURN_IF_ERROR(rm->LookupOrCreate<Buffer>(cinfo.container(), cinfo.name(),
                                                buf, create_fn));
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

    OP_REQUIRES(ctx, tuple.size() == (size_t)ctx->num_outputs(),
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

class StagePeekOp : public OpKernel {
 public:
  explicit StagePeekOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);
    Buffer::Tuple tuple;

    int index = ctx->input(0).scalar<int>()();

    OP_REQUIRES_OK(ctx, buf->Peek(index, &tuple));

    OP_REQUIRES(ctx, tuple.size() == (size_t)ctx->num_outputs(),
        errors::InvalidArgument("Mismatch stage/unstage: ", tuple.size(),
                                " vs. ", ctx->num_outputs()));

    for (size_t i = 0; i < tuple.size(); ++i) {
      ctx->set_output(i, tuple[i]);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StagePeek").Device(DEVICE_CPU),
                                              StagePeekOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("StagePeek").HostMemory("index").
                            Device(DEVICE_GPU), StagePeekOp);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("StagePeek").HostMemory("index")
                          .Device(DEVICE_SYCL), StagePeekOp);
#endif // TENSORFLOW_USE_SYCL


class StageSizeOp : public OpKernel {
 public:
  explicit StageSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);

    // Allocate size output tensor
    Tensor * size = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}),
                                                     &size));

    // Set it to the actual size
    size->scalar<int32>().setConstant(buf->Size());
  }
};

REGISTER_KERNEL_BUILDER(Name("StageSize").Device(DEVICE_CPU), StageSizeOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("StageSize").HostMemory("size")
                        .Device(DEVICE_GPU), StageSizeOp);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("StageSize").HostMemory("size")
                        .Device(DEVICE_SYCL), StageSizeOp);
#endif // TENSORFLOW_USE_SYCL

class StageClearOp : public OpKernel {
 public:
  explicit StageClearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  // Using this op in such a way that it blocks forever
  // is an error.  As such cancellation is not handled.
  void Compute(OpKernelContext* ctx) override {
    Buffer* buf = nullptr;
    OP_REQUIRES_OK(ctx, GetBuffer(ctx, def(), &buf));
    core::ScopedUnref scope(buf);

    buf->Clear();
  }
};

REGISTER_KERNEL_BUILDER(Name("StageClear").Device(DEVICE_CPU), StageClearOp);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("StageClear").Device(DEVICE_GPU), StageClearOp);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("StageClear").Device(DEVICE_SYCL), StageClearOp);
#endif // TENSORFLOW_USE_SYCL


}  // namespace tensorflow
