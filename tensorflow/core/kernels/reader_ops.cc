/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/io_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

class ReaderVerbSyncOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    ReaderInterface* reader;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "reader_handle", &reader));
    ComputeWithReader(context, reader);
    reader->Unref();
  }

 protected:
  virtual void ComputeWithReader(OpKernelContext* context,
                                 ReaderInterface* reader) = 0;
};

class ReaderVerbAsyncOpKernel : public AsyncOpKernel {
 public:
  using AsyncOpKernel::AsyncOpKernel;

  explicit ReaderVerbAsyncOpKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context),
        thread_pool_(new thread::ThreadPool(
            context->env(), strings::StrCat("reader_thread_",
                                            SanitizeThreadSuffix(def().name())),
            1 /* num_threads */)) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    ReaderInterface* reader;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "reader_handle", &reader));
    thread_pool_->Schedule([this, context, reader, done]() {
      ComputeWithReader(context, reader);
      reader->Unref();
      done();
    });
  }

 protected:
  virtual void ComputeWithReader(OpKernelContext* context,
                                 ReaderInterface* reader) = 0;

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class ReaderReadOp : public ReaderVerbAsyncOpKernel {
 public:
  using ReaderVerbAsyncOpKernel::ReaderVerbAsyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
    QueueInterface* queue;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "queue_handle", &queue));
    core::ScopedUnref unref_me(queue);
    Tensor* key = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("key", TensorShape({}), &key));
    Tensor* value = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("value", TensorShape({}), &value));

    auto key_scalar = key->scalar<string>();
    auto value_scalar = value->scalar<string>();
    reader->Read(queue, &key_scalar(), &value_scalar(), context);
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderRead").Device(DEVICE_CPU), ReaderReadOp);

class ReaderNumRecordsProducedOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("records_produced",
                                                     TensorShape({}), &output));
    output->scalar<int64>()() = reader->NumRecordsProduced();
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderNumRecordsProduced").Device(DEVICE_CPU),
                        ReaderNumRecordsProducedOp);

class ReaderNumWorkUnitsCompletedOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("units_completed",
                                                     TensorShape({}), &output));
    output->scalar<int64>()() = reader->NumWorkUnitsCompleted();
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderNumWorkUnitsCompleted").Device(DEVICE_CPU),
                        ReaderNumWorkUnitsCompletedOp);

class ReaderSerializeStateOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("state", TensorShape({}), &output));
    OP_REQUIRES_OK(context,
                   reader->SerializeState(&output->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderSerializeState").Device(DEVICE_CPU),
                        ReaderSerializeStateOp);

class ReaderRestoreStateOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
    const Tensor* tensor;
    OP_REQUIRES_OK(context, context->input("state", &tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(tensor->shape()),
        errors::InvalidArgument("Reader state must be scalar, but had shape: ",
                                tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, reader->RestoreState(tensor->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderRestoreState").Device(DEVICE_CPU),
                        ReaderRestoreStateOp);

class ReaderResetOp : public ReaderVerbSyncOpKernel {
 public:
  using ReaderVerbSyncOpKernel::ReaderVerbSyncOpKernel;

  void ComputeWithReader(OpKernelContext* context,
                         ReaderInterface* reader) override {
    OP_REQUIRES_OK(context, reader->Reset());
  }
};

REGISTER_KERNEL_BUILDER(Name("ReaderReset").Device(DEVICE_CPU), ReaderResetOp);

}  // namespace tensorflow
