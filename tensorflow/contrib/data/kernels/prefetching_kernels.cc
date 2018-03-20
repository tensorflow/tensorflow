/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

struct BufferElement {
  // The producer sets `status` if getting the input element fails.
  Status status;
  // The buffered data element.
  std::vector<Tensor> value;
};

using FunctionBufferCallback = std::function<void(const BufferElement&)>;

class FunctionBufferingResource : public ResourceBase {
 public:
  FunctionBufferingResource(FunctionLibraryRuntime* lib,
                            std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                            const NameAttrList& func, int64 buffer_size,
                            const string& source_device,
                            const string& target_device,
                            const std::vector<Tensor>& func_args,
                            int64 thread_pool_size)
      : lib_(lib),
        pflr_(std::move(pflr)),
        func_(func),
        buffer_size_(buffer_size),
        source_device_(source_device),
        target_device_(target_device),
        func_args_(func_args),
        handle_(kInvalidHandle),
        is_buffering_(false),
        end_of_sequence_(false),
        cancelled_(false) {
    if (thread_pool_size > 0) {
      thread_pool_ = new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                            "buffer_resource", thread_pool_size,
                                            false /* low_latency_hint */);
      runner_ = [this](std::function<void()> c) {
        thread_pool_->Schedule(std::move(c));
      };
    }
  }

  ~FunctionBufferingResource() override {
    Cancel();
    {
      mutex_lock l(mu_);
      while (is_buffering_) {
        cond_var_.wait(l);
      }
    }
    if (thread_pool_ != nullptr) {
      delete thread_pool_;
    }
  }

  string DebugString() override {
    return strings::StrCat("FunctionBufferingResource. Size: ", buffer_size_,
                           "; target_device: ", target_device_);
  }

  // Instantiates the function the first time it's called. After that it caches
  // the handle.
  Status Instantiate() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    // Re-use existing handle if it's been set, effectively caching it.
    if (handle_ != kInvalidHandle) {
      return Status::OK();
    }
    AttrValueMap attr_values = func_.attr();
    FunctionLibraryRuntime::InstantiateOptions opts;
    opts.target = target_device_;
    return lib_->Instantiate(func_.name(), AttrSlice(&attr_values), opts,
                             &handle_);
  }

  // Returns true if we've got to the end of the sequence and exhausted the
  // buffer.
  bool Finished() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    return end_of_sequence_ && buffer_.empty();
  }

  // Cancels any buffering / prefetching going on.
  void Cancel() LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    cancelled_ = true;
  }

  // If the buffer has anything, runs `callback` on the first element in the
  // buffer, else schedules the `callback` to be called. Requires `args` and
  // `lib` in case more function calls need to be scheduled.
  void MaybeGet(FunctionBufferCallback callback) LOCKS_EXCLUDED(mu_) {
    bool start_buffering = false;
    bool produced_output = false;
    BufferElement buffer_element;
    {
      mutex_lock l(mu_);
      if (!is_buffering_ && !end_of_sequence_) {
        start_buffering = true;
      }
      if (!buffer_.empty()) {
        produced_output = true;
        std::swap(buffer_element, buffer_.front());
        buffer_.pop_front();
      } else {
        produced_output = false;
        requests_.push_back(std::move(callback));
      }
    }
    if (produced_output) {
      callback(buffer_element);
    }
    if (start_buffering) {
      FillBuffer();
    }
  }

 private:
  void FillBuffer() LOCKS_EXCLUDED(mu_) {
    FunctionLibraryRuntime::Handle handle;
    std::vector<FunctionBufferCallback> cancellation_callbacks;
    std::vector<BufferElement> cancellation_buffer_elements;
    bool cancelled = false;
    {
      mutex_lock l(mu_);
      handle = handle_;
      if (cancelled_) {
        cancelled = true;
        // Run through and fulfill all pending requests, if possible.
        while (!requests_.empty()) {
          if (!buffer_.empty()) {
            cancellation_buffer_elements.push_back(std::move(buffer_.front()));
            buffer_.pop_front();
            cancellation_callbacks.push_back(std::move(requests_.front()));
            requests_.pop_front();
          } else {
            LOG(ERROR) << "Buffer ran out of elements and we couldn't satisfy: "
                       << requests_.size() << " requests";
            break;
          }
        }
        is_buffering_ = false;
      } else {
        is_buffering_ = true;
      }
    }
    if (cancelled) {
      for (int i = 0; i < cancellation_callbacks.size(); ++i) {
        cancellation_callbacks[i](cancellation_buffer_elements[i]);
      }
      // We only wait on cond_var_ in the destructor, so there would atmost be
      // one waiter to notify.
      cond_var_.notify_one();
      return;
    }
    FunctionLibraryRuntime::Options opts;
    // Copied from CapturedFunction::generate_step_id();
    opts.step_id = -std::abs(static_cast<int64>(random::New64()));
    if (runner_ != nullptr) {
      opts.runner = &runner_;
    }
    opts.source_device = source_device_;
    AllocatorAttributes arg_alloc_attr;
    arg_alloc_attr.set_on_host(true);
    opts.args_alloc_attrs.push_back(arg_alloc_attr);
    if (opts.source_device != target_device_) {
      opts.remote_execution = true;
    }
    opts.create_rendezvous = true;
    auto* rets = new std::vector<Tensor>;
    lib_->Run(opts, handle, func_args_, rets,
              [this, rets](const Status& status) {
                FunctionBufferCallback callback = nullptr;
                BufferElement buffer_front;
                bool restart_buffering = false;
                {
                  mutex_lock l(mu_);
                  BufferElement buffer_element;
                  buffer_element.status = status;
                  if (!status.ok()) {
                    end_of_sequence_ = true;
                    is_buffering_ = false;
                    buffer_.push_back(std::move(buffer_element));
                    return;
                  }
                  buffer_element.value.swap(*rets);
                  buffer_.push_back(std::move(buffer_element));
                  if (!requests_.empty()) {
                    buffer_front = std::move(buffer_.front());
                    buffer_.pop_front();
                    callback = std::move(requests_.front());
                    requests_.pop_front();
                  }
                  if (buffer_.size() < buffer_size_) {
                    restart_buffering = true;
                  } else {
                    is_buffering_ = false;
                  }
                }
                if (callback != nullptr) {
                  callback(buffer_front);
                }
                if (restart_buffering) {
                  FillBuffer();
                }
              });
  }

  mutex mu_;
  FunctionLibraryRuntime* lib_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  NameAttrList func_;
  const int64 buffer_size_;
  const string source_device_;
  const string target_device_;
  const std::vector<Tensor> func_args_;
  thread::ThreadPool* thread_pool_ = nullptr;
  FunctionLibraryRuntime::Handle handle_ GUARDED_BY(mu_);
  std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
  std::deque<FunctionBufferCallback> requests_ GUARDED_BY(mu_);
  std::function<void(std::function<void()>)> runner_ = nullptr;
  bool is_buffering_ GUARDED_BY(mu_);
  bool end_of_sequence_ GUARDED_BY(mu_);
  bool cancelled_ GUARDED_BY(mu_);
  condition_variable cond_var_;
};

class FunctionBufferResourceHandleOp : public OpKernel {
 public:
  explicit FunctionBufferResourceHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), flib_def_(nullptr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &buffer_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("thread_pool_size", &thread_pool_size_));
  }

  ~FunctionBufferResourceHandleOp() override {
    if (cinfo_.resource_is_private_to_kernel()) {
      if (!cinfo_.resource_manager()
               ->Delete<FunctionBufferingResource>(cinfo_.container(),
                                                   cinfo_.name())
               .ok()) {
        // Do nothing; the resource can have been deleted by session resets.
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* string_arg;
    OP_REQUIRES_OK(ctx, ctx->input("string_arg", &string_arg));
    std::vector<Tensor> func_args;
    func_args.push_back(*string_arg);

    // Obtain and canonicalize target_device.
    const Tensor* target_arg;
    OP_REQUIRES_OK(ctx, ctx->input("target_device", &target_arg));
    const string& target_device =
        DeviceNameUtils::CanonicalizeDeviceName(target_arg->scalar<string>()());

    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES(ctx, lib != nullptr,
                errors::Internal("No function library is provided."));

    const string& source_device = ctx->device()->name();

    mutex_lock l(mu_);
    if (!initialized_) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));
      FunctionLibraryRuntime* clone_lib;
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr;
      OP_REQUIRES_OK(ctx, lib->Clone(&flib_def_, &pflr, &clone_lib));
      // Create the resource.
      FunctionBufferingResource* buffer;
      OP_REQUIRES_OK(
          ctx,
          ctx->resource_manager()->LookupOrCreate<FunctionBufferingResource>(
              cinfo_.container(), cinfo_.name(), &buffer,
              [clone_lib, &pflr, &source_device, &target_device, func_args,
               this](FunctionBufferingResource** ptr) {
                *ptr = new FunctionBufferingResource(
                    clone_lib, std::move(pflr), func_, buffer_size_,
                    source_device, target_device, func_args, thread_pool_size_);
                return Status::OK();
              }));
      OP_REQUIRES_OK(ctx, buffer->Instantiate());
      initialized_ = true;
    }

    OP_REQUIRES_OK(ctx, MakeResourceHandleToOutput(
                            ctx, 0, cinfo_.container(), cinfo_.name(),
                            MakeTypeIndex<FunctionBufferingResource>()));
  }

 private:
  mutex mu_;
  ContainerInfo cinfo_ GUARDED_BY(mu_);
  bool initialized_ GUARDED_BY(mu_) = false;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  NameAttrList func_;
  int64 buffer_size_;
  string container_;
  string name_;
  int64 thread_pool_size_;
};

REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResource")
                            .Device(DEVICE_CPU)
                            .HostMemory("resource")
                            .HostMemory("string_arg")
                            .HostMemory("target_device"),
                        FunctionBufferResourceHandleOp);
REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResource")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("string_arg")
                            .HostMemory("target_device"),
                        FunctionBufferResourceHandleOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResource")
                            .Device(DEVICE_SYCL)
                            .HostMemory("resource")
                            .HostMemory("string_arg")
                            .HostMemory("target_device"),
                        FunctionBufferResourceHandleOp);
#endif  // TENSORFLOW_USE_SYCL

// Prefetches and fills up a buffer by calling a function that provides the
// elements to buffer.
class FunctionBufferingResourceGetNextOp : public AsyncOpKernel {
 public:
  explicit FunctionBufferingResourceGetNextOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {}

  ~FunctionBufferingResourceGetNextOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    ResourceHandle handle;
    OP_REQUIRES_OK_ASYNC(
        ctx, HandleFromInput(ctx, "function_buffer_resource", &handle), done);
    FunctionBufferingResource* buffer = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource<FunctionBufferingResource>(ctx, handle, &buffer),
        done);
    core::ScopedUnref s(buffer);

    if (buffer->Finished()) {
      ctx->SetStatus(errors::OutOfRange("end_of_sequence"));
      done();
      return;
    }

    FunctionBufferCallback callback =
        [ctx, done](const BufferElement& buffer_element) {
          Status s = buffer_element.status;
          if (!s.ok()) {
            ctx->SetStatus(s);
            done();
            return;
          }
          for (size_t i = 0; i < buffer_element.value.size(); ++i) {
            ctx->set_output(i, buffer_element.value[i]);
          }
          done();
        };
    buffer->MaybeGet(std::move(callback));
  }
};

REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResourceGetNext")
                            .Device(DEVICE_CPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceGetNextOp);
REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResourceGetNext")
                            .Device(DEVICE_GPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceGetNextOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResourceGetNext")
                            .Device(DEVICE_SYCL)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceGetNextOp);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
