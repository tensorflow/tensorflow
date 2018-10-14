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
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace data {
namespace {

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
                            const DataTypeVector& output_types)
      : lib_(lib),
        pflr_(std::move(pflr)),
        func_(func),
        buffer_size_(buffer_size),
        source_device_(source_device),
        target_device_(target_device),
        func_args_(func_args),
        output_types_(output_types),
        handle_(kInvalidHandle),
        is_buffering_(false),
        end_of_sequence_(false),
        cancelled_(false) {}

  ~FunctionBufferingResource() override {
    Cancel();
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
    while (is_buffering_) {
      cond_var_.wait(l);
    }
  }

  // Cancels all pending operations and then clears out the state.
  void Reset() LOCKS_EXCLUDED(mu_) {
    Cancel();
    mutex_lock l(mu_);
    buffer_.clear();
    requests_.clear();
    is_buffering_ = false;
    end_of_sequence_ = false;
    cancelled_ = false;
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
      cond_var_.notify_all();
      return;
    }
    FunctionLibraryRuntime::Options opts;
    // Copied from CapturedFunction::generate_step_id();
    opts.step_id = -std::abs(static_cast<int64>(random::New64()));
    opts.source_device = source_device_;
    AllocatorAttributes arg_alloc_attr;
    arg_alloc_attr.set_on_host(true);
    opts.args_alloc_attrs.push_back(arg_alloc_attr);
    for (const auto& dtype : output_types_) {
      AllocatorAttributes ret_alloc_attrs;
      if (DataTypeAlwaysOnHost(dtype)) {
        ret_alloc_attrs.set_on_host(true);
      }
      opts.rets_alloc_attrs.push_back(ret_alloc_attrs);
    }
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
                  if (status.ok()) {
                    buffer_element.value.swap(*rets);
                  } else {
                    end_of_sequence_ = true;
                    is_buffering_ = false;
                  }
                  buffer_.push_back(std::move(buffer_element));
                  if (!requests_.empty()) {
                    buffer_front = std::move(buffer_.front());
                    buffer_.pop_front();
                    callback = std::move(requests_.front());
                    requests_.pop_front();
                  }
                  if (buffer_.size() < buffer_size_ && !end_of_sequence_) {
                    restart_buffering = true;
                  } else {
                    // When the buffer is full, we don't want to call
                    // FillBuffer() unless we're in cancellation phase in which
                    // case FillBuffer() will do the final cleanup post
                    // cancellation.
                    if (cancelled_) {
                      restart_buffering = true;
                    }
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
  const DataTypeVector output_types_;
  FunctionLibraryRuntime::Handle handle_ GUARDED_BY(mu_);
  std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
  std::deque<FunctionBufferCallback> requests_ GUARDED_BY(mu_);
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
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

    const string& source_device = ctx->device()->name();

    // Obtain and canonicalize target_device.
    const Tensor* target_arg;
    OP_REQUIRES_OK(ctx, ctx->input("target_device", &target_arg));
    string target_device;
    OP_REQUIRES_OK(ctx, DeviceNameUtils::CanonicalizeDeviceName(
                            target_arg->scalar<string>()(), source_device,
                            &target_device));

    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES(ctx, lib != nullptr,
                errors::Internal("No function library is provided."));

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
                    source_device, target_device, func_args, output_types_);
                return Status::OK();
              }));
      core::ScopedUnref s(buffer);
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
  DataTypeVector output_types_;
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResource")
                            .Device(DEVICE_CPU)
                            .HostMemory("resource")
                            .HostMemory("string_arg")
                            .HostMemory("target_device"),
                        FunctionBufferResourceHandleOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResource")
                            .Device(DEVICE_GPU)
                            .HostMemory("resource")
                            .HostMemory("string_arg")
                            .HostMemory("target_device"),
                        FunctionBufferResourceHandleOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResource")
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

    if (buffer->Finished()) {
      buffer->Unref();
      ctx->SetStatus(errors::OutOfRange("end_of_sequence"));
      done();
      return;
    }

    FunctionBufferCallback callback =
        [ctx, buffer, done](const BufferElement& buffer_element) {
          Status s = buffer_element.status;
          if (!s.ok()) {
            ctx->SetStatus(s);
            buffer->Unref();
            done();
            return;
          }
          for (size_t i = 0; i < buffer_element.value.size(); ++i) {
            ctx->set_output(i, buffer_element.value[i]);
          }
          buffer->Unref();
          done();
        };
    buffer->MaybeGet(std::move(callback));
  }
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResourceGetNext")
                            .Device(DEVICE_CPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceGetNextOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResourceGetNext")
                            .Device(DEVICE_GPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceGetNextOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResourceGetNext")
                            .Device(DEVICE_SYCL)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceGetNextOp);
#endif  // TENSORFLOW_USE_SYCL

// Resets the FunctionBufferingResource, cancelling all pending requests and
// clearing out the buffer.
class FunctionBufferingResourceResetOp : public OpKernel {
 public:
  explicit FunctionBufferingResourceResetOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  ~FunctionBufferingResourceResetOp() override {}

  void Compute(OpKernelContext* ctx) override {
    ResourceHandle handle;
    OP_REQUIRES_OK(ctx,
                   HandleFromInput(ctx, "function_buffer_resource", &handle));
    FunctionBufferingResource* buffer = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource<FunctionBufferingResource>(ctx, handle, &buffer));
    core::ScopedUnref s(buffer);

    buffer->Reset();
  }
};

REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResourceReset")
                            .Device(DEVICE_CPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceResetOp);
REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResourceReset")
                            .Device(DEVICE_GPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceResetOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("ExperimentalFunctionBufferingResourceReset")
                            .Device(DEVICE_SYCL)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceResetOp);
#endif  // TENSORFLOW_USE_SYCL

class IteratorGetDeviceOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* ctx) override {
    // NOTE(mrry): We do not currently Validate that the handle
    // corresponds to a real IteratorResource, because that symbol is
    // not exposed from the framework library.
    Tensor* device_name_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &device_name_t));
    // NOTE(mrry): Since the operation's input is a resource, we must be
    // colocated with it, and so we can simply return the current device's
    // name without looking at the input.
    device_name_t->scalar<string>()() = ctx->device()->name();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ExperimentalIteratorGetDevice").Device(DEVICE_CPU),
    IteratorGetDeviceOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
