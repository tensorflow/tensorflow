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

REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResourceReset")
                            .Device(DEVICE_CPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceResetOp);
REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResourceReset")
                            .Device(DEVICE_GPU)
                            .HostMemory("function_buffer_resource"),
                        FunctionBufferingResourceResetOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("FunctionBufferingResourceReset")
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

REGISTER_KERNEL_BUILDER(Name("IteratorGetDevice").Device(DEVICE_CPU),
                        IteratorGetDeviceOp);

Status VerifyTypesMatch(const DataTypeVector& expected,
                        const DataTypeVector& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " types but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != received[i]) {
      return errors::InvalidArgument("Data type mismatch at component ", i,
                                     ": expected ", DataTypeString(expected[i]),
                                     " but got ", DataTypeString(received[i]),
                                     ".");
    }
  }
  return Status::OK();
}

Status VerifyShapesCompatible(const std::vector<PartialTensorShape>& expected,
                              const std::vector<PartialTensorShape>& received) {
  if (expected.size() != received.size()) {
    return errors::InvalidArgument(
        "Number of components does not match: expected ", expected.size(),
        " shapes but got ", received.size(), ".");
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (!expected[i].IsCompatibleWith(received[i])) {
      return errors::InvalidArgument("Incompatible shapes at component ", i,
                                     ": expected ", expected[i].DebugString(),
                                     " but got ", received[i].DebugString(),
                                     ".");
    }
  }

  return Status::OK();
}

string SanitizeThreadSuffix(string suffix) {
  string clean;
  for (int i = 0; i < suffix.size(); ++i) {
    const char ch = suffix[i];
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
      clean += ch;
    } else {
      clean += '_';
    }
  }
  return clean;
}

struct HostBufferElement {
  Status status;
  bool end_of_sequence;
  std::vector<Tensor> value;
};

using MultiDeviceIteratorCallback =
    std::function<void(const HostBufferElement&)>;

class MultiDeviceIterator : public ResourceBase {
 public:
  MultiDeviceIterator(const DataTypeVector& output_types,
                      const std::vector<PartialTensorShape>& output_shapes,
                      const std::vector<string>& devices,
                      std::unique_ptr<FunctionLibraryDefinition> flib_def,
                      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                      FunctionLibraryRuntime* lib)
      : output_types_(output_types),
        output_shapes_(output_shapes),
        devices_(devices),
        flib_def_(std::move(flib_def)),
        pflr_(std::move(pflr)),
        lib_(lib) {
    CHECK_NOTNULL(lib_);
  }

  string DebugString() override {
    return strings::StrCat("MultiDeviceIterator for ", devices_.size(),
                           " devices");
  }

  Status Init(std::unique_ptr<IteratorBase> iterator, int64 max_buffer_size,
              int64* incarnation_id) {
    if (iterator) {
      TF_RETURN_IF_ERROR(
          VerifyTypesMatch(output_types_, iterator->output_dtypes()));
      TF_RETURN_IF_ERROR(
          VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
    }

    mutex_lock l(mu_);
    if (multi_device_buffer_) {
      multi_device_buffer_->Reset();
    }

    ++incarnation_id_;
    *incarnation_id = incarnation_id_;

    multi_device_buffer_.reset(
        new MultiDeviceBuffer(devices_.size(), max_buffer_size, incarnation_id_,
                              std::move(iterator)));
    return Status::OK();
  }

  void GetNextFromShard(IteratorContext* ctx, int shard_num,
                        int64 incarnation_id,
                        MultiDeviceIteratorCallback callback) {
    if (lib_ != nullptr) {
      ctx->set_lib(lib_);
    }
    tf_shared_lock l(mu_);
    multi_device_buffer_->GetNextFromShard(ctx, shard_num, incarnation_id,
                                           std::move(callback));
  }

  const DataTypeVector& output_types() const { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
    return output_shapes_;
  }

  std::shared_ptr<const FunctionLibraryDefinition> function_library() {
    tf_shared_lock l(mu_);
    return lib_def_;
  }

  FunctionLibraryRuntime* const lib() {
    tf_shared_lock l(mu_);
    return lib_;
  }

 private:
  // A private class that uses a background thread to keep a per device buffer
  // full.
  class MultiDeviceBuffer {
   public:
    MultiDeviceBuffer(size_t size, int64 max_buffer_size, int64 incarnation_id,
                      std::unique_ptr<IteratorBase> host_iterator)
        : buffer_(size),
          size_(size),
          max_buffer_size_(max_buffer_size),
          incarnation_id_(incarnation_id),
          host_iterator_(std::move(host_iterator)) {}

    ~MultiDeviceBuffer() { Reset(); }

    void Reset() LOCKS_EXCLUDED(mu_) {
      {
        mutex_lock l(mu_);
        if (background_thread_finished_) {
          return;
        }

        cancelled_ = true;
        // Wake up the background thread.
        for (int i = 0; i < size_; ++i) {
          buffer_[i].cond_var.notify_all();
        }

        // Make sure background thread has finished first.
        while (!background_thread_finished_) {
          shutdown_cond_var_.wait(l);
        }
      }
      RunPendingCallbacks();
    }

    void GetNextFromShard(IteratorContext* ctx, int shard_num,
                          int64 incarnation_id,
                          MultiDeviceIteratorCallback callback) {
      HostBufferElement elem;
      if (incarnation_id_ != incarnation_id) {
        elem.status = errors::InvalidArgument("Invalid incarnation id");
        callback(elem);
        return;
      }

      bool produced_output = false;
      {
        mutex_lock l(mu_);
        if (cancelled_) {
          elem.status = errors::Cancelled("Cancelled Multidevice iterator");
          callback(elem);
          return;
        }

        EnsureBackgroundThreadStarted(ctx);

        if (!buffer_[shard_num].data.empty()) {
          produced_output = true;
          std::swap(elem, buffer_[shard_num].data.front());
          buffer_[shard_num].data.pop_front();
          // Wake up background thread if it is blocked on this element.
          if (buffer_[shard_num].data.size() == max_buffer_size_ - 1) {
            buffer_[shard_num].cond_var.notify_all();
          }
        } else {
          if (background_thread_finished_) {
            produced_output = true;
            elem.end_of_sequence = true;
          } else {
            buffer_[shard_num].callbacks.push_back(std::move(callback));
            callback = nullptr;
          }
        }
      }

      if (produced_output) {
        callback(elem);
      }
    }

   private:
    void EnsureBackgroundThreadStarted(IteratorContext* ctx)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!background_thread_) {
        background_thread_.reset(ctx->env()->StartThread(
            {}, "multi_device_iterator_background_thread",
            std::bind(&MultiDeviceIterator::MultiDeviceBuffer::BackgroundThread,
                      this, new IteratorContext(*ctx))));
      }
    }

    void RunPendingCallbacks() LOCKS_EXCLUDED(mu_) {
      // Run all remaining callbacks.
      std::vector<MultiDeviceIteratorCallback> cancellation_callbacks;
      std::vector<HostBufferElement> cancellation_elements;
      {
        mutex_lock l(mu_);

        for (int i = 0; i < size_; ++i) {
          while (!buffer_[i].callbacks.empty()) {
            if (buffer_[i].data.empty()) {
              HostBufferElement elem;
              elem.status =
                  errors::Cancelled("Cancelled and buffer not filled.");
              cancellation_elements.push_back(std::move(elem));
            } else {
              cancellation_elements.push_back(
                  std::move(buffer_[i].data.front()));
              buffer_[i].data.pop_front();
            }
            cancellation_callbacks.push_back(
                std::move(buffer_[i].callbacks.front()));
            buffer_[i].callbacks.pop_front();
          }
        }
      }
      for (int i = 0; i < cancellation_callbacks.size(); ++i) {
        cancellation_callbacks[i](cancellation_elements[i]);
      }
    }

    void BackgroundThread(IteratorContext* ctx) {
      std::unique_ptr<IteratorContext> cleanup(ctx);
      int shard_to_fetch = 0;
      while (true) {
        HostBufferElement elem;
        MultiDeviceIteratorCallback callback = nullptr;
        bool end_of_iterator = false;

        {
          mutex_lock l(mu_);
          while (!cancelled_ &&
                 buffer_[shard_to_fetch].data.size() >= max_buffer_size_) {
            buffer_[shard_to_fetch].cond_var.wait(l);
          }

          if (cancelled_) {
            background_thread_finished_ = true;
            shutdown_cond_var_.notify_all();
            return;
          }
        }

        elem.status =
            host_iterator_->GetNext(ctx, &elem.value, &elem.end_of_sequence);

        if (elem.status.ok() && elem.end_of_sequence) {
          end_of_iterator = true;
        }

        {
          mutex_lock l(mu_);
          // Try to find a callback, else just push stuff into buffer.
          if (!buffer_[shard_to_fetch].callbacks.empty()) {
            callback = buffer_[shard_to_fetch].callbacks.front();
            buffer_[shard_to_fetch].callbacks.pop_front();
          } else {
            buffer_[shard_to_fetch].data.push_back(std::move(elem));
            elem = HostBufferElement();
          }
        }

        if (callback) {
          (*ctx->runner())(std::bind(std::move(callback), std::move(elem)));
        }

        // Finish off the thread if we reach the end of the iterator. Runs
        // pending callbacks.
        if (end_of_iterator) {
          {
            mutex_lock l(mu_);
            background_thread_finished_ = true;
            shutdown_cond_var_.notify_all();
          }
          RunPendingCallbacks();
          return;
        }
        shard_to_fetch = (shard_to_fetch + 1) % size_;
      }
    }

    struct HostBuffer {
      condition_variable cond_var;
      std::deque<HostBufferElement> data;
      std::deque<MultiDeviceIteratorCallback> callbacks;
    };

    mutex mu_;
    std::unique_ptr<Thread> background_thread_ GUARDED_BY(mu_);
    bool background_thread_finished_ GUARDED_BY(mu_) = false;
    bool cancelled_ GUARDED_BY(mu_) = false;
    condition_variable shutdown_cond_var_ GUARDED_BY(mu_);

    std::vector<HostBuffer> buffer_;

    const size_t size_;
    const int64 max_buffer_size_;
    const int64 incarnation_id_;
    const std::unique_ptr<IteratorBase> host_iterator_;
  };

  mutex mu_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::vector<string> devices_;
  const std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  const std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* const lib_ = nullptr;  // not owned.
  std::shared_ptr<const FunctionLibraryDefinition> lib_def_ GUARDED_BY(mu_);

  int64 incarnation_id_ GUARDED_BY(mu_) = 0;
  std::unique_ptr<MultiDeviceBuffer> multi_device_buffer_ GUARDED_BY(mu_);
};

// Just creates a MultiDeviceIterator and returns it.
class MultiDeviceIteratorHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("devices", &devices_));
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel.
  ~MultiDeviceIteratorHandleOp() override {
    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<MultiDeviceIterator>(cinfo_.container(),
                                                        cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override LOCKS_EXCLUDED(mu_) {
    {
      mutex_lock l(mu_);
      if (resource_ == nullptr) {
        FunctionLibraryRuntime* lib;
        std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
        OP_REQUIRES_OK(context, context->function_library()->Clone(
                                    &flib_def, &pflr, &lib));
        ResourceMgr* mgr = context->resource_manager();
        OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

        MultiDeviceIterator* resource;
        OP_REQUIRES_OK(
            context,
            mgr->LookupOrCreate<MultiDeviceIterator>(
                cinfo_.container(), cinfo_.name(), &resource,
                [this, lib, &flib_def, &pflr](MultiDeviceIterator** ret)
                    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                      *ret = new MultiDeviceIterator(
                          output_types_, output_shapes_, devices_,
                          std::move(flib_def), std::move(pflr), lib);
                      return Status::OK();
                    }));

        Status s = VerifyResource(resource);
        if (TF_PREDICT_FALSE(!s.ok())) {
          resource->Unref();
          context->SetStatus(s);
          return;
        }

        resource_ = resource;
      }
    }
    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, cinfo_.container(), cinfo_.name(),
                                MakeTypeIndex<MultiDeviceIterator>()));
  }

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(MultiDeviceIterator* resource) {
    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_types_, resource->output_types()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
    return Status::OK();
  }

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  MultiDeviceIterator* resource_ GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
  string container_;
  std::vector<string> devices_;
};

REGISTER_KERNEL_BUILDER(Name("MultiDeviceIterator").Device(DEVICE_CPU),
                        MultiDeviceIteratorHandleOp);

// Calls init on the MultiDeviceIterator.
class MultiDeviceIteratorInitOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorInitOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* tensor_max_buffer_size;
    OP_REQUIRES_OK(ctx, ctx->input("max_buffer_size", &tensor_max_buffer_size));
    int64 max_buffer_size = tensor_max_buffer_size->scalar<int64>()();

    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    MultiDeviceIterator* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &resource));
    core::ScopedUnref unref(resource);

    std::unique_ptr<IteratorBase> iterator;
    IteratorContext iter_ctx(ctx);
    iter_ctx.set_lib(resource->lib());
    OP_REQUIRES_OK(
        ctx, dataset->MakeIterator(std::move(iter_ctx), "Iterator", &iterator));
    int64 incarnation_id;
    OP_REQUIRES_OK(ctx, resource->Init(std::move(iterator), max_buffer_size,
                                       &incarnation_id));
    Tensor tensor_incarnation_id(DT_INT64, TensorShape({}));
    tensor_incarnation_id.scalar<int64>()() = incarnation_id;
    OP_REQUIRES_OK(ctx,
                   ctx->set_output("incarnation_id", tensor_incarnation_id));
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiDeviceIteratorInit").Device(DEVICE_CPU),
                        MultiDeviceIteratorInitOp);

// Calls GetNextFromShard(shard) and returns a vector of Tensors as output.
// TODO(rohanj): Implement using BackgroundWorker that Derek built?
class MultiDeviceIteratorGetNextFromShardOp : public AsyncOpKernel {
 public:
  explicit MultiDeviceIteratorGetNextFromShardOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        thread_pool_(new thread::ThreadPool(
            ctx->env(), ThreadOptions(),
            strings::StrCat("multi_device_iterator_get_next_thread_",
                            SanitizeThreadSuffix(name())),
            1 /* num_threads */, false /* low_latency_hint */)) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor* tensor_shard_num;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("shard_num", &tensor_shard_num), done);
    int32 shard_num = tensor_shard_num->scalar<int32>()();

    const Tensor* tensor_incarnation_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->input("incarnation_id", &tensor_incarnation_id), done);
    int64 incarnation_id = tensor_incarnation_id->scalar<int64>()();

    MultiDeviceIterator* iterator;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator), done);
    thread_pool_->Schedule(std::bind(
        [ctx, iterator, shard_num, incarnation_id](DoneCallback done) {
          IteratorContext::Params params;
          params.env = ctx->env();
          params.runner = *(ctx->runner());
          params.function_library = iterator->function_library();
          DeviceBase* device = ctx->function_library()->device();
          params.allocator_getter = [device](AllocatorAttributes attrs) {
            return device->GetAllocator(attrs);
          };
          IteratorContext iter_ctx(std::move(params));

          MultiDeviceIteratorCallback callback = std::bind(
              [ctx](const HostBufferElement& elem, DoneCallback done) {
                // iterator->Unref();
                Status s = elem.status;
                if (!s.ok()) {
                  ctx->SetStatus(s);
                } else if (elem.end_of_sequence) {
                  ctx->SetStatus(errors::OutOfRange("End of sequence"));
                } else {
                  for (int i = 0; i < elem.value.size(); ++i) {
                    ctx->set_output(i, elem.value[i]);
                  }
                }
                done();
              },
              std::placeholders::_1, std::move(done));

          iterator->GetNextFromShard(&iter_ctx, shard_num, incarnation_id,
                                     callback);
          iterator->Unref();
        },
        std::move(done)));
  }

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorGetNextFromShard").Device(DEVICE_CPU),
    MultiDeviceIteratorGetNextFromShardOp);

class MultiDeviceIteratorToStringHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    // Validate that the handle corresponds to a real resource, and
    // that it is an MultiDeviceIterator.
    MultiDeviceIterator* resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
    resource->Unref();

    Tensor* string_handle_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &string_handle_t));
    string_handle_t->scalar<string>()() =
        resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorToStringHandle").Device(DEVICE_CPU),
    MultiDeviceIteratorToStringHandleOp);

class MultiDeviceIteratorFromStringHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorFromStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    OP_REQUIRES(
        ctx,
        output_types_.empty() || output_shapes_.empty() ||
            output_types_.size() == output_shapes_.size(),
        errors::InvalidArgument("If both 'output_types' and 'output_shapes' "
                                "are set, they must have the same length."));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& string_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
                errors::InvalidArgument("string_handle must be a scalar"));

    ResourceHandle resource_handle;
    OP_REQUIRES(
        ctx,
        resource_handle.ParseFromString(string_handle_t.scalar<string>()()),
        errors::InvalidArgument(
            "Could not parse string_handle as a valid ResourceHandle"));

    OP_REQUIRES(
        ctx, resource_handle.device() == ctx->device()->attributes().name(),
        errors::InvalidArgument("Attempted create an iterator on device \"",
                                ctx->device()->attributes().name(),
                                "\" from handle defined on device \"",
                                resource_handle.device(), "\""));

    // Validate that the handle corresponds to a real resource, and
    // that it is an MultiDeviceIterator.
    MultiDeviceIterator* resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, resource_handle, &resource));
    core::ScopedUnref unref_iterator(resource);
    if (!output_types_.empty()) {
      OP_REQUIRES_OK(ctx,
                     VerifyTypesMatch(output_types_, resource->output_types()));
    }
    if (!output_shapes_.empty()) {
      OP_REQUIRES_OK(ctx, VerifyShapesCompatible(output_shapes_,
                                                 resource->output_shapes()));
    }

    Tensor* resource_handle_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
    resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorFromStringHandle").Device(DEVICE_CPU),
    MultiDeviceIteratorFromStringHandleOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
