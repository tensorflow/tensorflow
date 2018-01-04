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

#define EIGEN_USE_THREADS

#include <deque>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/captured_function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class CriticalSection : public ResourceBase {
 public:
  explicit CriticalSection() : is_locked_(false) {}
  ~CriticalSection() override {
    // Wait for all closures to finish running.
    mutex_lock lock(mu_);
    while (!closures_.empty()) {
      queue_empty_cv_.wait(lock);
    }
  }

 private:
  friend class ExecuteInCriticalSectionOp;

  void Acquire(std::function<void()> closure) {
    std::function<void()> next;
    {
      mutex_lock ml(mu_);
      if (is_locked_) {
        closures_.push_back(std::move(closure));
      } else {
        // This branch is the common case.  Avoid the queue.
        is_locked_ = true;
        next = std::move(closure);
      }
    }
    if (next) {
      next();
    }
  }

  void Release() {
    std::function<void()> next;
    {
      mutex_lock ml(mu_);
      CHECK(is_locked_);
      if (!closures_.empty()) {
        // if queue is not empty, start the next entry off the queue.
        std::swap(next, closures_.front());
        closures_.pop_front();
      } else {
        is_locked_ = false;
        queue_empty_cv_.notify_all();
      }
    }
    if (next) {
      next();
    }
  }

  string DebugString() override {
    tf_shared_lock ml(mu_);
    return strings::StrCat("CriticalSection(locked: ", is_locked_,
                           " queue_size: ", closures_.size(), ")");
  }

 private:
  mutex mu_;
  std::deque<std::function<void()>> closures_ GUARDED_BY(mu_);
  bool is_locked_ GUARDED_BY(mu_);
  condition_variable queue_empty_cv_ GUARDED_BY(mu_);
};

class ExecuteInCriticalSectionOp : public AsyncOpKernel {
 public:
  explicit ExecuteInCriticalSectionOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("f", &func_));
  }

 public:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CriticalSection* critical_section = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         LookupOrCreateResource<CriticalSection>(
                             c, HandleFromInput(c, 0), &critical_section,
                             [this, c](CriticalSection** ptr) {
                               *ptr = new CriticalSection;
                               return Status::OK();
                             }),
                         done);
    // No need to Unref critical_section; the Closure below will take
    // care of the Unref associated with this execution.

    auto* execution = new Closure{std::move(done), c, critical_section, &func_};
    execution->Start();
  }

 private:
  class Closure {
   public:
    AsyncOpKernel::DoneCallback done_;
    OpKernelContext* ctx_;
    CriticalSection* cs_;
    FunctionLibraryRuntime::Handle handle_;
    FunctionLibraryRuntime::Options opts_;
    std::vector<Tensor> arguments_t_;
    std::vector<Tensor> output_t_;
    NameAttrList* func_;

    explicit Closure(AsyncOpKernel::DoneCallback done, OpKernelContext* ctx,
                     CriticalSection* critical_section, NameAttrList* func)
        : done_(std::move(done)),
          ctx_(ctx),
          cs_(critical_section),
          handle_(-1),
          func_(func) {}

    ~Closure();

    void Start() {
      // Perform ExecuteFunction isnide a separate thread to avoid
      // having lightweight Functions be inlined in this thread.
      // That inlining would in turn inline DoneAndDelete inside the
      // same thread.  Since DoneAndDelete can call the next
      // ExecuteFunction in the CriticalSection, this can cause a
      // stack overflow.
      cs_->Acquire(
          [this]() { (*ctx_->runner())([this]() { ExecuteFunction(); }); });
    }

   private:
    void ExecuteFunction();
    void DoneAndDelete(const Status& status);
  };

  NameAttrList func_;
};

void ExecuteInCriticalSectionOp::Closure::ExecuteFunction() {
  // Arguments to a Function are in the order:
  //   concat(<formal arguments>, <captured arguments>)
  OpInputList arguments;
  Status s = ctx_->input_list("arguments", &arguments);
  if (!s.ok()) {
    DoneAndDelete(s);
    return;
  }

  arguments_t_.reserve(arguments.size());
  for (const Tensor& t : arguments) {
    arguments_t_.push_back(t);
  }

  auto* function_library = ctx_->function_library();
  s = function_library->Instantiate(func_->name(), AttrSlice(&func_->attr()),
                                    &handle_);
  if (!s.ok()) {
    DoneAndDelete(s);
    return;
  }

  opts_.step_id = CapturedFunction::generate_step_id();
  auto* step_container =
      new ScopedStepContainer(opts_.step_id, [this](const string& name) {
        ctx_->resource_manager()->Cleanup(name).IgnoreError();
      });
  opts_.cancellation_manager = ctx_->cancellation_manager();
  opts_.step_container = step_container;
  opts_.runner = ctx_->runner();

  function_library->Run(opts_, handle_, arguments_t_, &output_t_,
                        [this](const Status& s) { DoneAndDelete(s); });
}

void ExecuteInCriticalSectionOp::Closure::DoneAndDelete(const Status& status) {
  cs_->Release();

  if (!status.ok()) {
    ctx_->SetStatus(status);
  } else {
    OpOutputList output;
    const Status s = ctx_->output_list("outputs", &output);
    if (!s.ok()) {
      ctx_->SetStatus(s);
    } else if (output_t_.size() != output.size()) {
      ctx_->SetStatus(errors::Internal(
          "Could not set all outputs.  Expected output size is ", output.size(),
          " but function set ", output_t_.size(), " output values."));
    } else {
      for (int i = 0; i < output_t_.size(); ++i) {
        output.set(i, output_t_[i]);
      }
    }
  }

  delete opts_.step_container;
  opts_.step_container = nullptr;
  done_();
  cs_->Unref();
  delete this;
}

ExecuteInCriticalSectionOp::Closure::~Closure() {
  CHECK(!opts_.step_container)
      << "Initialized closure destroyed without calling Done";
}

REGISTER_KERNEL_BUILDER(Name("ExecuteInCriticalSection").Device(DEVICE_CPU),
                        ExecuteInCriticalSectionOp);

REGISTER_KERNEL_BUILDER(Name("CriticalSectionOp").Device(DEVICE_CPU),
                        ResourceHandleOp<CriticalSection>);

// TODO(ebrevdo): Re-enable once the cross-device function execution works.
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("ExecuteInCriticalSection")
                            .Device(DEVICE_GPU)
                            .HostMemory("critical_section"),
                        ExecuteInCriticalSectionOp);
REGISTER_KERNEL_BUILDER(
    Name("CriticalSectionOp").Device(DEVICE_GPU).HostMemory("resource"),
    ResourceHandleOp<CriticalSection>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
