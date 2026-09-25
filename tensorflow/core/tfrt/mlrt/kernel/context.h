/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_KERNEL_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_MLRT_KERNEL_CONTEXT_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {

// The context for tensorflow::OpKernel.
class Context : public mlrt::UserContext<Context> {
 public:
  explicit Context(
      const tfd::KernelFallbackCompatRequestState* fallback_request_state,
      tfrt::ResourceContext* resource_context,
      const tfrt::CancellationContext* cancellation_context = nullptr)
      : fallback_request_state_(fallback_request_state),
        op_kernel_context_(fallback_request_state_),
        resource_context_(resource_context),
        cancellation_context_(cancellation_context) {
    DCHECK(resource_context_);
  }

  Context(const Context&) = default;
  Context& operator=(const Context&) = default;

  const tfd::KernelFallbackCompatRequestState& fallback_request_state() const {
    return *fallback_request_state_;
  }
  void set_fallback_request_state(
      const tfd::KernelFallbackCompatRequestState* fallback_request_state) {
    DCHECK(fallback_request_state);
    fallback_request_state_ = fallback_request_state;
  }

  OpKernelContext::Params& params() { return op_kernel_context_.params; }
  OpKernelContext& op_kernel_context() {
    return op_kernel_context_.op_kernel_context;
  }

  tfrt::ResourceContext& resource_context() const { return *resource_context_; }

  const tfrt::CancellationContext* cancellation_context() const {
    return cancellation_context_;
  }

  mlrt::ExecutionContext* root_execution_context() const {
    return root_execution_context_;
  }
  void set_root_execution_context(
      mlrt::ExecutionContext* root_execution_context) {
    root_execution_context_ = root_execution_context;
  }

  tfrt_stub::OpKernelRunState& run_state() {
    // Keep states needed by kernel execution in a thread local storage to avoid
    // repeated reallocation and destruction of them.
    thread_local tfrt_stub::OpKernelRunState run_state;
    return run_state;
  }

  // Return true if there is a cancellation request.
  bool IsCancelled() {
    return cancellation_context_ != nullptr &&
           cancellation_context_->IsCancelled();
  }

 private:
  const tfd::KernelFallbackCompatRequestState* fallback_request_state_ =
      nullptr;

  struct CopyableOpKernelContext {
    OpKernelContext::Params params;
    OpKernelContext op_kernel_context;

    explicit CopyableOpKernelContext(
        const tfd::KernelFallbackCompatRequestState* fallback_request_state)
        : params(),
          op_kernel_context(
              [this, fallback_request_state]() {
                DCHECK(fallback_request_state);
                params.step_id = fallback_request_state->step_id();
                auto* device = fallback_request_state->cpu_device();
                params.device = device;
                // Still use original device's resource_manager.
                params.resource_manager = device->resource_manager();
                params.step_container =
                    fallback_request_state->step_container();
                // Following two parameters are used to support executing
                // tf.data via fallback.
                params.function_library =
                    fallback_request_state->cpu_function_library_runtime();
                params.runner = fallback_request_state->runner();
                params.collective_executor =
                    fallback_request_state->collective_executor();
                params.rendezvous = fallback_request_state->rendezvous();
                params.session_metadata =
                    &fallback_request_state->session_metadata();
                params.cancellation_manager =
                    fallback_request_state->cancellation_manager();
                return &params;
              }(),
              0) {}
    CopyableOpKernelContext(const CopyableOpKernelContext& other)
        : params(other.params),
          op_kernel_context(&params, other.op_kernel_context.num_outputs()) {}
    CopyableOpKernelContext& operator=(const CopyableOpKernelContext& other) {
      params = other.params;
      op_kernel_context.ResetOutputs(other.op_kernel_context.num_outputs());
      return *this;
    }
    ~CopyableOpKernelContext() { op_kernel_context.ResetOutputs(); }
  };
  CopyableOpKernelContext op_kernel_context_;

  tfrt::ResourceContext* resource_context_ = nullptr;
  const tfrt::CancellationContext* cancellation_context_;
  mlrt::ExecutionContext* root_execution_context_ = nullptr;
};

// Sets up exit_handler and deferred AsyncOpKernel tracking on
// `execution_context` and its `tf_mlrt::Context`, returning a Chain AsyncValue
// that becomes concrete only after both `mlrt::Execute` and all launched
// AsyncOpKernels have finished. Also records `&execution_context` as
// `root_execution_context` so child `mlrt.async` streams can reference a
// context whose lifetime spans all deferred ops.
inline tsl::RCReference<tsl::AsyncValue> SetUpExitAndDeferredOpsHandler(
    mlrt::ExecutionContext& execution_context) {
  struct PendingCompletionState {
    absl::Mutex mu;
    int num_pending ABSL_GUARDED_BY(mu) = 1;
    tsl::AsyncValueRef<tsl::Chain> chain =
        tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

    void Increment() {
      absl::MutexLock lock(&mu);
      ++num_pending;
    }

    void Decrement() {
      bool is_done = false;
      {
        absl::MutexLock lock(&mu);
        DCHECK_GT(num_pending, 0);
        is_done = (--num_pending == 0);
      }
      if (is_done) {
        chain.SetStateConcrete();
      }
    }
  };
  auto pending_state = std::make_shared<PendingCompletionState>();
  auto& tf_context = execution_context.GetUserContext<Context>();
  tf_context.set_root_execution_context(&execution_context);
  tf_context.params().inc_num_deferred_ops_function = [pending_state]() {
    pending_state->Increment();
  };
  tf_context.params().dec_num_deferred_ops_function = [pending_state]() {
    pending_state->Decrement();
  };
  execution_context.set_exit_handler(
      [pending_state]() { pending_state->Decrement(); });
  return pending_state->chain.CopyRCRef();
}

}  // namespace tf_mlrt
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_MLRT_KERNEL_CONTEXT_H_
