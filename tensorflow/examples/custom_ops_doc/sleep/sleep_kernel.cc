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

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/threadpool.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::errors::Internal;
using ::tensorflow::errors::InvalidArgument;

class AsyncSleepOp : public AsyncOpKernel {
 public:
  explicit AsyncSleepOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}
  AsyncSleepOp(const AsyncSleepOp& other) = delete;
  AsyncSleepOp& operator=(const AsyncSleepOp& other) = delete;
  ~AsyncSleepOp() override = default;

  // Implementations of ComputeAsync() must ensure that `done` is (eventually)
  // called exactly once to signal the completion of the computation. The
  // implementation of ComputeAsync() must not block on the execution of another
  // OpKernel. `done` may be called by the current thread, or by another thread.
  // `context` is guaranteed to stay alive until the `done` callback starts.
  // For example, use OP_REQUIRES_ASYNC which takes the `done` parameter
  // as an input and calls `done` for the case of exiting early with an error
  // (instead of OP_REQUIRES).
  //
  // Since it is possible that the unblocking kernel may never run (due to an
  // error or cancellation), in most cases the AsyncOpKernel should implement
  // cancellation support via `context->cancellation_manager()`.
  // TODO (schwartzedward): should this use cancellation support?
  //
  // WARNING: As soon as the `done` callback starts, `context` and `this` may be
  // deleted. No code depending on these objects should execute after the call
  // to `done`.
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const auto& delay_tensor = ctx->input(0);
    OP_REQUIRES_ASYNC(
        ctx, ::tensorflow::TensorShapeUtils::IsScalar(delay_tensor.shape()),
        InvalidArgument("Input `delay` must be a scalar."),
        done);  // Important: call `done` in every execution path
    const float delay = delay_tensor.flat<float>()(0);
    OP_REQUIRES_ASYNC(ctx, delay >= 0.0,
                      InvalidArgument("Input `delay` must be non-negative."),
                      done);  // Important: call `done` in every execution path
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    OP_REQUIRES_ASYNC(ctx, thread_pool != nullptr,
                      Internal("No thread_pool found."),
                      done);  // Important: call `done` in every execution path

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->allocate_output(0, delay_tensor.shape(), &output_tensor),
        done);  // Important: call `done` in every execution path

    absl::Time now = absl::Now();
    absl::Time when = now + absl::Seconds(delay);
    VLOG(1) << "BEFORE ASYNC SLEEP " << ctx->op_kernel().name() << " now "
            << now << " when " << when;
    thread_pool->Schedule([this, output_tensor, when, done] {
      this->sleeper(output_tensor, when, done);
    });
    // Note that `done` is normally called by sleeper(), it is not normally
    // called by this function.
  }

 private:
  void sleeper(Tensor* output_tensor, absl::Time when, DoneCallback done) {
    absl::Time now = absl::Now();
    int64_t delay_us = 0;
    if (now < when) {
      delay_us = absl::ToInt64Microseconds(when - now);
      VLOG(1) << "MIDDLE ASYNC SLEEP " << delay_us;
      absl::SleepFor(when - now);
      VLOG(1) << "AFTER ASYNC SLEEP " << delay_us;
    } else {
      VLOG(1) << "MIDDLE/AFTER ASYNC SKIP SLEEP";
    }
    auto output = output_tensor->template flat<float>();
    output(0) = static_cast<float>(delay_us) / 1000000.0;
    done();  // Important: call `done` in every execution path
  }
};

class SyncSleepOp : public OpKernel {
 public:
  explicit SyncSleepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  SyncSleepOp(const SyncSleepOp& other) = delete;
  SyncSleepOp& operator=(const SyncSleepOp& other) = delete;
  ~SyncSleepOp() override = default;

  void Compute(OpKernelContext* ctx) override {
    const auto& delay_tensor = ctx->input(0);
    OP_REQUIRES(ctx,
                ::tensorflow::TensorShapeUtils::IsScalar(delay_tensor.shape()),
                InvalidArgument("Input `delay` must be a scalar."));
    const float delay = delay_tensor.flat<float>()(0);
    OP_REQUIRES(ctx, delay >= 0.0,
                InvalidArgument("Input `delay` must be non-negative."));
    VLOG(1) << "BEFORE SYNC SLEEP" << ctx->op_kernel().name();
    absl::SleepFor(absl::Seconds(delay));
    VLOG(1) << "AFTER SYNC SLEEP" << ctx->op_kernel().name();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, delay_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<float>();
    output(0) = delay;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Examples>AsyncSleep").Device(::tensorflow::DEVICE_CPU), AsyncSleepOp)
REGISTER_KERNEL_BUILDER(
    Name("Examples>SyncSleep").Device(::tensorflow::DEVICE_CPU), SyncSleepOp)

}  // namespace custom_op_examples
}  // namespace tensorflow
