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

// This library contains test ops and kernels needed by fallback unit tests.

#include <memory>
#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {

REGISTER_OP("TestAsyncIdentity")
    .Input("in: T")
    .Output("out: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, uint32, int32, "
        "int64, complex64, complex128}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

class TestAsyncIdentityKernel : public AsyncOpKernel {
 public:
  explicit TestAsyncIdentityKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& in = ctx->input(0);
    ctx->set_output(0, in);
    done();
  }

 private:
  TestAsyncIdentityKernel(const TestAsyncIdentityKernel&) = delete;
  void operator=(const TestAsyncIdentityKernel&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("TestAsyncIdentity").Device(DEVICE_CPU),
                        TestAsyncIdentityKernel);

// For testing TFRT thread integrity. The AndThen op should not be invoked in
// AsyncOpKernel thread.
REGISTER_OP("TestAsyncTfrtAsyncThread")
    .Input("in: T")
    .Output("out: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, uint32, int32, "
        "int64, complex64, complex128}")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

class TestAsyncTfrtAsyncThreadKernel : public AsyncOpKernel {
 public:
  explicit TestAsyncTfrtAsyncThreadKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    const Tensor& in = ctx->input(0);
    ctx->set_output(0, in);
    Env* env = Env::Default();
    {
      mutex_lock l(mu_);
      thread_ = std::unique_ptr<tsl::Thread>(env->StartThread(
          ThreadOptions(), "test_thread_in_compute_async", [done]() {
            Env* env = Env::Default();
            std::string name;
            env->GetCurrentThreadName(&name);
            LOG(ERROR) << "TestAsyncTfrtAsyncThread thread name: " << name;
            done();
          }));
    }
  }

 private:
  mutex mu_;
  std::unique_ptr<Thread> thread_ TF_GUARDED_BY(mu_);
  TestAsyncTfrtAsyncThreadKernel(const TestAsyncTfrtAsyncThreadKernel&) =
      delete;
  void operator=(const TestAsyncTfrtAsyncThreadKernel&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("TestAsyncTfrtAsyncThread").Device(DEVICE_CPU),
                        TestAsyncTfrtAsyncThreadKernel);

// For testing TFRT thread integrity. This is sync op kernel to print out the
// thread name to confirm the TFRT thread integrity.
REGISTER_OP("TestPrintThreadName");

class TestPrintThreadNameKernel : public OpKernel {
 public:
  explicit TestPrintThreadNameKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    Env* env = Env::Default();
    std::string name;
    env->GetCurrentThreadName(&name);
    LOG(ERROR) << "TestPrintThreadName thread name: " << name;
  }

 private:
  TestPrintThreadNameKernel(const TestPrintThreadNameKernel&) = delete;
  void operator=(const TestPrintThreadNameKernel&) = delete;
};

REGISTER_KERNEL_BUILDER(Name("TestPrintThreadName").Device(DEVICE_CPU),
                        TestPrintThreadNameKernel);

}  // namespace tensorflow
