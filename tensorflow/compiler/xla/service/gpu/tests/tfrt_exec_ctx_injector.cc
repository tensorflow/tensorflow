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

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/service/gpu/bef_thunk.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace {

class TfrtExecutionContextInjector : public ::testing::Environment {
 public:
  void SetUp() override {
    runtime_ = tensorflow::tfrt_stub::Runtime::Create();
    // Create request context and prepare deadline tracker.
    tfrt::RequestContextBuilder request_context_builder(
        runtime_->core_runtime()->GetHostContext(),
        /*resource_context=*/nullptr);
    tensorflow::thread::ThreadPoolInterface* intra_op_threadpool = nullptr;
    ASSERT_TRUE(
        runtime_->work_queue()
            ->InitializeRequest(&request_context_builder, &intra_op_threadpool)
            .ok());
    auto req_ctx = std::move(request_context_builder).build();
    ASSERT_TRUE(static_cast<bool>(req_ctx));
    exec_ctx_ = std::make_unique<tfrt::ExecutionContext>(std::move(*req_ctx));
    xla::gpu::SetExecutionContext(exec_ctx_.get());
  }

  void TearDown() override {
    xla::gpu::SetExecutionContext(nullptr);
    exec_ctx_.reset();
    runtime_.reset();
  }

  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_;
  std::unique_ptr<tfrt::ExecutionContext> exec_ctx_;
};

const ::testing::Environment* const kTfrtExecutionContextInjector =
    ::testing::AddGlobalTestEnvironment(new TfrtExecutionContextInjector);

}  // namespace
