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
#include "tensorflow/core/tfrt/runtime/runtime.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(RuntimeTest, GlobalRuntimeWorks) {
  // Before SetGlobalRuntime, it is null.
  EXPECT_EQ(GetGlobalRuntime(), nullptr);
  // After SetGlobalRuntime, it is not null.
  SetGlobalRuntime(Runtime::Create(/*num_inter_op_threads=*/4));
  EXPECT_NE(GetGlobalRuntime(), nullptr);
  // It is only allocated once.
  EXPECT_EQ(GetGlobalRuntime(), GetGlobalRuntime());
}

TEST(RuntimeTest, DiagHandler) {
  bool was_called = false;
  auto diag_handler = [&was_called](const tfrt::DecodedDiagnostic& diag) {
    was_called = true;
  };
  auto work_queue =
      WrapDefaultWorkQueue(tfrt::CreateMultiThreadedWorkQueue(1, 1));
  auto runtime = Runtime::Create(std::move(work_queue), diag_handler);
  runtime->core_runtime()->GetHostContext()->diag_handler()(
      tfrt::DecodedDiagnostic(absl::UnknownError("Some failure.")));
  EXPECT_TRUE(was_called);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
