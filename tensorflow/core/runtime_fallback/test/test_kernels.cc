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

// This library contains test kernels needed by fallback unit tests.

#include <string>
#include <utility>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/attribute_utils.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/host_context/sync_kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/logging.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime

namespace tfd {

namespace {
using ::tensorflow::Env;
using ::tfrt::Attribute;
using ::tfrt::DType;
using ::tfrt::EnqueueWork;
using ::tfrt::ExecutionContext;
using ::tfrt::HostContext;
using ::tfrt::KernelRegistry;
using ::tfrt::MakeAvailableAsyncValueRef;
using ::tfrt::RemainingArguments;
using ::tfrt::RemainingResults;
using ::tfrt::StringHostTensor;
using ::tfrt::TensorHandle;
using ::tfrt::TensorMetadata;

static void CreateTensorHandleWithDelayedAsyncTensor(
    RemainingArguments inputs, RemainingResults output_tensors,
    const ExecutionContext& exec_ctx) {
  std::string kTestString = "test string";
  HostContext* host = exec_ctx.host();
  int32_t sleep_time_us = inputs[0]->get<int32_t>();
  TensorMetadata metadata(DType(DType::String), 1);
  auto tensor_ref =
      StringHostTensor::MakeConstructedAsyncValueRef(metadata, host);
  tensor_ref.get().strings()[0] = kTestString;

  // Mark the AsyncTensor available after the sleep.
  EnqueueWork(exec_ctx, [sleep_time_us, tensor_ref = tensor_ref.CopyRef()]() {
    Env::Default()->SleepForMicroseconds(sleep_time_us);
    TFRT_LOG(INFO) << "Slept for " << sleep_time_us << " microseconds";
    tensor_ref.SetStateConcrete();
    TFRT_LOG(INFO) << "Marked AsyncTensor available.";
  });

  output_tensors[0] = MakeAvailableAsyncValueRef<TensorHandle>(
      host->GetHostDeviceRef(), metadata, std::move(tensor_ref));
  TFRT_LOG(INFO) << "Created TensorHandle (" << kTestString << ")";
}

}  // namespace

void RegisterTestKernels(KernelRegistry* registry) {
  registry->AddKernel(
      "tfrt_fallback_test.create_tensorhandle_with_delayed_async_tensor",
      TFRT_KERNEL(CreateTensorHandleWithDelayedAsyncTensor));
}

}  // namespace tfd
