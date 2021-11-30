/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_executor_api.h"

namespace tensorflow {
namespace tpu {

TfTpu_ExecutorApiFn* ExecutorApiFn() {
  static TfTpu_ExecutorApiFn executor_api_fn;
  return &executor_api_fn;
}

bool IsStreamExecutorEnabled(TfTpu_ExecutorApiFn* executor_api_fn) {
  if (!IsInitialized(executor_api_fn)) {
    return false;
  }
  bool is_se_enabled = false;
  auto* tpu_platform = executor_api_fn->TpuPlatform_NewFn();
  if (tpu_platform != nullptr) {
    is_se_enabled = true;
    executor_api_fn->TpuPlatform_FreeFn(tpu_platform);
  }
  return is_se_enabled;
}

bool IsInitialized(TfTpu_ExecutorApiFn* executor_api_fn) {
  // Check if an arbitrary function pointer is initialized. We could check more
  // functions or add an explicit 'initialized' field to TfTpu_ExecutorApiFn,
  // but this works well enough.
  return executor_api_fn->TpuPlatform_NewFn != nullptr;
}

}  // namespace tpu
}  // namespace tensorflow
