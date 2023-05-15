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
#include "tensorflow/core/tfrt/utils/tfrt_runtime.h"

#include <memory>
#include <utility>

namespace tensorflow {
namespace tfrt_stub {

TfrtRuntime& TfrtRuntime::GetGlobalTfrtRuntime() {
  static TfrtRuntime* tfrt_runtime = new TfrtRuntime();
  return *tfrt_runtime;
}

void TfrtRuntime::SetRuntime(std::unique_ptr<Runtime> runtime) {
  absl::MutexLock l(&m_);
  runtime_ = std::move(runtime);
}

Runtime* TfrtRuntime::GetRuntime() {
  absl::MutexLock l(&m_);
  return runtime_.get();
}

}  // namespace tfrt_stub
}  // namespace tensorflow
