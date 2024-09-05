
/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"


#include "absl/status/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {

tsl::thread::ThreadPool& IfrtModelContext::GetThreadPool() const {
  return thread_pool_;
}

absl::Status IfrtModelContext::Freeze() {
  restore_tensor_registry_.Freeze();
  for (auto& program_handle : handles_) {
    TF_RETURN_IF_ERROR(program_handle.Freeze());
  }
  frozen_ = true;
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
