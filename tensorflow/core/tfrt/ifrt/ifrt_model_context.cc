
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

#include <cstddef>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {

tsl::thread::ThreadPool& IfrtModelContext::GetThreadPool() const {
  return thread_pool_;
}

absl::Status IfrtModelContext::Freeze() {
  LOG(INFO) << "IfrtModelContext::Freeze: Freezing restore tensor registry, "
               "loaded variable registry, and "
            << handles_.size() << " program handles.";
  restore_tensor_registry_.Freeze();
  loaded_variable_registry_.Freeze();
  for (size_t i = 0; i < handles_.size(); ++i) {
    LOG(INFO) << "IfrtModelContext::Freeze: Freezing program handle " << i
              << " of " << handles_.size();
    TF_RETURN_IF_ERROR(handles_[i].Freeze());
  }
  frozen_ = true;
  LOG(INFO) << "IfrtModelContext::Freeze: Model context successfully frozen.";
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
