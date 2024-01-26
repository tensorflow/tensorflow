
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

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

// Enable Eigen::ThreadPoolDevice structure definition, rather than just
// declaration.
#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/python/ifrt/array.h"
#include "tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

const Eigen::ThreadPoolDevice& IfrtModelContext::GetThreadPoolDevice() const {
  return thread_pool_device_;
}
absl::Status IfrtModelContext::RegisterLoadedVariable(
    absl::string_view name,
    tsl::RCReference<xla::ifrt::Array> loaded_variable) {
  absl::MutexLock lock(&mutex_);
  auto& variable = loaded_variable_map_[name];
  if (variable != nullptr) {
    return absl::AlreadyExistsError(
        absl::StrCat("Variable '", name, "' already exists."));
  }
  variable = std::move(loaded_variable);
  return absl::OkStatus();
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
IfrtModelContext::GetLoadedVariable(absl::string_view name) const {
  absl::MutexLock lock(&mutex_);
  auto it = loaded_variable_map_.find(name);
  if (it == loaded_variable_map_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Variable '", name, "' not found."));
  }
  return it->second;
}

}  // namespace ifrt_serving
}  // namespace tensorflow
