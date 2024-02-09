/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/array.h"
#include "tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

// This class is thread safe.
class IfrtLoadedVariableRegistry {
 public:
  absl::Status RegisterLoadedVariable(
      absl::string_view name,
      tsl::RCReference<xla::ifrt::Array> loaded_variable)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> GetLoadedVariable(
      absl::string_view name) const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, tsl::RCReference<xla::ifrt::Array>>
      loaded_variable_map_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
