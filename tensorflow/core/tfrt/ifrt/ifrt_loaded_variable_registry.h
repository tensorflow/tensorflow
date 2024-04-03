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

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/future.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tsl/concurrency/ref_count.h"

namespace tensorflow {
namespace ifrt_serving {

// A placeholder device index used when there is no core selection.
inline constexpr int kSDeviceIndexNoCoreSelection = -1;

// This class is thread safe.
class IfrtLoadedVariableRegistry {
 public:
  struct LoadedVariable {
    DtypeAndShape dtype_and_shape;
    xla::ifrt::Future<absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>> array;
    tensorflow::ifrt_serving::VariableDeviceShardingConfigProto sharding_config;
  };

  using LoadedVariableConstructor =
      absl::AnyInvocable<absl::StatusOr<LoadedVariable>() const>;

  // Tries to register a loaded variable with the given name.
  // Returns an error if the named array does not already exists and
  // loaded_variable_constructor failed to create an array. Note that it returns
  // OK if the named array already exists.
  // loaded_variable_constructor is invoked in the caller thread.
  absl::Status TryRegisterLoadedVariable(
      absl::string_view name,
      const tensorflow::ifrt_serving::VariableDeviceShardingConfigProto&
          sharding_config,
      LoadedVariableConstructor&& loaded_variable_constructor)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Looks for loaded variable per input name and optional device. If device
  // isn't specified, a random loaded variable with the given name will be
  // returned. Otherwise, a loaded variable on specific device will be returned.
  // For variables on SPMD, we currently don't support key by per device index
  // and we use -1 as a common index.
  absl::StatusOr<LoadedVariable> GetLoadedVariable(
      absl::string_view name,
      std::optional<int> device_index = std::nullopt) const
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, absl::flat_hash_map<int, LoadedVariable>>
      loaded_variable_map_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
