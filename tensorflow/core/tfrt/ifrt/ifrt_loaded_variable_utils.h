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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_UTILS_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/client.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

// An index to indicate a non per-core executable bundle cache.
inline constexpr int kNoCoreSelectedIndex = -1;

// TODO(b/352551302) Delete VariableDeviceShardingConfigProto.
struct VariableDeviceShardingConfig {
  std::vector<int> device_ids;
  xla::HloSharding hlo_sharding;
};

absl::StatusOr<ifrt_serving::DtypeAndShape> GetDtypeAndShape(
    const ResourceHandle& resource_handle);

// Returns the runtime name from the resource handle. The name will be concat of
// handle's container name and handle's name.
std::string GetRuntimeNameFromVarHandle(const ResourceHandle& handle);

// Loads a restored tensor as an IFRT loaded variable and set the restored
// tensor in the `restored_tensor_promise` as output. It is an async loading. We
// look for the restored tensor in `ifrt_restore_tensor_registry` and save a
// future of IFRT loaded variable in `ifrt_loaded_variable_registry`. The caller
// can look for the actual loaded variable value in
// `ifrt_loaded_variable_registry`.
absl::Status AsyncLoadRestoredTensorAsIfrtLoadedVariable(
    absl::string_view runtime_name,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const tsl::thread::ThreadPool& thread_pool,
    const ifrt_serving::IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry,
    ifrt_serving::IfrtLoadedVariableRegistry& ifrt_loaded_variable_registry,
    tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
    const VariableDeviceShardingConfig& sharding_config);

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_UTILS_H_
