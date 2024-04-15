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

#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/future.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

namespace {

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> LoadIfrtVariable(
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const tsl::thread::ThreadPool& thread_pool,
    const tensorflow::Tensor& variable,
    const VariableDeviceShardingConfigProto& sharding_config) {
  std::vector<int> device_ids{sharding_config.device_ids().begin(),
                              sharding_config.device_ids().end()};
  TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                      xla::HloSharding::FromProto(sharding_config.sharding()));
  return tensorflow::ifrt_serving::MakeArrayFromTensor(
      *ifrt_client, variable, sharding_config.device_ids(), hlo_sharding,
      thread_pool);
}

absl::StatusOr<ifrt_serving::DtypeAndShape> GetDtypeAndShape(
    const ResourceHandle& resource_handle) {
  const std::vector<DtypeAndPartialTensorShape>& dtype_and_partial_shapes =
      resource_handle.dtypes_and_shapes();

  if (dtype_and_partial_shapes.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected 1 dtype and shape, got ", dtype_and_partial_shapes.size()));
  }
  ifrt_serving::DtypeAndShape dtype_and_shape;
  if (!dtype_and_partial_shapes.front().shape.AsTensorShape(
          &dtype_and_shape.shape)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to convert partial shape to full tensor shape: ",
                     dtype_and_partial_shapes.front().shape.DebugString()));
  }

  dtype_and_shape.dtype = dtype_and_partial_shapes.front().dtype;
  return dtype_and_shape;
}

}  // namespace

std::string GetRuntimeNameFromVarHandle(const ResourceHandle& handle) {
  return absl::StrCat(handle.container(), "__", handle.name());
}

absl::Status LoadRestoredTensorAsIfrtLoadedVariable(
    const tensorflow::Tensor& variable_handle_tensor,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    const tsl::thread::ThreadPool& thread_pool,
    ifrt_serving::IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry,
    ifrt_serving::IfrtLoadedVariableRegistry& ifrt_loaded_variable_registry,
    tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
    const VariableDeviceShardingConfigProto& sharding_config,
    mlrt::Promise* restored_tensor_promise) {
  if (variable_handle_tensor.dtype() != DT_RESOURCE) {
    return absl::InvalidArgumentError(
        absl::StrCat("variable_handle_tensor is ",
                     DataTypeString(variable_handle_tensor.dtype()),
                     " but expected DT_RESOURCE"));
  }
  const ResourceHandle& handle =
      variable_handle_tensor.scalar<ResourceHandle>()();
  std::string runtime_name = GetRuntimeNameFromVarHandle(handle);
  xla::ifrt::Future<absl::StatusOr<tensorflow::Tensor>> restored_tensor_future =
      ifrt_restore_tensor_registry.Get(runtime_name);
  if (!restored_tensor_future.IsValid()) {
    return absl::InternalError(absl::StrCat(
        "LoadVariableOp: failed to fetch variable tensor: ", runtime_name));
  }

  auto loaded_variable_promise = xla::ifrt::Future<
      absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>>::CreatePromise();
  auto loaded_variable_future =
      xla::ifrt::Future<absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>>(
          loaded_variable_promise);

  TF_ASSIGN_OR_RETURN(ifrt_serving::DtypeAndShape dtype_and_shape,
                      GetDtypeAndShape(handle));
  // TODO(b/330360798) Load variable on devices from the result of core
  // selection.
  TF_RETURN_IF_ERROR(ifrt_loaded_variable_registry.TryRegisterLoadedVariable(
      runtime_name,
      [&]() -> absl::StatusOr<
                ifrt_serving::IfrtLoadedVariableRegistry::LoadedVariable> {
        return ifrt_serving::IfrtLoadedVariableRegistry::LoadedVariable(
            {.dtype_and_shape = dtype_and_shape,
             .array = loaded_variable_future});
      }));
  restored_tensor_future.OnReady(
      [ifrt_client = ifrt_client, &thread_pool = thread_pool,
       checkpoint_loader_queue = checkpoint_loader_queue,
       sharding_config = sharding_config,
       loaded_variable_promise = std::move(loaded_variable_promise),
       restored_tensor_promise = restored_tensor_promise](
          absl::StatusOr<tensorflow::Tensor> restored_tensor) mutable {
        if (!restored_tensor.ok()) {
          loaded_variable_promise.Set(restored_tensor.status());
          std::move(*restored_tensor_promise)
              .SetError(restored_tensor.status());
          return;
        }
        std::move(*restored_tensor_promise)
            .Set<tensorflow::tfrt_stub::FallbackTensor>(*restored_tensor);

        // Transfer tensor to array in a separate thread.
        checkpoint_loader_queue->AddTask(
            [ifrt_client = ifrt_client, &thread_pool = thread_pool,
             sharding_config = std::move(sharding_config),
             restored_tensor = std::move(*restored_tensor),
             loaded_variable_promise =
                 std::move(loaded_variable_promise)]() mutable {
              absl::StatusOr<tsl::RCReference<xla::ifrt::Array>>
                  variable_array =
                      LoadIfrtVariable(ifrt_client, thread_pool,
                                       restored_tensor, sharding_config);
              loaded_variable_promise.Set(std::move(variable_array));
            });
      });
  return absl::OkStatus();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
