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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/support/pointer_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

using ::tensorflow::tfrt_stub::OpKernelRunnerTable;

void FallbackResourceArray::SetResource(
    int index, tensorflow::tfrt_stub::ImmutableTensor tensor) {
  if (resource_async_values_.size() <= index) {
    resource_storage_.resize(index + 1);
    resource_async_values_.resize(index + 1);
  }

  DCHECK(resource_storage_[index].get() == nullptr);
  DCHECK(resource_async_values_[index].AsPtr().value() == nullptr);

  resources_.push_back(std::make_unique<tensorflow::tfrt_stub::ImmutableTensor>(
      std::move(tensor)));

  resource_storage_[index] = std::make_unique<
      tfrt::internal::AsyncValueStorage<tfrt_stub::FallbackTensor>>();

  resource_async_values_[index] =
      tfrt::MakeAvailableAsyncValueRef<tfrt_stub::FallbackTensor>(
          *resource_storage_[index], resources_.back().get());
}

KernelFallbackCompatRequestState::KernelFallbackCompatRequestState(
    std::function<void(std::function<void()>)>* runner,
    const tensorflow::DeviceMgr* device_manager, int64_t step_id,
    tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container,
    std::unique_ptr<CollectiveExecutor::Handle> collective_executor_handle,
    core::RefCountPtr<Rendezvous> rendezvous, OpKernelRunnerTable* runner_table,
    FallbackResourceArray* resource_array,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const absl::optional<SessionMetadata>& model_metadata,
    const tensorflow::ProcessFunctionLibraryRuntime* pflr)
    : step_id_(step_id),
      runner_(runner),
      step_container_(std::move(step_container)),
      collective_executor_handle_(std::move(collective_executor_handle)),
      collective_executor_(collective_executor_handle_
                               ? collective_executor_handle_->get()
                               : nullptr),
      rendezvous_(std::move(rendezvous)),
      device_manager_(device_manager),
      runner_table_(runner_table),
      resource_array_(resource_array),
      intra_op_threadpool_(user_intra_op_threadpool),
      pflr_(pflr) {
  DCHECK(runner_);
  DCHECK(device_manager_);
  DCHECK(runner_table_);
  DCHECK(resource_array_);
  DCHECK(rendezvous_);
  DCHECK(pflr_);

  cpu_device_ = device_manager_->HostCPU();
  cpu_function_library_runtime_ = pflr_->GetFLR(cpu_device_->name());
  if (user_intra_op_threadpool != nullptr) {
    custom_cpu_device_ = tensorflow::RenamedDevice::NewRenamedDevice(
        cpu_device_->name(), cpu_device_, /*owns_underlying=*/false,
        /*isolate_session_state=*/false, user_intra_op_threadpool);
    cpu_device_ = custom_cpu_device_.get();

    for (auto* device : device_manager_->ListDevices()) {
      custom_device_[device] = tensorflow::RenamedDevice::NewRenamedDevice(
          device->name(), device, /*owns_underlying=*/false,
          /*isolate_session_state=*/false, user_intra_op_threadpool);
    }
  }
  if (model_metadata.has_value()) {
    session_metadata_ = *model_metadata;
  }
}

KernelFallbackCompatRequestState::KernelFallbackCompatRequestState(
    std::function<void(std::function<void()>)>* runner,
    const tensorflow::DeviceMgr* device_manager, int64_t step_id,
    OpKernelRunnerTable* runner_table, FallbackResourceArray* resource_array,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const absl::optional<SessionMetadata>& model_metadata,
    const tensorflow::ProcessFunctionLibraryRuntime* pflr)
    : KernelFallbackCompatRequestState(
          runner, device_manager, step_id,
          // The following code is copied from
          // third_party/tensorflow/core/common_runtime/direct_session.cc
          tfrt::OwnedOrUnownedPtr<ScopedStepContainer>{
              std::make_unique<ScopedStepContainer>(
                  step_id,
                  [step_id, device_manager](const std::string& name) {
                    for (tensorflow::Device* device :
                         device_manager->ListDevices()) {
                      auto status = device->resource_manager()->Cleanup(name);
                      (void)status;
                      tensorflow::ScopedAllocatorMgr* sam =
                          device->GetScopedAllocatorMgr();
                      if (sam) sam->Cleanup(step_id);
                    }
                  })},
          /*collective_executor=*/nullptr,
          /*rendezvous=*/
          core::RefCountPtr<RefCountedIntraProcessRendezvous>(
              new RefCountedIntraProcessRendezvous(device_manager)),
          runner_table, resource_array, user_intra_op_threadpool,
          model_metadata, pflr) {}

static std::function<void(std::function<void()>)>* GetDefaultRunner() {
  static auto* const default_runner =
      new std::function<void(std::function<void()>)>(
          [](const std::function<void()>& f) { f(); });
  return default_runner;
}

absl::Status SetUpKernelFallbackCompatRequestContext(
    tfrt::RequestContextBuilder* builder,
    const tensorflow::DeviceMgr* device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime* pflr,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    FallbackResourceArray* resource_array,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const absl::optional<SessionMetadata>& model_metadata,
    std::function<void(std::function<void()>)>* runner,
    tfrt_stub::CostRecorder* cost_recorder,
    tfrt::ResourceContext* client_graph_resource_context,
    tensorflow::CancellationManager* cancellation_manager,
    const tensorflow::tfrt_stub::RuntimeConfig* runtime_config) {
  DCHECK(builder);
  DCHECK(device_manager);
  DCHECK(pflr);
  DCHECK(runner_table);
  DCHECK(resource_array);

  auto& fallback_request_state =
      builder->context_data().emplace<KernelFallbackCompatRequestState>(
          runner ? runner : GetDefaultRunner(), device_manager, builder->id(),
          runner_table, resource_array, user_intra_op_threadpool,
          model_metadata, pflr);

  fallback_request_state.set_cost_recorder(cost_recorder);
  fallback_request_state.set_client_graph_resource_context(
      client_graph_resource_context);
  fallback_request_state.set_cancellation_manager(cancellation_manager);
  fallback_request_state.set_runtime_config(runtime_config);

  return absl::OkStatus();
}

}  // namespace tfd
}  // namespace tensorflow
