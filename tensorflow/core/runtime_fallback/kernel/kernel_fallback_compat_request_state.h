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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/support/pointer_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

// FallbackResourceArray holds the tensors that are computed only once during
// initialization and read-only afterwards.
class FallbackResourceArray {
 public:
  // Sets `tensor` in the array at `index`. `index` should be dense and
  // duplicate indices are not allowed.
  void SetResource(int index, tensorflow::tfrt_stub::ImmutableTensor tensor);

  // Returns the resource tensor wrapped in AsyncValue value at `index`.
  tfrt::UnRefCountedAsyncValue<tensorflow::tfrt_stub::FallbackTensor>*
  GetResource(int index) const {
    return resource_async_values_.at(index).get();
  }

  // Returns the resource tensor at `index`.
  const tensorflow::tfrt_stub::FallbackTensor& GetResourceAsFallbackTensor(
      int index) const {
    return resource_async_values_.at(index)->get();
  }

 private:
  // `resources_` holds the ownership of all the resource tensors. Note that it
  // may not be a one-to-one mapping between `resources_` and
  // `resource_async_values_`.
  std::vector<std::unique_ptr<tensorflow::tfrt_stub::ImmutableTensor>>
      resources_;
  // `resource_async_values_` holds the UnRefCountedAsyncValue of the fallback
  // tensors that can be directly used by fallback kernels in the graph.
  std::vector<std::unique_ptr<
      tfrt::UnRefCountedAsyncValue<tensorflow::tfrt_stub::FallbackTensor>>>
      resource_async_values_;
};

// Per-request state in kernel falllback compat mode.
class KernelFallbackCompatRequestState {
 public:
  // NOTE: This is the constructor for training.
  KernelFallbackCompatRequestState(
      std::function<void(std::function<void()>)>* runner,
      const tensorflow::DeviceMgr* device_manager, int64_t step_id,
      tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container,
      std::unique_ptr<CollectiveExecutor::Handle> collective_executor,
      core::RefCountPtr<Rendezvous> rendezvous,
      tfrt_stub::OpKernelRunnerTable* runner_table,
      FallbackResourceArray* resource_array,
      tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
      const absl::optional<SessionMetadata>& model_metadata,
      const tensorflow::ProcessFunctionLibraryRuntime* pflr);

  // NOTE: This is the constructor for inference.
  KernelFallbackCompatRequestState(
      std::function<void(std::function<void()>)>* runner,
      const tensorflow::DeviceMgr* device_manager, int64_t step_id,
      tfrt_stub::OpKernelRunnerTable* runner_table,
      FallbackResourceArray* resource_array,
      tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
      const absl::optional<SessionMetadata>& model_metadata,
      const tensorflow::ProcessFunctionLibraryRuntime* pflr);

  // Returns the user-specified custom device for this request. It is currently
  // only used for configure per-request intra op threadpool.
  tensorflow::Device* custom_device() const { return custom_device_.get(); }

  ScopedStepContainer* step_container() const { return step_container_.get(); }

  const tensorflow::DeviceMgr& device_manager() const {
    return *device_manager_;
  }

  const tensorflow::ProcessFunctionLibraryRuntime&
  process_function_library_runtime() const {
    return *pflr_;
  }

  CollectiveExecutor* collective_executor() const {
    return collective_executor_;
  }

  tfrt_stub::OpKernelRunnerTable* runner_table() const { return runner_table_; }

  FallbackResourceArray* resource_array() const { return resource_array_; }

  std::function<void(std::function<void()>)>* runner() const { return runner_; }

  CancellationManager* cancellation_manager() const {
    return default_cancellation_manager_;
  }

  RendezvousInterface* rendezvous() const { return rendezvous_.get(); }

  void set_log_device_placement(bool log) { log_device_placement_ = log; }
  bool log_device_placement() const { return log_device_placement_; }

  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool() const {
    return intra_op_threadpool_;
  }

  const SessionMetadata& session_metadata() const { return session_metadata_; }

 private:
  // Below are resources needed by current tensorflow.
  std::function<void(std::function<void()>)>* runner_ = nullptr;
  ::tfrt::OwnedOrUnownedPtr<ScopedStepContainer> step_container_;
  std::unique_ptr<tensorflow::Device> custom_device_;
  std::unique_ptr<CollectiveExecutor::Handle> collective_executor_handle_;
  CollectiveExecutor* collective_executor_ = nullptr;
  core::RefCountPtr<Rendezvous> rendezvous_;
  CancellationManager* default_cancellation_manager_ = nullptr;

  const tensorflow::DeviceMgr* device_manager_ = nullptr;

  // `runner_table` holds the prepopulated tensorflow::OpKernel instances for
  // kernel fallback compat mode.
  tfrt_stub::OpKernelRunnerTable* runner_table_ = nullptr;

  // Resource array is used for keeping static values in the runtime. It is
  // accessed through tfrt_fallback_async.set_resource and
  // tfrt_fallback_async.get_resource kernels.
  FallbackResourceArray* resource_array_ = nullptr;

  tensorflow::thread::ThreadPoolInterface* intra_op_threadpool_ = nullptr;

  // Model metadata used for monitoring and tracing purpose.
  SessionMetadata session_metadata_;

  const tensorflow::ProcessFunctionLibraryRuntime* pflr_ = nullptr;

  bool log_device_placement_ = false;
};

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_KERNEL_FALLBACK_COMPAT_REQUEST_STATE_H__
