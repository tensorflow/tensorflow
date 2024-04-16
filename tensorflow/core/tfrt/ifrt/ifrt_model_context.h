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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tsl/concurrency/ref_count.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

inline constexpr absl::string_view kIfrtModelContextName = "IfrtModelContext";

// Device specific configuration not available through ifrt. This should be
// rare.
struct DeviceConfig {
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn =
      tensorflow::IdentityShapeRepresentationFn();
};

// The runtime context for ifrt to be used in TFRT serving.
//
// This class is thread compatible.
class IfrtModelContext {
 public:
  explicit IfrtModelContext(std::shared_ptr<xla::ifrt::Client> client,
                            const tsl::thread::ThreadPool* thread_pool)
      : client_(std::move(client)), thread_pool_(*thread_pool) {}
  IfrtModelContext(
      std::shared_ptr<xla::ifrt::Client> client,
      const tsl::thread::ThreadPool* thread_pool,
      std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn)
      : client_(std::move(client)),
        thread_pool_(*thread_pool),
        device_mgr_(std::move(device_mgr)),
        shape_representation_fn_(shape_representation_fn) {}

  void RegisterHandle(ServingExecutableRegistry::Handle handle) {
    handles_.push_back(std::move(handle));
  }

  std::shared_ptr<xla::ifrt::Client> GetClient() const { return client_; }

  const tensorflow::XlaHelpers::ShapeRepresentationFn&
  GetShapeRepresentationFn() const {
    return shape_representation_fn_;
  }

  const tsl::thread::ThreadPool& GetThreadPool() const;

  const IfrtLoadedVariableRegistry& GetLoadedVariableRegistry() const {
    return loaded_variable_registry_;
  }
  IfrtLoadedVariableRegistry& GetLoadedVariableRegistry() {
    return loaded_variable_registry_;
  }

  const IfrtRestoreTensorRegistry& GetRestoreTensorRegistry() const {
    return restore_tensor_registry_;
  }
  IfrtRestoreTensorRegistry& GetRestoreTensorRegistry() {
    return restore_tensor_registry_;
  }

  tensorflow::StaticDeviceMgr* GetDeviceMgr() const {
    return device_mgr_.get();
  }

  tfrt::ConcurrentWorkQueue* checkpoint_loader_queue() const {
    return checkpoint_loader_queue_;
  }
  void set_checkpoint_loader_queue(tfrt::ConcurrentWorkQueue* work_queue) {
    checkpoint_loader_queue_ = work_queue;
  }

 private:
  std::shared_ptr<xla::ifrt::Client> client_;
  const tsl::thread::ThreadPool& thread_pool_;

  std::unique_ptr<tensorflow::StaticDeviceMgr> device_mgr_;
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn_ =
      tensorflow::IdentityShapeRepresentationFn();

  // Dedicated work queue for heavy task such as variable tensor restoration.
  tfrt::ConcurrentWorkQueue* checkpoint_loader_queue_ = nullptr;

  std::vector<ServingExecutableRegistry::Handle> handles_;

  IfrtLoadedVariableRegistry loaded_variable_registry_;
  IfrtRestoreTensorRegistry restore_tensor_registry_;
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_
