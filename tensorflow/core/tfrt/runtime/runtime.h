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
#ifndef TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_
#define TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_

#include <memory>

#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {
class CoreRuntime;
class ConcurrentWorkQueue;
}  // namespace tfrt

namespace tensorflow {
namespace tfrt_stub {

// This defines the runtime abstraction in tensorflow for TFRT. It is supposed
// to provide tensorflow specific functionalities that are implemented using
// TFRT. Currently, the only intended uses for this class are:
//  1) Creating the runtime instance with user specified dependencies (eg.
//  thread pool).
//  2) Creating tensors that can be used by the runtime.
//
// It is temporary and will be replaced by the official
// tensorflow::experimental::cc::Runtime when it lands.
class Runtime {
 public:
  // Creates a runtime instance with specified threading configuration. Returns
  // null upon creation error.
  static std::unique_ptr<Runtime> Create(int num_inter_op_threads,
                                         int num_intra_op_threads = 0);

  // Creates a runtime instance with the specified work_queue. Returns null upon
  // creation error.
  static std::unique_ptr<Runtime> Create(
      std::unique_ptr<WorkQueueInterface> work_queue);

  ~Runtime();

  Runtime(Runtime&&) = default;
  Runtime& operator=(Runtime&&) = default;

  // TODO(tfrt-devs): Add methods for creating TFRT tensors.

  // TODO(chky): Make this method private as it should be only used by
  // tfrt::SavedModel. Simply making tfrt::SavedModel a friend class does not
  // work because the it resides in a different namespace. But we should
  // consider moving it to the same namespace.
  tfrt::CoreRuntime* core_runtime() const { return core_runtime_.get(); }
  WorkQueueInterface* work_queue() const { return work_queue_; }

  // `AddCreateRuntimeResourceFn` allows the client to inject per model
  // resources that are related to system-wide concepts, such as devices, when
  // loading a SavedModel.
  //
  // A longer term plan is to use a Device concept for this purpose, so that
  // Runtime contains a vector of Devices. Since it will take some time to
  // iterate on the Device concept and integrate with the existing
  // `tfrt::Device` class, we use the callback function as a temporary solution.
  //
  // The argument `fn` should be thread-safe.
  void AddCreateRuntimeResourceFn(
      std::function<void(tfrt::ResourceContext*)> fn) {
    runtime_resource_fns_.emplace_back(std::move(fn));
  }

  // `CreateRuntimeResources` populates `resource_ctx` with runtime-related
  // resources.
  //
  // This function is thread-safe.
  void CreateRuntimeResources(tfrt::ResourceContext* resource_ctx) const {
    for (auto& fn : runtime_resource_fns_) {
      fn(resource_ctx);
    }
  }

 private:
  explicit Runtime(std::unique_ptr<tfrt::CoreRuntime> core_runtime,
                   WorkQueueInterface* work_queue);

  std::unique_ptr<tfrt::CoreRuntime> core_runtime_;
  WorkQueueInterface* work_queue_ = nullptr;
  std::vector<std::function<void(tfrt::ResourceContext*)>>
      runtime_resource_fns_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_
