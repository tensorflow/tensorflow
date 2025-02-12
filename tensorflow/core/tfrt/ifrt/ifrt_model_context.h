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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/topology.h"
#include "xla/tsl/platform/threadpool.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_executable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tsl/platform/protobuf.h"
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
  explicit IfrtModelContext(
      std::shared_ptr<xla::ifrt::Client> client,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      tsl::thread::ThreadPool* thread_pool,
      std::unique_ptr<tsl::protobuf::Message> compilation_environment_proto)
      : client_(std::move(client)),
        ifrt_serving_core_selector_(ifrt_serving_core_selector),
        thread_pool_(*thread_pool),
        compilation_environment_proto_(
            std::move(compilation_environment_proto)) {}
  IfrtModelContext(
      std::shared_ptr<xla::ifrt::Client> client,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      tsl::thread::ThreadPool* thread_pool, tensorflow::DeviceMgr* device_mgr,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
      std::unique_ptr<tsl::protobuf::Message> compilation_environment_proto,
      std::shared_ptr<const void> topology, TfToHloCompiler* tf_to_hlo_compiler,
      IfrtPersistentCompilationCache* persistent_compilation_cache = nullptr)
      : client_(std::move(client)),
        topology_(topology),
        ifrt_serving_core_selector_(ifrt_serving_core_selector),
        thread_pool_(*thread_pool),
        device_mgr_(device_mgr),
        shape_representation_fn_(shape_representation_fn),
        compilation_environment_proto_(
            std::move(compilation_environment_proto)),
        tf_to_hlo_compiler_(tf_to_hlo_compiler),
        persistent_compilation_cache_(persistent_compilation_cache) {}

  void RegisterHandle(ServingExecutableRegistry::Handle handle) {
    handles_.push_back(std::move(handle));
  }

  std::shared_ptr<xla::ifrt::Client> GetClient() const { return client_; }

  const tensorflow::XlaHelpers::ShapeRepresentationFn&
  GetShapeRepresentationFn() const {
    return shape_representation_fn_;
  }

  tsl::thread::ThreadPool& GetThreadPool() const;

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

  IfrtPersistentCompilationCache* GetPersistentCompilationCache() const {
    return persistent_compilation_cache_;
  }

  tensorflow::DeviceMgr* GetDeviceMgr() const { return device_mgr_; }
  IfrtServingCoreSelector* GetIfrtServingCoreSelector() const {
    return ifrt_serving_core_selector_;
  }

  tfrt::ConcurrentWorkQueue* checkpoint_loader_queue() const {
    return checkpoint_loader_queue_;
  }
  void set_checkpoint_loader_queue(tfrt::ConcurrentWorkQueue* work_queue) {
    checkpoint_loader_queue_ = work_queue;
  }

  void set_default_signature_inputs(
      const DefaultSignatureInputConfig& default_signature_inputs) {
    default_signature_inputs_ = default_signature_inputs;
  }

  const DefaultSignatureInputConfig& default_signature_inputs() const {
    return default_signature_inputs_;
  }

  tsl::protobuf::Message* GetCompilationEnvironmentProto() const {
    return compilation_environment_proto_.get();
  }

  TfToHloCompiler* GetTfToHloCompiler() const { return tf_to_hlo_compiler_; }

  // Freeze the model: release the resources such as host tensors that are used
  // by the device only. The caller guarantees all resources released in this
  // function is no longer in use in regular execution path.
  // After Freeze() is called, no new model signature will be compiled. Using a
  // signature or an input shape that wasn't compiled before the freeze will
  // leads to an error.
  absl::Status Freeze();

  bool IsFrozen() const { return frozen_; }

 private:
  std::shared_ptr<xla::ifrt::Client> client_;
  // Keep hardware specific topology info alive. This is currently used for
  // shape determination.
  std::shared_ptr<const void> topology_;

  IfrtServingCoreSelector* ifrt_serving_core_selector_;  // May be nullptr
  tsl::thread::ThreadPool& thread_pool_;

  tensorflow::DeviceMgr* device_mgr_ = nullptr;  // Not owned.
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn_ =
      tensorflow::IdentityShapeRepresentationFn();
  std::unique_ptr<tsl::protobuf::Message> compilation_environment_proto_ =
      nullptr;

  // Dedicated work queue for heavy task such as variable tensor restoration.
  tfrt::ConcurrentWorkQueue* checkpoint_loader_queue_ = nullptr;

  std::vector<ServingExecutableRegistry::Handle> handles_;

  DefaultSignatureInputConfig default_signature_inputs_;

  IfrtLoadedVariableRegistry loaded_variable_registry_;
  IfrtRestoreTensorRegistry restore_tensor_registry_;
  TfToHloCompiler* tf_to_hlo_compiler_ = nullptr;
  IfrtPersistentCompilationCache* persistent_compilation_cache_ = nullptr;
  bool frozen_ = false;
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_MODEL_CONTEXT_H_
