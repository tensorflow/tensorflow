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

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tsl/platform/errors.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

// ModelRuntimeContext provides model contexts for injected backends to
// initialize their per-model states.
class ModelRuntimeContext {
 public:
  ModelRuntimeContext(const GraphExecutionOptions* graph_execution_options,
                      std::string export_dir,
                      tfrt::ResourceContext* resource_context)
      : graph_execution_options_(graph_execution_options),
        export_dir_(std::move(export_dir)),
        resource_context_(resource_context) {
    DCHECK(graph_execution_options_);
    DCHECK(resource_context_);
  }

  absl::string_view name() const {
    return graph_execution_options_->model_metadata.name();
  }
  int64_t version() const {
    return graph_execution_options_->model_metadata.version();
  }

  absl::string_view export_dir() const { return export_dir_; }

  const GraphDef* graph_def() const { return graph_def_; }
  void set_graph_def(const GraphDef* graph_def) { graph_def_ = graph_def; }

  const CallableOptions* callable_options() const { return callable_options_; }
  void set_callable_options(const CallableOptions* callable_options) {
    callable_options_ = callable_options;
  }

  FunctionLibraryDefinition* function_library_definition() const {
    return flib_def_;
  }
  void set_function_library_definition(FunctionLibraryDefinition* flib_def) {
    flib_def_ = flib_def;
  }

  bool is_local_session() const { return is_local_session_; }

  void set_is_local_session(bool is_local_session) {
    is_local_session_ = is_local_session;
  }

  tfrt::ResourceContext& resource_context() { return *resource_context_; }

  const GraphExecutionOptions& graph_execution_options() const {
    return *graph_execution_options_;
  }

  absl::string_view checkpoint_path() const { return checkpoint_path_; }

  void set_checkpoint_path(absl::string_view checkpoint_path) {
    checkpoint_path_ = checkpoint_path;
  }

 private:
  const GraphExecutionOptions* graph_execution_options_ = nullptr;

  std::string export_dir_;
  const GraphDef* graph_def_ = nullptr;
  const CallableOptions* callable_options_ = nullptr;
  tfrt::ResourceContext* resource_context_ = nullptr;

  FunctionLibraryDefinition* flib_def_ = nullptr;

  bool is_local_session_ = false;
  std::string checkpoint_path_;
};

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
    runtime_resource_fns_.emplace_back(
        [fn = std::move(fn)](ModelRuntimeContext& model_context) {
          fn(&model_context.resource_context());
          return absl::OkStatus();
        });
  }

  void AddCreateRuntimeResourceFn(
      std::function<absl::Status(ModelRuntimeContext& model_context)> fn) {
    runtime_resource_fns_.emplace_back(std::move(fn));
  }

  // `CreateRuntimeResources` populates `resource_ctx` with runtime-related
  // resources.
  //
  // This function is thread-safe.
  absl::Status CreateRuntimeResources(
      ModelRuntimeContext& model_context) const {
    for (auto& fn : runtime_resource_fns_) {
      TF_RETURN_IF_ERROR(fn(model_context));
    }
    return absl::OkStatus();
  }

  ABSL_DEPRECATED("Use the overload that take ModelRuntimeContext instead.")
  void CreateRuntimeResources(const GraphExecutionOptions& options,
                              tfrt::ResourceContext* resource_ctx) const {
    ModelRuntimeContext model_context(
        &options, options.compile_options.saved_model_dir, resource_ctx);
    for (auto& fn : runtime_resource_fns_) {
      auto status = fn(model_context);
      if (!status.ok()) {
        LOG(ERROR) << "Failed to create runtime resource: " << status;
        return;
      }
    }
  }

  void SetCreateRequestQueueFn(
      std::function<
          absl::StatusOr<std::unique_ptr<WorkQueueInterface>>(int64_t)>
          create_request_queue_fn) {
    create_request_queue_fn_ = std::move(create_request_queue_fn);
  }

  // Creates a work queue for a request.
  absl::StatusOr<std::unique_ptr<WorkQueueInterface>> CreateRequestQueue(
      int64_t request_id) const {
    if (create_request_queue_fn_) {
      return create_request_queue_fn_(request_id);
    }

    return work_queue_->InitializeRequest(request_id);
  }

 private:
  explicit Runtime(std::unique_ptr<tfrt::CoreRuntime> core_runtime,
                   WorkQueueInterface* work_queue);

  std::unique_ptr<tfrt::CoreRuntime> core_runtime_;
  std::function<absl::StatusOr<std::unique_ptr<WorkQueueInterface>>(int64_t)>
      create_request_queue_fn_;
  WorkQueueInterface* work_queue_ = nullptr;
  std::vector<std::function<absl::Status(ModelRuntimeContext&)>>
      runtime_resource_fns_;
};

// Get a singleton instance of tfrt_stub::Runtime. Returns nullptr until
// SetGlobalRuntime has been called.
// Not thread safe.
Runtime* GetGlobalRuntime();

// Instantiates the singleton instance of tfrt_stub::Runtime by transferring
// an instance of tfrt_stub::Runtime.
// Not thread safe.
void SetGlobalRuntime(std::unique_ptr<Runtime> runtime);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_RUNTIME_RUNTIME_H_
