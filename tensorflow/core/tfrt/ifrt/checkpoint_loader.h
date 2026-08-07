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
#ifndef TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_
#define TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/tsl/concurrency/future.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {

class ResourceMgr;

namespace ifrt_serving {

// TODO(b/352551302) Move the unit test in ifrt_ops_kernel for restore to test
// this class's APIs.
// Implement the `CheckpointLoaderInterface` by using RestoreV2.
class CheckpointLoader {
 public:
  struct PrepareRestoreArgs {
    mlir::MLIRContext* context;
    tensorflow::MetaGraphDef meta_graph_def;
    tfrt_stub::FallbackState* fallback_state;
    std::string saved_model_dir;
    bool run_placer_grappler_on_functions;
  };

  // Bookkeeping record of a variable materialized in the ResourceManager, so
  // FreezeCleanup can delete the entries that turn out to be device-only.
  struct MaterializedVariable {
    std::string runtime_name;
    std::string container;
    std::string name;
  };

  explicit CheckpointLoader(
      IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue,
      bool use_async_restore = true,
      bool materialize_variables_in_resource_manager = false)
      : ifrt_restore_tensor_registry_(ifrt_restore_tensor_registry),
        checkpoint_loader_work_queue_(checkpoint_loader_work_queue),
        use_async_restore_(use_async_restore),
        materialize_variables_in_resource_manager_(
            materialize_variables_in_resource_manager) {}
  virtual ~CheckpointLoader() = default;

  bool materialize_variables_in_resource_manager() const {
    return materialize_variables_in_resource_manager_;
  }
  void set_materialize_variables_in_resource_manager(bool enabled) {
    materialize_variables_in_resource_manager_ = enabled;
  }

  // Called before `Load` to do some preparation work.
  virtual absl::Status PrepareRestore(const PrepareRestoreArgs& args);

  // Load the checkpoint. This API is designed to be compatible with the
  // `tf_mlrt.ifrt_restore_variable` kernel.
  virtual absl::Status Load(
      const tensorflow::tfrt_stub::FallbackTensor& prefix,
      const std::vector<tensorflow::tfrt_stub::FallbackTensor>& var_handles,
      const tensorflow::tfrt_stub::FallbackTensor& tensor_names,
      const tensorflow::tfrt_stub::FallbackTensor& shape_and_slices,
      absl::Span<const tensorflow::DataType> restored_dtypes,
      const std::vector<bool>& truncate_in_cast, tf_mlrt::Context& context);

  // Visibility barrier for host-materialized variables: blocks until every
  // variable routed to the ResourceManager by prior `Load` calls has been
  // materialized as a `Var` (or restore failed). MUST be called before the
  // model serves (when materializing variables in ResourceManager: before
  // warmup); a raw fallback `ReadVariableOp` does not await anything, so
  // without this barrier a request racing the asynchronous restore would fail
  // with "Could not find variable".
  absl::Status AwaitMutableVariables();

  // Deletes from the ResourceManager every materialized variable that was
  // loaded on device (`device_variables`) and is NOT needed on host
  // (`host_needed`). Host-only variables (not in `device_variables`) are
  // preserved on the host.
  absl::Status FreezeCleanup(
      const absl::flat_hash_set<std::string>& device_variables,
      const absl::flat_hash_set<std::string>& host_needed);

 protected:
  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry_;
  tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue_;
  bool use_async_restore_ = true;
  bool materialize_variables_in_resource_manager_ = false;

  absl::Mutex mutable_variables_mu_;
  // One future per materialized variable, fulfilled AFTER its `Var` has been
  // created in the ResourceManager, so "future ready" implies "Var exists".
  std::vector<tsl::Future<void>> mutable_variable_ready_futures_
      ABSL_GUARDED_BY(mutable_variables_mu_);

  // When materialize_variables_in_resource_manager is true: everything
  // materialized in the ResourceManager so far.
  std::vector<MaterializedVariable> materialized_variables_
      ABSL_GUARDED_BY(mutable_variables_mu_);
  ResourceMgr* host_resource_manager_ ABSL_GUARDED_BY(mutable_variables_mu_) =
      nullptr;  // Not owned.
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_
