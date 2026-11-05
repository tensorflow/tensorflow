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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_H_

#include <stdbool.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_persistent_compilation_cache.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_program.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/sharding_utils.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

// Serves one call site of a TPU program: executes the program with the
// variables bound at that call site.
//
// The program itself (its module, compilation and compiled executables) is an
// `IfrtProgram`. This class looks up the executable compiled for the inputs'
// shapes, loads the call site's variables on it and runs it.
class IfrtServingExecutable {
 public:
  // Creates an executable over a program of its own, built from `module`.
  // TODO(b/514902739): Remove the default value of `use_output_arena`.
  static absl::StatusOr<std::unique_ptr<IfrtServingExecutable>> Create(
      int64_t program_id, absl::string_view model_name,
      absl::string_view signature_name,
      mlir::OwningOpRef<mlir::ModuleOp> module,
      std::shared_ptr<xla::ifrt::Client> client,
      tsl::thread::ThreadPool* thread_pool,
      IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
      const IfrtRestoreTensorRegistry* ifrt_restore,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
      tensorflow::DeviceMgr* device_mgr,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      std::variant<tsl::protobuf::Message*,
                   xla::CompileOptions::EnvironmentOptionOverrides>
          compilation_env_or_overrides,
      TfToHloCompiler* tf_to_hlo_compiler,
      IfrtPersistentCompilationCache* persistent_compilation_cache,
      H2DTransferExecutorFactory* h2d_transfer_executor_factory,
      bool use_output_arena = false,
      bool use_undonatable_buffer_converter = false);

  // Creates an executable over an existing `program`, which may be shared with
  // the executables of other call sites of the same TPU program. Each
  // executable binds the variables of its own call site.
  static absl::StatusOr<std::unique_ptr<IfrtServingExecutable>> Create(
      int64_t program_id, std::shared_ptr<IfrtProgram> program,
      std::shared_ptr<xla::ifrt::Client> client,
      tsl::thread::ThreadPool* thread_pool,
      IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
      const IfrtRestoreTensorRegistry* ifrt_restore,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      H2DTransferExecutorFactory* h2d_transfer_executor_factory,
      bool use_output_arena = false,
      bool use_undonatable_buffer_converter = false);

  // Movable but not copyable.
  IfrtServingExecutable(IfrtServingExecutable&& other) = default;
  IfrtServingExecutable& operator=(IfrtServingExecutable&& other) = default;
  IfrtServingExecutable(const IfrtServingExecutable& other) = delete;
  IfrtServingExecutable& operator=(const IfrtServingExecutable& other) = delete;

  absl::string_view model_name() const { return program_->model_name(); }
  absl::string_view signature_name() const {
    return program_->signature_name();
  }

  // Executes the computation.
  // variable_arg_indices are in sorted order.
  absl::StatusOr<std::vector<tensorflow::Tensor>> Execute(
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices);

  // Executes the computation asynchronously.
  // variable_arg_indices are in sorted order.
  absl::StatusOr<tsl::Future<std::vector<tensorflow::Tensor>>> ExecuteAsync(
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices);

  // Freezes the model. After the Freeze(), JIT compile is not supported and
  // Execute() will return error if inputs contain uncompiled shapes.
  void Freeze();

  // Returns the number of input shape sets compiled (or failed to compile) so
  // far by this executable's program.
  int num_executables() const { return program_->num_compiled(); }

  // The program this executable runs.
  const std::shared_ptr<IfrtProgram>& program() const { return program_; }

 private:
  friend class IfrtBackendCompilerTest;

  using CompiledExecutable = IfrtProgram::CompiledExecutable;
  using Key = IfrtProgram::Key;
  using KeyView = IfrtProgram::KeyView;
  using KeyHash = IfrtProgram::KeyHash;
  using KeyEq = IfrtProgram::KeyEq;

  // What this executable needs to run for one set of input shapes: the
  // compiled program and the variables bound by this executable's call site,
  // loaded on the device(s).
  struct CachedExecutableBundle {
    std::shared_ptr<const CompiledExecutable> compiled;

    using LoadedVariableList =
        std::vector<IfrtLoadedVariableRegistry::LoadedVariable>;
    using DeviceLoadedVariableListMap =
        absl::flat_hash_map<xla::ifrt::DeviceId, LoadedVariableList>;
    std::variant<LoadedVariableList, DeviceLoadedVariableListMap>
        variable_arrays;
  };

  IfrtServingExecutable(
      int64_t program_id, std::shared_ptr<IfrtProgram> program,
      std::shared_ptr<xla::ifrt::Client> client,
      tsl::thread::ThreadPool* thread_pool,
      IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
      const IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
      IfrtServingCoreSelector* ifrt_serving_core_selector,
      H2DTransferExecutorFactory* h2d_transfer_executor_factory,
      bool use_output_arena = false,
      bool use_undonatable_buffer_converter = false)
      : program_id_(program_id),
        program_(std::move(program)),
        assigned_device_ids_([this] {
          std::vector<int> assigned_device_ids;
          assigned_device_ids.reserve(program_->assigned_device_list()->size());
          for (const auto& device :
               program_->assigned_device_list()->devices()) {
            assigned_device_ids.push_back(device->Id().value());
          }
          return assigned_device_ids;
        }()),
        ifrt_client_(std::move(client)),
        thread_pool_(*thread_pool),
        ifrt_loaded_variable_registry_(*ifrt_loaded_variable_registry),
        ifrt_restore_tensor_registry_(*ifrt_restore_tensor_registry),
        checkpoint_loader_queue_(checkpoint_loader_queue),
        ifrt_serving_core_selector_(ifrt_serving_core_selector),
        h2d_transfer_executor_factory_(h2d_transfer_executor_factory),
        use_output_arena_(use_output_arena),
        use_undonatable_buffer_converter_(use_undonatable_buffer_converter) {
    execute_options_.fill_status = true;
    if (use_output_arena_) {
      execute_options_.custom_options =
          xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
              {"use_output_arena", xla::ifrt::AttributeMap::BoolValue(true)}});
    }
  }

  int64_t program_id_;
  using SharedCachedExecutableBundle = std::shared_ptr<CachedExecutableBundle>;

  struct ExecutionInfo {
    xla::ifrt::LoadedExecutable::ExecuteResult execution_result;
    SharedCachedExecutableBundle executable_bundle;
    xla::ifrt::DeviceListRef device_list;
    std::vector<xla::ifrt::ArrayRef> transfer_result;
    std::shared_ptr<tsl::DeviceReservation> device_reservation;
  };

  // The program this executable runs.
  std::shared_ptr<IfrtProgram> program_;
  // Pre-calculated device IDs to avoid redundant computation on the critical
  // path within the Execute() call.
  const std::vector<int> assigned_device_ids_;

  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  tsl::thread::ThreadPool& thread_pool_;

  IfrtLoadedVariableRegistry& ifrt_loaded_variable_registry_;
  const IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry_;
  tfrt::ConcurrentWorkQueue* checkpoint_loader_queue_;
  IfrtServingCoreSelector* ifrt_serving_core_selector_;

  // Compiled executables with this call site's variables loaded, keyed by
  // input shapes.
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Key, tsl::Future<SharedCachedExecutableBundle>, KeyHash,
                      KeyEq>
      executable_bundles_ ABSL_GUARDED_BY(mutex_);

  H2DTransferExecutorFactory* h2d_transfer_executor_factory_ = nullptr;
  bool use_output_arena_ = false;
  bool use_undonatable_buffer_converter_ = false;

  xla::ifrt::ExecuteOptions execute_options_;

  // Asynchronously load the restored variable tensors to Ifrt array.
  absl::Status LoadAndRegisterVariableOnExecutable(
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices,
      const xla::ifrt::DeviceListRef& device_list,
      CachedExecutableBundle* executable_bundle);
  absl::Status AsyncLoadIfrtArray(absl::Span<const tensorflow::Tensor> inputs,
                                  absl::Span<const int> variable_arg_indices,
                                  const CompiledExecutable& compiled,
                                  const xla::ifrt::DeviceListRef& devices);

  // Returns the cached executable bundle future if it exists, otherwise creates
  // a new one from the executable compiled for `dtypes_and_shapes` (see
  // `IfrtProgram::GetOrCompile`) and calls
  // `LoadAndRegisterVariableOnExecutable` to load variables on it.
  absl::StatusOr<
      tsl::Future<IfrtServingExecutable::SharedCachedExecutableBundle>>
  LookUpOrCreateExecutable(absl::Span<const tensorflow::Tensor> inputs,
                           absl::Span<const DtypeAndShape> dtypes_and_shapes,
                           absl::Span<const int> variable_arg_indices,
                           const xla::ifrt::DeviceListRef& device_list);

  // Core implementation for Execute and ExecuteAsync.
  absl::StatusOr<ExecutionInfo> ExecuteCore(
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices);

  bool UsePortableExecution() const { return program_->UsePortableExecution(); }
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_H_
