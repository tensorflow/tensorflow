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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/ifrt/tf_host_callback.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

class IfrtServingExecutable {
 public:
  IfrtServingExecutable(
      int64_t program_id, absl::string_view model_name,
      absl::string_view signature_name,
      mlir::OwningOpRef<mlir::ModuleOp> module,
      std::shared_ptr<xla::ifrt::Client> client,
      const tsl::thread::ThreadPool* thread_pool,
      IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
      const IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_queue,
      tensorflow::StaticDeviceMgr* device_mgr,
      tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn,
      IfrtServingCoreSelector* ifrt_serving_core_selector)
      : program_id_(program_id),
        model_name_(std::string(model_name)),
        signature_name_(std::string(signature_name)),
        module_(std::move(module)),
        ifrt_client_(std::move(client)),
        thread_pool_(*thread_pool),
        ifrt_loaded_variable_registry_(*ifrt_loaded_variable_registry),
        ifrt_restore_tensor_registry_(*ifrt_restore_tensor_registry),
        checkpoint_loader_queue_(checkpoint_loader_queue),
        device_mgr_(device_mgr),
        shape_representation_fn_(std::move(shape_representation_fn)),
        ifrt_serving_core_selector_(std::move(ifrt_serving_core_selector)) {}

  // Movable but not copyable.
  IfrtServingExecutable(IfrtServingExecutable&& other) = default;
  IfrtServingExecutable& operator=(IfrtServingExecutable&& other) = default;
  IfrtServingExecutable(const IfrtServingExecutable& other) = delete;
  IfrtServingExecutable& operator=(const IfrtServingExecutable& other) = delete;

  absl::string_view model_name() const { return model_name_; }
  absl::string_view signature_name() const { return signature_name_; }

  // Executes the computation.
  // variable_arg_indices are in sorted order.
  absl::StatusOr<std::vector<tensorflow::Tensor>> Execute(
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices);

  int num_executables() const {
    absl::MutexLock lock(&mutex_);
    return executable_bundles_.size();
  }

 private:
  // In memory cache key.
  struct Key {
    std::vector<tensorflow::TensorShape> input_shapes;
    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      for (const auto& shape : key.input_shapes) {
        for (auto size : shape.dim_sizes()) {
          h = H::combine(std::move(h), size);
        }
      }
      return h;
    }

    friend bool operator==(const Key& x, const Key& y) {
      return x.input_shapes == y.input_shapes;
    }
  };

  struct CachedExecutableBundle {
    std::unique_ptr<xla::ifrt::LoadedExecutable> ifrt_executable;
    tensorflow::tpu::TPUCompileMetadataProto compile_metadata;
    std::vector<std::unique_ptr<TfHostCallback>> host_callbacks;

    CachedExecutableBundle() = default;
    // Move only
    CachedExecutableBundle(CachedExecutableBundle&& other) = default;
    CachedExecutableBundle& operator=(CachedExecutableBundle&& other) = default;
    CachedExecutableBundle(const CachedExecutableBundle& other) = delete;
    CachedExecutableBundle& operator=(const CachedExecutableBundle& other) =
        delete;
  };

  int64_t program_id_;
  using SharedCachedExecutableBundle = std::shared_ptr<CachedExecutableBundle>;

  std::string model_name_;
  std::string signature_name_;

  mlir::OwningOpRef<mlir::ModuleOp> module_;

  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  const tsl::thread::ThreadPool& thread_pool_;

  IfrtLoadedVariableRegistry& ifrt_loaded_variable_registry_;
  const IfrtRestoreTensorRegistry& ifrt_restore_tensor_registry_;
  tfrt::ConcurrentWorkQueue* checkpoint_loader_queue_;
  tensorflow::StaticDeviceMgr* device_mgr_;  // Not owned. For host callback.
  tensorflow::XlaHelpers::ShapeRepresentationFn shape_representation_fn_;
  IfrtServingCoreSelector* ifrt_serving_core_selector_;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<
      Key, xla::ifrt::Future<absl::StatusOr<SharedCachedExecutableBundle>>>
      executable_bundles_ ABSL_GUARDED_BY(mutex_);

  // Asynchronously load the restored variable tensors to Ifrt array.
  absl::Status AsyncLoadIfrtArray(
      absl::Span<const tensorflow::Tensor> inputs,
      absl::Span<const int> variable_arg_indices,
      const CachedExecutableBundle& executable_bundle,
      const std::vector<xla::ifrt::Device*>& devices);

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> ConvertTensorToArray(
      const tensorflow::Tensor& tensor,
      const xla::ifrt::DeviceList& device_list,
      const xla::OpSharding& sharding);

  xla::ifrt::Future<absl::StatusOr<SharedCachedExecutableBundle>>
  LookUpOrCreateExecutable(
      const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
      absl::Span<const DtypeAndShape> dtypes_and_shapes);
  absl::StatusOr<IfrtServingExecutable::SharedCachedExecutableBundle>
  CreateExecutableSynchronously(
      const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata,
      absl::Span<const DtypeAndShape> dtypes_and_shapes);

  absl::StatusOr<std::unique_ptr<xla::ifrt::Sharding>> CreateSharding(
      int num_devices, const xla::ifrt::Shape& arg_xla_shape,
      const xla::ifrt::Shape& sharded_shapes);

  std::vector<xla::ifrt::Shape> GetArgShape(
      int arg_index, const CachedExecutableBundle& entry);

  bool UsePortableExecution(
      const tensorflow::tpu::TPUCompileMetadataProto& compile_metadata);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_SERVING_EXECUTABLE_H_
