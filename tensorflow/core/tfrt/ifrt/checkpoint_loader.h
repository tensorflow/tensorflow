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

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_restore_tensor_registry.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace ifrt_serving {

// TODO(b/352551302) Move the unit test in ifrt_ops_kernel for restore to test
// this class's APIs.
// Implement the `CheckpointLoaderInterface` by using RestoreV2.
class CheckpointLoader {
 public:
  explicit CheckpointLoader(
      IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry,
      tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue,
      bool use_async_restore = true)
      : ifrt_restore_tensor_registry_(ifrt_restore_tensor_registry),
        checkpoint_loader_work_queue_(checkpoint_loader_work_queue),
        use_async_restore_(use_async_restore) {}
  virtual ~CheckpointLoader() = default;

  // Called before `Load` to do some preparation work.
  virtual absl::Status PrepareRestore(mlir::OwningOpRef<mlir::ModuleOp> module);

  // Load the checkpoint. This API is designed to be compatible with the
  // `tf_mlrt.ifrt_restore_variable` kernel.
  virtual absl::Status Load(
      const tensorflow::tfrt_stub::FallbackTensor& prefix,
      const std::vector<tensorflow::tfrt_stub::FallbackTensor>& var_handles,
      const tensorflow::tfrt_stub::FallbackTensor& tensor_names,
      const tensorflow::tfrt_stub::FallbackTensor& shape_and_slices,
      absl::Span<const tensorflow::DataType> restored_dtypes,
      const std::vector<bool>& truncate_in_cast, tf_mlrt::Context& context);

 protected:
  IfrtRestoreTensorRegistry* ifrt_restore_tensor_registry_;
  tfrt::ConcurrentWorkQueue* checkpoint_loader_work_queue_;
  bool use_async_restore_ = true;
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_CHECKPOINT_LOADER_H_
