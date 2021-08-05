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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_OP_KERNEL_RUNNER_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_OP_KERNEL_RUNNER_H_

#include <assert.h>
#include <stddef.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/meta/type_traits.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_properties.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/tfrt/utils/statusor.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/sync_kernel_frame.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

class OpKernelRunner {
 public:
  static tfrt::StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, absl::string_view device_name, int num_args,
      const std::function<llvm::Error(tensorflow::AttrValueMap*)>& attr_builder,
      const KernelFallbackCompatRequestState& fallback_request_state);

  void Run(OpKernelContext* context) const {
    DVLOG(1) << "KernelFallbackExecuteCompat Running Op: "
             << op_kernel_->def().DebugString()
             << ", on Device: " << device_->name();

    op_kernel_->Compute(context);
  }

  void RunAsync(OpKernelContext* context,
                AsyncOpKernel::DoneCallback done_callback) const;

  bool IsAsync() const { return is_async_; }

  tensorflow::OpKernel* op_kernel() const { return op_kernel_.get(); }
  tensorflow::Device* device() const { return device_; }
  tensorflow::FunctionLibraryRuntime* function_library_runtime() const {
    return function_library_runtime_;
  }
  tensorflow::ResourceMgr* resource_manager() const {
    return resource_manager_;
  }

  const gtl::InlinedVector<AllocatorAttributes, 4>& input_alloc_attrs() const {
    return input_alloc_attrs_;
  }
  const gtl::InlinedVector<AllocatorAttributes, 1>& output_alloc_attrs() const {
    return output_alloc_attrs_;
  }

 private:
  explicit OpKernelRunner(
      tensorflow::Device* device,
      tensorflow::FunctionLibraryRuntime* function_library_runtime,
      std::unique_ptr<OpKernel> op_kernel);

  tensorflow::Device* device_ = nullptr;
  tensorflow::FunctionLibraryRuntime* function_library_runtime_ = nullptr;
  tensorflow::ResourceMgr* resource_manager_ = nullptr;
  std::unique_ptr<OpKernel> op_kernel_;
  bool is_async_ = false;
  gtl::InlinedVector<AllocatorAttributes, 4> input_alloc_attrs_;
  gtl::InlinedVector<AllocatorAttributes, 1> output_alloc_attrs_;
};

class OpLocationKey {
 public:
  explicit OpLocationKey(tfrt::Location loc) : loc_(loc) {}

  template <typename H>
  friend H AbslHashValue(H h, const OpLocationKey& key) {
    // NOTE: Each BEF file has its own LocationHandler. Using LocationHandler
    // as part of cache key here can avoid cache collision between different
    // BEF file.
    return H::combine(std::move(h), key.loc_.data, key.loc_.GetHandler());
  }

  friend bool operator==(const OpLocationKey& x, const OpLocationKey& y) {
    return x.loc_.data == y.loc_.data &&
           x.loc_.GetHandler() == y.loc_.GetHandler();
  }

 private:
  tfrt::Location loc_;
};

// OpKernelRunnerTable for keeping OpKernelRunner instances to avoid expensive
// reinstantiation of OpKernel and other repeated setup per kernel execution.
// OpKernelRunnerTable is thread-compatible.
class OpKernelRunnerTable {
 public:
  OpKernelRunnerTable() = default;

  // Return true if it successfully inserts `runner`. `index` is supposed to be
  // dense.
  bool Insert(int64_t index, OpKernelRunner runner) {
    if (runners_.size() <= index) runners_.resize(index + 1);
    if (runners_[index].has_value()) return false;
    runners_[index] = std::move(runner);
    return true;
  }

  // Return the OpKernelRunner at the corresponding `index` in the table. The
  // result can never be nullptr. It is a fatal error to use an index that is
  // not in the table. Note that the returned pointer will be invalidated if
  // Insert() is called.
  const OpKernelRunner* Get(int64_t index) const {
    DCHECK_GT(runners_.size(), index);
    auto& result = runners_.at(index);
    DCHECK(result.has_value());
    return &(*result);
  }

 private:
  std::vector<absl::optional<OpKernelRunner>> runners_;
};

// OpKernelRunnerCache is similar to OpKernelRunnerTable but thread-safe.
class OpKernelRunnerCache {
 public:
  OpKernelRunnerCache();

  tfrt::StatusOr<OpKernelRunner*> GetOrCreate(
      tfrt::Location loc, absl::string_view op_name,
      absl::string_view device_name, int num_args,
      const std::function<llvm::Error(tensorflow::AttrValueMap*)>& attr_builder,
      const KernelFallbackCompatRequestState& fallback_request_state);

 private:
  mutable mutex mu_;
  absl::flat_hash_map<OpLocationKey, std::unique_ptr<OpKernelRunner>> map_
      TF_GUARDED_BY(mu_);
};

}  // namespace tfd
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_KERNEL_OP_KERNEL_RUNNER_H_
