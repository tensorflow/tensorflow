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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_

#include <assert.h>
#include <stddef.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tfrt_stub {

class OpKernelRunner {
 public:
  static StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, absl::string_view device_name, int num_args,
      const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
      const tensorflow::DeviceMgr& device_manager,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime);

  static StatusOr<OpKernelRunner> Create(
      absl::string_view op_name, int num_args,
      const std::function<Status(tensorflow::AttrValueMap*)>& attr_builder,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime,
      tensorflow::Device* device);

  OpKernelRunner() = default;

  explicit operator bool() const { return op_kernel_ != nullptr; }

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

// OpKernelRunState keeps the states needed for per-kernel execution.
struct OpKernelRunState {
  gtl::InlinedVector<tensorflow::Tensor, 4> input_tf_tensors;
  gtl::InlinedVector<tensorflow::TensorValue, 4> input_tf_tensor_values;
  OpKernelContext::Params params;

  OpKernelRunState() = default;
  OpKernelRunState(
      const gtl::InlinedVector<tensorflow::TensorValue, 4>& tensor_values,
      const OpKernelContext::Params& p) {
    // `input_tf_tensor_values` contains the reference to all tensor used,
    // while `input_tf_tensors` only contains those needs ownership so their
    // sizes may not match. For this copy assignment, we conservatively copy all
    // tensors.
    input_tf_tensors.reserve(tensor_values.size());
    for (const auto& tensor_value : tensor_values) {
      input_tf_tensors.push_back(*tensor_value.tensor);
    }
    for (auto& tensor : input_tf_tensors) {
      input_tf_tensor_values.emplace_back(&tensor);
    }

    // Since `input_tf_tensor_values` and `params` contains pointers to
    // `input_tf_tensors`, we need to change those pointers to the correct ones
    // after copying.
    params = p;
    params.inputs = &input_tf_tensor_values;
  }

  OpKernelRunState(const OpKernelRunState& other) = delete;
  OpKernelRunState& operator=(const OpKernelRunState& other) = delete;

  ~OpKernelRunState() = default;
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
    // Out of bounds vector access will throw an exception and anyway will crash
    // the binary, prefer a more readable error message.
    CHECK_GT(runners_.size(), index)  // Crash OK
        << "runner index is out of bounds: index=" << index
        << " size=" << runners_.size();
    auto& result = runners_.at(index);
    CHECK(result.has_value())  // Crash OK
        << "runner is not available: index=" << index;
    return &(*result);
  }

 private:
  std::vector<absl::optional<OpKernelRunner>> runners_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_H_
