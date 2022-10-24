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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_FUNCTION_CACHE_H_
#define TENSORFLOW_CORE_TFRT_EAGER_FUNCTION_CACHE_H_

#include "tensorflow/compiler/mlir/tfrt/function/function.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime_op.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/aligned_buffer.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/mutex.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

// A reference counted `state` object that contains a BEF file, which represents
// a lowered FunctionDef. The CoreRuntimeOp is a callable handle to the function
// to be called.
class FunctionState : public ReferenceCounted<FunctionState> {
 public:
  static RCReference<FunctionState> CreateFunctionState(
      TfrtDataTypeSlice arg_types, tensorflow::DataTypeSlice ret_types,
      BefBuffer bef_buffer, RCReference<BEFFile> bef_file, CoreRuntimeOp fn,
      std::unique_ptr<tensorflow::tfrt_stub::OpKernelRunnerTable>
          runner_table) {
    return TakeRef(new FunctionState(arg_types, ret_types,
                                     std::move(bef_buffer), std::move(bef_file),
                                     std::move(fn), std::move(runner_table)));
  }

  const CoreRuntimeOp& GetFunc() const { return fn_; }

  const TfrtDataTypeVector& GetArgTypes() { return arg_types_; }

  const tensorflow::DataTypeVector& GetRetTypes() { return ret_types_; }

  tensorflow::tfrt_stub::OpKernelRunnerTable* GetRunnerTable() {
    return runner_table_.get();
  }

 private:
  FunctionState(
      TfrtDataTypeSlice arg_types, tensorflow::DataTypeSlice ret_types,
      BefBuffer bef_buffer, RCReference<BEFFile> bef_file, CoreRuntimeOp fn,
      std::unique_ptr<tensorflow::tfrt_stub::OpKernelRunnerTable> runner_table)
      : arg_types_(arg_types.begin(), arg_types.end()),
        ret_types_(ret_types.begin(), ret_types.end()),
        bef_buffer_(std::move(bef_buffer)),
        bef_file_(std::move(bef_file)),
        fn_(std::move(fn)),
        runner_table_(std::move(runner_table)) {}

  TfrtDataTypeVector arg_types_;
  tensorflow::DataTypeVector ret_types_;
  BefBuffer bef_buffer_;
  RCReference<BEFFile> bef_file_;
  const CoreRuntimeOp fn_;

  // This is the op_kernel cache used by kernel fallback compact mode. We will
  // initialize this table right after lowering the function.
  std::unique_ptr<tensorflow::tfrt_stub::OpKernelRunnerTable> runner_table_;
};

// Cache for a single core runtime op or function (composite op). Thread safe.
class FunctionCache {
 public:
  // Iterate the cache and erase the op(s) with the specified op_name.
  void RemoveFunction(string_view op_name) TFRT_EXCLUDES(cache_mu_);

  struct FunctionCacheResult {
    RCReference<FunctionState> function_state;
    bool is_cache_miss;
  };

  typedef std::function<tensorflow::Status(
      tensorflow::tfrt_stub::OpKernelRunnerTable*,
      RCReference<RequestContext>*)>
      RequestCtxBuilder;

  // Helper function to look up the cache. If miss, insert the function to the
  // cache.
  // When the return status is OK, `result` is set.
  tensorflow::Status GetOrAddFunction(
      const std::string& op_name, const std::string& device_name,
      const tensorflow::DeviceSet& device_set,
      tensorflow::EagerContext* eager_ctx, tfrt::CoreRuntime* corert,
      RequestCtxBuilder request_ctx_fn, Location loc,
      tensorflow::TfrtFunctionCompileOptions compile_options,
      tfrt::ArrayRef<const Device*> input_devices, FunctionCacheResult* result);

  // The following helper functions are for debugging and testing only.
  size_t Size() const {
    mutex_lock l(cache_mu_);
    return cache_.size();
  }

  bool Contains(string_view op_name, string_view device_name) const {
    const CacheKey& cache_key{op_name.str(), device_name.str()};
    mutex_lock l(cache_mu_);
    return cache_.find(cache_key) != cache_.end();
  }

 private:
  // Note: Currently the key is a pair of op_name and device_name. New features
  // may be added in the future.
  struct CacheKey {
    std::string op_name, device_name;

    bool operator==(const CacheKey& other) const {
      return (this->op_name == other.op_name &&
              this->device_name == other.device_name);
    }
  };

  struct CacheKeyHash {
    size_t operator()(const CacheKey& pair) const {
      return std::hash<std::string>()(pair.op_name) ^
             std::hash<std::string>()(pair.device_name);
    }
  };

  mutable mutex cache_mu_;
  std::unordered_map<CacheKey, RCReference<FunctionState>, CacheKeyHash> cache_
      TFRT_GUARDED_BY(cache_mu_);
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_FUNCTION_CACHE_H_
