/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_CACHE_H_
#define TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_CACHE_H_

#include <functional>
#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tfrt/host_context/location.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

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

// OpKernelRunnerCache is similar to OpKernelRunnerTable but thread-safe.
class OpKernelRunnerCache {
 public:
  OpKernelRunnerCache() = default;

  absl::StatusOr<OpKernelRunner*> GetOrCreate(
      tfrt::Location loc, absl::string_view op_name,
      absl::string_view device_name, int num_args,
      const std::function<absl::Status(tensorflow::AttrValueMap*)>&
          attr_builder,
      const tensorflow::DeviceMgr& device_manager,
      const tensorflow::ProcessFunctionLibraryRuntime&
          process_function_library_runtime);

 private:
  mutable mutex mu_;
  absl::flat_hash_map<OpLocationKey, std::unique_ptr<OpKernelRunner>> map_
      TF_GUARDED_BY(mu_);
};

}  // namespace tfrt_stub
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TFRT_FALLBACK_OP_KERNEL_RUNNER_CACHE_H_
