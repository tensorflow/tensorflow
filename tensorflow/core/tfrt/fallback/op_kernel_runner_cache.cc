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
#include "tensorflow/core/tfrt/fallback/op_kernel_runner_cache.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace tfrt_stub {

absl::StatusOr<OpKernelRunner*> OpKernelRunnerCache::GetOrCreate(
    tfrt::Location loc, absl::string_view op_name,
    absl::string_view device_name, int num_args,
    const std::function<absl::Status(tensorflow::AttrValueMap*)>& attr_builder,
    const tensorflow::DeviceMgr& device_manager,
    const tensorflow::ProcessFunctionLibraryRuntime&
        process_function_library_runtime) {
  OpLocationKey key(loc);
  {
    tf_shared_lock lock(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      DCHECK_EQ(it->second->op_kernel()->def().op(), op_name);
      return it->second.get();
    }
  }

  mutex_lock lock(mu_);

  auto it = map_.find(key);
  if (it != map_.end()) {
    DCHECK_EQ(it->second->op_kernel()->def().op(), op_name);
    return it->second.get();
  }

  VLOG(1) << "KernelFallbackExecuteCompat creating op " << op_name
          << " at location " << loc.data << " on device " << device_name;

  std::string node_name = absl::StrCat(
      op_name, "_", loc.data, "_", absl::bit_cast<uintptr_t>(loc.GetHandler()));

  TF_ASSIGN_OR_RETURN(
      auto runner, OpKernelRunner::Create(
                       op_name, node_name, device_name, num_args, attr_builder,
                       device_manager, process_function_library_runtime));

  auto runner_uptr = std::make_unique<OpKernelRunner>(std::move(runner));

  auto* runner_ptr = runner_uptr.get();
  auto r = map_.emplace(key, std::move(runner_uptr)).second;
  DCHECK(r);

  return runner_ptr;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
