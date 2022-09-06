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

#include "tensorflow/core/tfrt/eager/tfrt_context.h"

#include <string>
#include <utility>

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime

namespace tfrt {
namespace tf {

TfrtContext::TfrtContext(
    const tensorflow::SessionOptions& opts,
    tensorflow::ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async) {
  tensorflow::tfd::EagerContextResource* eager_context_resource =
      resource_context_
          .GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName, opts,
              default_device_placement_policy, is_async);
  auto eager_context_expected = eager_context_resource->GetTFEagerContext();
  DCHECK(eager_context_expected) << StrCat(eager_context_expected.takeError());
  eager_context_ = eager_context_expected.get();

  eager_ctx_thread_pool_ = std::make_unique<ThreadPoolInterfaceWrapper>(
      eager_context_->GetThreadPool()->AsEigenThreadPool());

  local_thread_pool_.reset(tensorflow::NewThreadPoolFromSessionOptions(opts));

  local_thread_pool_wrapper_ = std::make_unique<ThreadPoolInterfaceWrapper>(
      local_thread_pool_->AsEigenThreadPool());

  tf_thread_pool_work_queue_ =
      std::make_unique<tensorflow::tfrt_stub::TfThreadPoolWorkQueue>(
          /*intra_op_threadpool=*/local_thread_pool_wrapper_.get(),
          /*inter_op_threadpool=*/eager_ctx_thread_pool_.get());
  LOG(INFO) << "Created work queue from TF thread pool. inter op thread pool "
            << "# threads: " << eager_ctx_thread_pool_->NumThreads()
            << " intra op thread pool # threads: "
            << local_thread_pool_wrapper_->NumThreads();

  // Default cpu device name is "/job:localhost/replica:0/task:0/device:CPU:0".
  const std::string& host_cpu_name = eager_context_->HostCPU()->name();

  auto diag_handler = [](const DecodedDiagnostic& diag) {
    LOG(ERROR) << diag.message();
  };

  auto rt = CoreRuntime::Create(diag_handler, CreateMallocAllocator(),
                                CreateMultiThreadedWorkQueue(
                                    /*num_threads=*/4,
                                    /*num_blocking_threads=*/64),
                                host_cpu_name);
  DCHECK(rt) << StrCat(rt.takeError());
  corert_ = std::move(rt.get());
  host_context_ = corert_->GetHostContext();

  // Create multiple (currently virtual) CPU devices according to options.
  // TODO(b/174877837): Support multiple physical cpu devices.
  int requested_num_cpus = 1;
  auto iter = opts.config.device_count().find("CPU");
  if (iter != opts.config.device_count().end()) {
    requested_num_cpus = iter->second;
  }

  std::string cpu_name_prefix{host_cpu_name};
  cpu_name_prefix.pop_back();  // remove the `id` from host cpu device name.
  for (int i = 1; i < requested_num_cpus; ++i) {
    host_context_->GetDeviceManager()->MaybeAddDevice(TakeRef(
        new CpuDevice(absl::StrCat(cpu_name_prefix, std::to_string(i)))));
  }

  // Specifically register RuntimeFallbackOpHandler.
  auto runtime_fallback_op_handler =
      tensorflow::tfd::CreateRuntimeFallbackOpHandler(corert_.get(), "");
  DCHECK(runtime_fallback_op_handler)
      << StrCat(runtime_fallback_op_handler.takeError());
  fallback_op_handler_ = runtime_fallback_op_handler.get();
  corert_->RegisterOpHandler("tf", fallback_op_handler_);

  RegisterOpHandlers(corert_.get(), &resource_context_,
                     eager_context_->local_device_mgr());

  // Set the global host context singleton.
  tensorflow::tfrt_global::GlobalHostContext::Set(corert_->GetHostContext());
}

const tensorflow::DeviceNameUtils::ParsedName& TfrtContext::HostCPUParsedName()
    const {
  return eager_context_->HostCPU()->parsed_name();
}

bool TfrtContext::IsAsync() const { return eager_context_->Executor().Async(); }

TfrtContext::~TfrtContext() {}

}  // namespace tf
}  // namespace tfrt
