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
#include "tensorflow/core/tfrt/common/global_state.h"

#include <memory>
#include <utility>

#include "xla/pjrt/utils.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_global {
namespace {

tfrt::HostContext* GetStaticHostContext() {
  static ::tfrt::HostContext* host_context = [] {
    // Create host context.
    auto decoded_diagnostic_handler =
        [&](const ::tfrt::DecodedDiagnostic& diag) { abort(); };
    std::unique_ptr<::tfrt::ConcurrentWorkQueue> work_queue =
        ::tfrt::CreateMultiThreadedWorkQueue(
            /*num_threads=*/xla::DefaultThreadPoolSize(),
            /*num_blocking_threads=*/64);
    std::unique_ptr<::tfrt::HostAllocator> host_allocator =
        ::tfrt::CreateMallocAllocator();
    return new ::tfrt::HostContext(decoded_diagnostic_handler,
                                   std::move(host_allocator),
                                   std::move(work_queue));
  }();
  return host_context;
}

}  // namespace

/*static*/ ::tfrt::HostContext* GlobalHostContext::host_ctx_ = nullptr;

/*static*/ void GlobalHostContext::Set(::tfrt::HostContext* host_ctx) {
  host_ctx_ = host_ctx;
}

/*static*/ ::tfrt::HostContext* GlobalHostContext::Get() {
  // If HostContext is explicitly injected at context creation, use it here.
  if (host_ctx_) return host_ctx_;

  // Otherwise we assume it is running TFRT TF OpKernels, and currently it is
  // implicitly created.
  return GetStaticHostContext();
}

ResourceMgr* GetTFGlobalResourceMgr() {
  static ResourceMgr* const rmgr = new ResourceMgr();
  return rmgr;
}

}  // namespace tfrt_global
}  // namespace tensorflow
