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
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context_distributed_manager.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session_options.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

tfrt::Expected<OwnedEagerContext> InitEagerContext(
    DynamicDeviceMgr* device_mgr, const SessionOptions& session_opts,
    ContextDevicePlacementPolicy default_device_placement_policy,
    bool is_async) {
  // Copied from TFE_NewContext.
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  tensorflow::Status status = tensorflow::DeviceFactory::AddDevices(
      session_opts, "/job:localhost/replica:0/task:0", &devices);
  if (!status.ok()) {
    return tfrt::MakeStringError(tfrt::StrCat(status.error_message()));
  }

  if (device_mgr != nullptr) {
    Status s = device_mgr->AddDevices(std::move(devices));
    DCHECK(s.ok()) << "Failed to initialize device manager.";
    auto r = new tensorflow::IntraProcessRendezvous(device_mgr);

    OwnedEagerContext owned_eager_context{new tensorflow::EagerContext(
        session_opts, default_device_placement_policy, is_async, device_mgr,
        /*device_mgr_owned=*/false, r)};

#if !defined(IS_MOBILE_PLATFORM)
    owned_eager_context->SetDistributedManager(
        std::make_unique<tensorflow::EagerContextDistributedManager>(
            owned_eager_context.get()));
#endif

    return std::move(owned_eager_context);
  }

  auto owned_device_mgr =
      std::make_unique<tensorflow::StaticDeviceMgr>(std::move(devices));
  auto r = new tensorflow::IntraProcessRendezvous(owned_device_mgr.get());

  OwnedEagerContext owned_eager_context{new tensorflow::EagerContext(
      session_opts, default_device_placement_policy, is_async,
      owned_device_mgr.release(), /*device_mgr_owned=*/true, r)};

#if !defined(IS_MOBILE_PLATFORM)
  owned_eager_context->SetDistributedManager(
      std::make_unique<tensorflow::EagerContextDistributedManager>(
          owned_eager_context.get()));
#endif

  return std::move(owned_eager_context);
}

tfrt::Expected<OwnedEagerContext> InitEagerContext() {
  tensorflow::SessionOptions session_opts;
  return InitEagerContext(
      /*device_mgr=*/nullptr, session_opts,
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /*is_async=*/false);
}

tfrt::Expected<EagerContext*> GetEagerContext(tfrt::ExecutionContext exec_ctx) {
  tfrt::ResourceContext* resource_context = exec_ctx.resource_context();
  tensorflow::tfd::EagerContextResource* eager_context_resource =
      resource_context
          ->GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  return eager_context_resource->GetTFEagerContext();
}

tfrt::Expected<tfrt::CoreRuntimeOp> GetFallbackOp(tfrt::string_view op_name,
                                                  tfrt::HostContext* host) {
  auto* runtime = tfrt::CoreRuntime::GetFromHostContext(host);
  assert(runtime);
  // TODO(b/161993570): Cleanup this magic string constant.
  constexpr tfrt::string_view kRuntimeFallbackOpHandlerName = "tf";

  tfrt::OpHandler* op_handler = nullptr;
  // TODO(b/165334630): Cleanup GPU macros.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  op_handler = runtime->GetOpHandler(kRuntimeFallbackOpHandlerName);
#else
  constexpr tfrt::string_view kKernelFallbackOpHandlerName = "tfkernel";
  op_handler = runtime->GetOpHandler(kKernelFallbackOpHandlerName);
  if (op_handler == nullptr) {
    op_handler = runtime->GetOpHandler(kRuntimeFallbackOpHandlerName);
  }
#endif
  assert(op_handler && "fallback op_handler not found");

  return runtime->MakeOp(op_name, op_handler);
}

}  // namespace tfd
}  // namespace tensorflow
