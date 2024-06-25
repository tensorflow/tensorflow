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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat_eager.h"

#include <functional>
#include <utility>

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_utils.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tsl/platform/refcount.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {
namespace {

using ::tfrt::string_view;

constexpr char kOpKernelRunnerTableResourceName[] =
    "OpKernelRunnerTableResourceName";
constexpr char kFallbackResourceArray[] = "FallbackResourceArray";


}  // namespace

Status SetUpKernelFallbackCompatRequestContext(
    tfrt::RequestContextBuilder* builder,
    tfrt_stub::OpKernelRunnerTable* runner_table,
    tensorflow::EagerContext* eager_context,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool,
    const absl::optional<SessionMetadata>& model_metadata) {
  auto* resource_array =
      builder->resource_context()->GetOrCreateResource<FallbackResourceArray>(
          kFallbackResourceArray);

  if (runner_table == nullptr)
    runner_table = builder->resource_context()
                       ->GetOrCreateResource<tfrt_stub::OpKernelRunnerTable>(
                           kOpKernelRunnerTableResourceName);

  auto step_id = builder->id();

  Rendezvous::Factory creator = eager_context->RendezvousFactory();
  tsl::core::RefCountPtr<Rendezvous> rendezvous;
  TF_RETURN_IF_ERROR(
      creator(step_id, eager_context->local_device_mgr(), &rendezvous));

  // TODO(hhb): Clean up rendezvous from factory after run.

  auto& fallback_request_state =
      builder->context_data().emplace<KernelFallbackCompatRequestState>(
          GetDefaultRunner(), eager_context->local_device_mgr(), step_id,
          tfrt::OwnedOrUnownedPtr<ScopedStepContainer>{
              eager_context->StepContainer()},
          eager_context->GetCollectiveExecutorHandle(), std::move(rendezvous),
          runner_table, resource_array, user_intra_op_threadpool,
          model_metadata, eager_context->pflr());

  fallback_request_state.set_log_device_placement(
      eager_context->LogDevicePlacement());

  return absl::OkStatus();
}

}  // namespace tfd
}  // namespace tensorflow
