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
#include "tensorflow/core/runtime_fallback/util/fallback_test_util.h"

#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_request_context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"

namespace tensorflow {
namespace tfd {

tfrt::ExecutionContext CreateFallbackTestExecutionContext(
    tfrt::HostContext* host, tfrt::ResourceContext* resource_context,
    tensorflow::thread::ThreadPoolInterface* user_intra_op_threadpool) {
  static std::atomic<int64_t> id{0};

  // We should better decouple eager context and resource context. In prod code,
  // we shouldn't store eager context in resource context.
  auto* eager_context_resource =
      resource_context->GetOrCreateResource<EagerContextResource>(
          tensorflow::tfd::kEagerContextResourceName);
  assert(eager_context_resource);
  auto expected_eager_context = eager_context_resource->GetTFEagerContext();
  assert(expected_eager_context);
  auto* eager_context = expected_eager_context.get();
  assert(eager_context);

  // Add a dummy FunctionDef to test creating ops with function attributes.
  const FunctionDef dummy_function_def = FunctionDefHelper::Define(
      /*function_name=*/"dummy_fn",
      /*arg_def=*/{},
      /*return values=*/{},
      /*attr def=*/{},
      /*node_def=*/{});
  tensorflow::Status status = eager_context->AddFunctionDef(dummy_function_def);
  TF_DCHECK_OK(status);

  auto request_id = id.fetch_add(1);
  tfrt::RequestContextBuilder request_context_builder(host, resource_context,
                                                      request_id);
  status = SetUpKernelFallbackCompatRequestContext(
      &request_context_builder, eager_context->local_device_mgr(),
      eager_context->pflr(), user_intra_op_threadpool);
  TF_DCHECK_OK(status);

  status = SetUpTfJitRtRequestContext(&request_context_builder);
  TF_DCHECK_OK(status);

  auto request_context = std::move(request_context_builder).build();
  assert(request_context);

  return tfrt::ExecutionContext{std::move(request_context.get())};
}

}  // namespace tfd
}  // namespace tensorflow
