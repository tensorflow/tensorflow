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
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"

#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/attr_util.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner_cache.h"
#include "tfrt/core_runtime/dispatch_utils.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_invocation.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_metadata_function.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {
namespace tfd {

class KernelFallbackOpHandler : public tfrt::OpHandler {
 public:
  ~KernelFallbackOpHandler() override;

  llvm::Expected<tfrt::CoreRuntimeOp> MakeOp(
      tfrt::string_view op_name) override;

  // TODO(b/166199701) obtain result device from the result tensor, similar to
  // what runtime fallback op handler does.
  tfrt::RCReference<tfrt::Device> GetDeviceRef() { return device_; }

  tfrt::Device* device() const { return device_.get(); }

 private:
  explicit KernelFallbackOpHandler(tfrt::CoreRuntime* runtime,
                                   tfrt::RCReference<tfrt::Device> device);
  friend llvm::Expected<tfrt::OpHandler*> CreateKernelFallbackOpHandler(
      tfrt::CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device);

  llvm::Error Initialize();
  tfrt::RCReference<tfrt::Device> device_;
};

namespace {

using ::tensorflow::tfrt_stub::OpKernelRunner;
using tfrt::AsyncValue;
using tfrt::AsyncValueRef;
using tfrt::Chain;
using tfrt::CoreRuntime;
using tfrt::CoreRuntimeOp;
using tfrt::ExecutionContext;
using tfrt::Expected;
using tfrt::OpAttrsRef;
using tfrt::OpHandler;
using tfrt::OpInvocation;
using tfrt::OpMetadataFn;
using tfrt::raw_ostream;
using tfrt::RCReference;
using tfrt::string_view;
using tfrt::TensorMetadata;

using CompatDispatchFn = AsyncValueRef<Chain> (*)(
    const ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, tfrt::ArrayRef<tfrt::Tensor*> arguments,
    tfrt::MutableArrayRef<RCReference<AsyncValue>> results,
    const KernelFallbackCompatRequestState& fallback_request_state,
    const OpKernelRunner& op_kernel_runner);

struct CompatOpEntry {
  // TODO(tfrt-devs): Avoid having string here, which can be expensive to copy.
  std::string op_name;
  OpMetadataFn metadata_fn = nullptr;
  // All ops use the same dispatch function.
  CompatDispatchFn dispatch_fn =
      &KernelFallbackExecuteCompatCoreRuntimeDispatch;
  KernelFallbackCompatRequestState* fallback_request_state = nullptr;
  OpKernelRunner* op_kernel_runner = nullptr;
};

struct KernelFallbackOpHandlerCompatTraits {
  using InputTensorTy = tfrt::Tensor;
  using OpEntryTy = CompatOpEntry;
  using OpHandlerInfoTy = KernelFallbackOpHandler*;

  static void Dispatch(const CompatOpEntry& op_entry,
                       KernelFallbackOpHandler* tf_op_handler,
                       llvm::ArrayRef<tfrt::Tensor*> inputs,
                       const OpAttrsRef& attrs,
                       llvm::ArrayRef<TensorMetadata> result_mds,
                       llvm::MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    auto ch = op_entry.dispatch_fn(
        exec_ctx, op_entry.op_name, tf_op_handler->device()->name(), inputs,
        results, *op_entry.fallback_request_state, *op_entry.op_kernel_runner);

    if (chain) *chain = std::move(ch);
  }

  // TODO(fishx): Remove this method.
  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(KernelFallbackOpHandler* kernel_fallback_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    return kernel_fallback_op_handler->GetDeviceRef();
  }

  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(const CompatOpEntry& op_entry,
                  KernelFallbackOpHandler* kernel_fallback_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  int index, const ExecutionContext& exec_ctx) {
    auto* op_kernel = op_entry.op_kernel_runner->op_kernel();
    DCHECK(index < op_kernel->num_outputs());
    // NOTE: For DT_RESOURCE, we use the resource device as the device of the
    // resource handle.
    if (op_kernel->output_memory_types()[index] == MemoryType::HOST_MEMORY &&
        op_kernel->output_type(index) != DT_RESOURCE) {
      return exec_ctx.host()->GetHostDeviceRef();
    } else {
      return kernel_fallback_op_handler->GetDeviceRef();
    }
  }
};

}  // namespace

Expected<CoreRuntimeOp> KernelFallbackOpHandler::MakeOp(string_view op_name) {
  // NOTE(fishx): Copying string here will cost extra overhead in graph
  // execution. Because in current implementation, we needs to prepare the op
  // before each executions.
  // TODO(fishx): Avoid this heap allocation by getting op registration
  // information from current TF.
  op_name.consume_front("tf.");
  return CoreRuntimeOp(
      [op_name = op_name.str(), this](const OpInvocation& invocation) {
        auto propagate_error = [&invocation](Status s) {
          auto error = tfrt::EmitErrorAsync(
              invocation.exec_ctx,
              absl::Status(
                  ToAbslStatus(s).code(),
                  tfrt::StrCat("Error running kernel fallback OpHandler ",
                               invocation.op_name, ":", s.error_message())));
          for (auto& result : invocation.results) {
            result = tfrt::TensorHandle::CreateError(error.CopyRef());
          }
          if (invocation.chain) {
            *invocation.chain = error.CopyRef();
          }
        };
        // If the op does not have outputs, then it is expected to output an
        // out chain.
        bool update_chain = invocation.results.empty();
        CompatOpEntry fallback_op_entry;
        fallback_op_entry.op_name = std::move(op_name);

        // Convert the argument tensors to RuntimeFallbackTensors.
        for (auto& argument : invocation.arguments) {
          argument = argument.TransferToSameDevice(
              invocation.exec_ctx, KernelFallbackTensor::kTensorType);
        }

        fallback_op_entry.fallback_request_state =
            invocation.exec_ctx.request_ctx()
                ->GetDataIfExists<KernelFallbackCompatRequestState>();

        if (!fallback_op_entry.fallback_request_state) {
          propagate_error(tensorflow::errors::NotFound(
              "KernelFallbackCompatRequestState not found in RequestContext."));
          return;
        }

        DCHECK(invocation.exec_ctx.location());

        DCHECK(invocation.exec_ctx.request_ctx()->resource_context());
        auto* runner_cache =
            invocation.exec_ctx.request_ctx()
                ->resource_context()
                ->GetOrCreateResource<tfrt_stub::OpKernelRunnerCache>(
                    kOpKernelRunnerCacheResourceName);

        auto kernel_runner_or_status = runner_cache->GetOrCreate(
            invocation.exec_ctx.location(),
            ToAbslStringView(fallback_op_entry.op_name),
            ToAbslStringView(device()->name()), invocation.arguments.size(),
            [&attrs = invocation.attrs, host = invocation.exec_ctx.host()](
                tensorflow::AttrValueMap* attr_value_map) {
              if (auto error =
                      tfd::FillAttrValueMap(attrs, host, attr_value_map))
                return tensorflow::errors::InvalidArgument(tfrt::StrCat(error));
              return OkStatus();
            },
            fallback_op_entry.fallback_request_state->device_manager(),
            fallback_op_entry.fallback_request_state
                ->process_function_library_runtime());

        if (!kernel_runner_or_status.ok()) {
          propagate_error(kernel_runner_or_status.status());
          return;
        }
        fallback_op_entry.op_kernel_runner = kernel_runner_or_status.value();

        tfrt::ExecuteOnOpHandler<KernelFallbackOpHandlerCompatTraits>(
            update_chain, invocation, fallback_op_entry, this);
      },
      // device and arg_tensor_type are currently not used in kernel fallback
      // ops.
      /*is_fallback=*/true, /*device=*/device_);
}

llvm::Expected<tfrt::OpHandler*> CreateKernelFallbackOpHandler(
    tfrt::CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device) {
  std::unique_ptr<KernelFallbackOpHandler> op_handler(
      new KernelFallbackOpHandler(runtime, std::move(device)));
  if (auto error = op_handler->Initialize()) {
    return std::move(error);
  }
  auto op_handler_ptr = op_handler.get();
  runtime->TakeOpHandler(std::move(op_handler));
  return op_handler_ptr;
}

KernelFallbackOpHandler::KernelFallbackOpHandler(
    CoreRuntime* runtime, RCReference<tfrt::Device> device)
    : OpHandler("tfkernel", runtime, nullptr), device_(std::move(device)) {}

KernelFallbackOpHandler::~KernelFallbackOpHandler() {}

llvm::Error KernelFallbackOpHandler::Initialize() {
  return llvm::Error::success();
}

}  // namespace tfd
}  // namespace tensorflow
