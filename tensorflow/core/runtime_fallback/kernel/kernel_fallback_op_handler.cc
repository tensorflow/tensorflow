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

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tfrt/core_runtime/dispatch_utils.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_invocation.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_metadata_function.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

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
using tfrt::SmallVector;
using tfrt::string_view;
using tfrt::TensorMetadata;

using CompatDispatchFn = AsyncValueRef<Chain> (*)(
    const ExecutionContext& exec_ctx, tfrt::string_view op_name,
    tfrt::string_view device_name, tfrt::ArrayRef<tfrt::Tensor*> arguments,
    tfrt::MutableArrayRef<RCReference<AsyncValue>> results,
    const OpAttrsRef& attrs);

struct CompatOpEntry {
  std::string op_name;
  OpMetadataFn metadata_fn = nullptr;
  // All ops use the same dispatch function.
  CompatDispatchFn dispatch_fn =
      &KernelFallbackExecuteCompatCoreRuntimeDispatch;
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
    auto ch = op_entry.dispatch_fn(exec_ctx, op_entry.op_name,
                                   tf_op_handler->device()->name(), inputs,
                                   results, attrs);

    if (chain) *chain = std::move(ch);
  }

  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(KernelFallbackOpHandler* kernel_fallback_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    return kernel_fallback_op_handler->GetDeviceRef();
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
