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

// This file implements RuntimeFallbackOpHandler, responsible for running TFRT
// ops on Tensorflow.

#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_kernels.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/dispatch_utils.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_invocation.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/device.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime
#include "tfrt/tensor/conversion_registry.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_metadata.h"  // from @tf_runtime
// TODO(b/160798174): Avoid CUDA/ROCM macro.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"  // from @tf_runtime
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace tfd {
// TODO(tfrt-devs): Rename it.
class RuntimeFallbackOpHandler : public tfrt::OpHandler {
 public:
  ~RuntimeFallbackOpHandler() override;

  llvm::Expected<tfrt::CoreRuntimeOp> MakeOp(
      tfrt::string_view op_name) override;

  tfrt::string_view DeviceName() const { return device_->name(); }

  const std::string& TfDeviceName() const { return tf_device_name_; }

  tfrt::RCReference<tfrt::Device> GetDeviceRef() { return device_; }

 private:
  explicit RuntimeFallbackOpHandler(tfrt::CoreRuntime* runtime,
                                    tfrt::RCReference<tfrt::Device> device,
                                    const std::string& tf_device_name);

  llvm::Error Initialize();

  friend llvm::Expected<tfrt::OpHandler*> CreateRuntimeFallbackOpHandler(
      tfrt::CoreRuntime* runtime, tfrt::string_view tf_device_name);

  tfrt::RCReference<tfrt::Device> device_;
  // Tensorflow device name, e.g., /device:CPU:0.
  std::string tf_device_name_;
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
using tfrt::OpInvocation;
using tfrt::OpMetadataFn;
using tfrt::raw_ostream;
using tfrt::RCReference;
using tfrt::string_view;
using tfrt::Tensor;
using tfrt::TensorMetadata;

using RuntimeFallbackDispatchFn = AsyncValueRef<Chain> (*)(
    const ExecutionContext& exec_ctx, const char* op_name,
    const char* device_name, llvm::ArrayRef<Tensor*> arguments,
    const OpAttrsRef& attrs,
    llvm::MutableArrayRef<RCReference<AsyncValue>> results);

struct RuntimeFallbackOpEntry {
  std::string op_name;
  OpMetadataFn metadata_fn = nullptr;
  // All ops use the same dispatch function.
  RuntimeFallbackDispatchFn dispatch_fn = &RuntimeFallbackExecute;
};

static Expected<tfrt::RCReference<tfrt::Device>> GetDeviceFromFallbackTensor(
    const RuntimeFallbackTensor& result_tensor,
    const ExecutionContext& exec_ctx) {
  tensorflow::Status status;
  // Obtain the device. Please note that this device is probably not
  // the device that the TensorHandle is located on. E.g. for a GPU resource
  // its device is GPU but it is physicially located on CPU.
  // We use this device because upper layer (e.g. distributed strategy) may
  // use it for colocation. On the other hand, the actual device is not widely
  // used in upper layers.
  // In the future, if we need BackingDevice in higher layer as well, we can
  // update c_api_tfrt layer to get it directly from tensorflow::TensorHandle.
  const char* tf_device_name =
      result_tensor.GetTensorHandle()->DeviceName(&status);
  if (!status.ok()) {
    return tfrt::MakeStringError(status.error_message());
  }

  // TODO(b/165872892): Unify device name for tests.
  auto device = exec_ctx.host()->GetDeviceManager()->GetDeviceRef<tfrt::Device>(
      tf_device_name);
  if (!device) {
    // Convert device name to the short form, e.g. "GPU:0".
    const char* tfrt_device_name =
        ConvertTfDeviceNameToTfrtDefault(tf_device_name);
    device = exec_ctx.host()->GetDeviceManager()->GetDeviceRef<tfrt::Device>(
        tfrt_device_name);
  }
  assert(device);
  return std::move(device);
}

struct RuntimeFallbackOpHandlerTraits {
  using InputTensorTy = Tensor;
  using OpEntryTy = RuntimeFallbackOpEntry;
  using OpHandlerInfoTy = RuntimeFallbackOpHandler*;

  static void Dispatch(const RuntimeFallbackOpEntry& op_entry,
                       RuntimeFallbackOpHandler* tf_op_handler,
                       llvm::ArrayRef<Tensor*> inputs, const OpAttrsRef& attrs,
                       llvm::ArrayRef<TensorMetadata> result_mds,
                       llvm::MutableArrayRef<RCReference<AsyncValue>> results,
                       AsyncValueRef<Chain>* chain,
                       const ExecutionContext& exec_ctx) {
    // Call RuntimeFallbackExecute.
    auto ch = op_entry.dispatch_fn(exec_ctx, op_entry.op_name.c_str(),
                                   tf_op_handler->TfDeviceName().c_str(),
                                   inputs, attrs, results);

    if (chain) *chain = std::move(ch);
  }

  // TODO(fishx): Remove this method.
  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(RuntimeFallbackOpHandler* tf_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  const ExecutionContext& exec_ctx) {
    if (result_tensor_av.IsAvailable()) {
      if (result_tensor_av.IsError()) {
        return tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>(
            result_tensor_av.CopyRCRef());
      }
      auto expected_device = GetDeviceFromFallbackTensor(
          result_tensor_av.get<RuntimeFallbackTensor>(), exec_ctx);
      if (!expected_device) {
        return tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>(
            tfrt::MakeErrorAsyncValueRef(
                tfrt::StrCat(expected_device.takeError())));
      }
      return std::move(expected_device.get());
    }

    auto result_device =
        tfrt::MakeUnconstructedAsyncValueRef<tfrt::RCReference<tfrt::Device>>();

    result_tensor_av.AndThen([result_tensor_av_ref = result_tensor_av.CopyRef(),
                              result_device = result_device.CopyRef(),
                              exec_ctx] {
      assert(result_tensor_av_ref.IsAvailable());
      if (result_tensor_av_ref.IsError()) {
        result_device.SetError(result_tensor_av_ref.GetError());
      }
      auto expected_device = GetDeviceFromFallbackTensor(
          result_tensor_av_ref.get<RuntimeFallbackTensor>(), exec_ctx);
      tfrt::Emplace(
          result_device,
          GetDeviceFromFallbackTensor(
              result_tensor_av_ref.get<RuntimeFallbackTensor>(), exec_ctx));
    });
    return std::move(result_device);
  }

  static tfrt::Variant<tfrt::RCReference<tfrt::Device>,
                       tfrt::AsyncValueRef<tfrt::RCReference<tfrt::Device>>>
  GetResultDevice(const RuntimeFallbackOpEntry& op_entry,
                  RuntimeFallbackOpHandler* tf_op_handler,
                  const tfrt::AsyncValueRef<tfrt::Tensor>& result_tensor_av,
                  int index, const ExecutionContext& exec_ctx) {
    return GetResultDevice(tf_op_handler, result_tensor_av, exec_ctx);
  }
};

}  // namespace

Expected<CoreRuntimeOp> RuntimeFallbackOpHandler::MakeOp(string_view op_name) {
  // NOTE(fishx): Copying string here will cost extra overhead in graph
  // execution. Because in current implementation, we needs to prepare the op
  // before each executions.
  // TODO(fishx): Avoid this heap allocation by getting op registration
  // information from current TF.
  RuntimeFallbackOpEntry op_entry;
  if (!op_name.consume_front("tf."))
    return tfrt::MakeStringError(op_name, " does not start with 'tf.'");
  op_entry.op_name.assign(op_name.begin(), op_name.end());
  return CoreRuntimeOp(
      [op_entry = std::move(op_entry), this](const OpInvocation& invocation) {
        // If the op does not have outputs, then it is expected to output an
        // out chain.
        bool update_chain = invocation.results.empty();

        // Convert the argument tensors to RuntimeFallbackTensors.
        for (auto& argument : invocation.arguments) {
          argument = argument.TransferToSameDevice(
              invocation.exec_ctx, RuntimeFallbackTensor::kTensorType);
        }

        tfrt::ExecuteOnOpHandler<RuntimeFallbackOpHandlerTraits>(
            update_chain, invocation, std::move(op_entry), this);

// TODO(b/160798174): Avoid CUDA/ROCM macro.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
        // If the RuntimeFallbackTensor contains a tensorflow::TensorHandle
        // that holds a GPU tensor, convert it to tfrt::DenseGpuTensor, and
        // populate the correct device name to the result tfrt::TensorHandle.
        //
        // Note that if the GPU tensor contains a DataType that is not natively
        // supported by TFRT, e.g. Resource DataType, we skip the conversion.
        //
        // If the RuntimeFallbackTensor's tensorflow::TensorHandle holds a CPU
        // tensor, do not convert it to DenseHostTensor (it will be lazily
        // converted) for performance reason.
        for (auto& result : invocation.results) {
          auto* host_ctx = invocation.exec_ctx.host();
          auto* result_tensor_av = result.GetAsyncTensor();

          if (!result_tensor_av->IsAvailable())
            host_ctx->Await(FormRef(result_tensor_av));

          if (result_tensor_av->IsError()) continue;

          auto result_tensor_tf_th =
              result_tensor_av->get<RuntimeFallbackTensor>().GetTensorHandle();

          // Check if we need to convert the RuntimeFallbackTensor.
          if (!(IsGpuTensorHandle(*result_tensor_tf_th) &&
                IsSupportedByTFRTGpu(result_tensor_tf_th->DataType())))
            continue;

          result = result.TransferToSameDevice(
              invocation.exec_ctx, tfrt::gpu::DenseGpuTensor::kTensorType);
        }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      },
      // device and arg_tensor_type are not used in runtime fallback ops.
      /*is_fallback=*/true, /*device=*/device_);
}

llvm::Expected<tfrt::OpHandler*> CreateRuntimeFallbackOpHandler(
    tfrt::CoreRuntime* runtime, tfrt::string_view tf_device_name) {
  // TODO(fishx): Remove the device field from fallback op handler.
  std::unique_ptr<RuntimeFallbackOpHandler> op_handler(
      new RuntimeFallbackOpHandler(
          runtime, runtime->GetHostContext()->GetHostDeviceRef(),
          tf_device_name.str()));
  if (auto error = op_handler->Initialize()) {
    return std::move(error);
  }
  auto op_handler_ptr = op_handler.get();
  runtime->TakeOpHandler(std::move(op_handler));
  return op_handler_ptr;
}

RuntimeFallbackOpHandler::RuntimeFallbackOpHandler(
    CoreRuntime* runtime, tfrt::RCReference<tfrt::Device> device,
    const std::string& tf_device_name)
    : OpHandler("tf", runtime, nullptr),
      device_(std::move(device)),
      tf_device_name_(tf_device_name) {}

RuntimeFallbackOpHandler::~RuntimeFallbackOpHandler() {}

llvm::Error RuntimeFallbackOpHandler::Initialize() {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  Status status = InjectTfGpuResources();
  if (!status.ok()) {
    return tfrt::MakeStringError(tfrt::StrCat("error injecting GPU resources: ",
                                              status.error_message()));
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  return llvm::Error::success();
}

}  // namespace tfd
}  // namespace tensorflow
