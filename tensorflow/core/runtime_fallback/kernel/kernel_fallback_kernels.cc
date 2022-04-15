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

// TFRT kernels for calling directly into current TF kernels, bypassing the
// current TF runtime.

#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_frame.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_utils.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime

namespace tensorflow {

// Directly invoke TFRTOpKernel::Compute on a kernel specified by
// 'op_name'. Pass a TFRTOpKernelContext that forwards to the provided
// AsyncKernelFrame.
//
// Directly invoked kernels must be registered with the
// REGISTER_KERNEL_FALLBACK_KERNEL macro and must work with the
// TFRTOpKernel{,Construction,Context} objects instead of the usual
// OpKernel objects.
static void TFDForwardKernel(tfrt::RemainingArguments arguments,
                             tfrt::RemainingResults results,
                             tfrt::StringAttribute op_name,
                             tfrt::RemainingAttributes attributes,
                             tfrt::AsyncKernelFrame* frame,
                             const tfrt::ExecutionContext& exec_ctx) {
  for (int i = 0; i < results.size(); ++i) {
    results.AllocateAt<tensorflow::Tensor>(i);
  }
  std::string op_name_str = op_name.str();
  tfrt::OpAttrs opattrs;
  Status s = FillOpAttrs(attributes, &opattrs);
  if (!s.ok()) {
    frame->ReportError("TFDForwardKernel: Error while parsing attributes: ",
                       s.error_message());
  }

  tfrt::OpAttrsRef opattrsref(opattrs);
  bool work_enqueued = tfd::KernelFallbackExecute(
      exec_ctx, op_name_str, arguments.values(), results.values(), opattrsref,
      tfd::KernelFallbackOutputType::TENSOR);
  if (!work_enqueued) {
    frame->ReportError("TFDForwardKernel: couldn't EnqueueBlockingWork");
  }
}

// Return an initialized scalar Tensor with the specified value.
static void TFDConstantTensor(tfrt::Argument<int32_t> value,
                              tfrt::Result<Tensor> tensor) {
  // TODO(annarev): tensor.Emplace(value.get()) would be simpler but
  // it causes a missing typeinfo error when using -fno-rtti. Investigate
  // if we can make it work with no-rtti.
  Tensor out(DT_INT32, TensorShape({}));
  out.flat<int32>()(0) = value.get();
  tensor.Emplace(out);
}

// Print a Tensor.
static void TFDPrintTensor(tfrt::Argument<Tensor> tensor) {
  llvm::outs() << "tensor=" << tensor.get().DebugString() << "\n";
  llvm::outs().flush();
}

// Log a Tensor.
static void TFDLogTensor(tfrt::Argument<Tensor> tensor) {
  LOG(INFO) << "tensor=" << tensor.get().DebugString() << "\n";
}

void CreateKernelFallbackOpHandlerKernel(
    tfrt::Result<tfrt::OpHandler*> op_handler,
    const tfrt::ExecutionContext& exec_ctx) {
  auto* runtime = tfrt::CoreRuntime::GetFromHostContext(exec_ctx.host());
  assert(runtime);
  auto op_handler_ptr = tensorflow::tfd::CreateKernelFallbackOpHandler(
      runtime, exec_ctx.host()->GetHostDeviceRef());
  assert(op_handler_ptr);
  op_handler.Emplace(op_handler_ptr.get());
}

tfrt::Chain AddKernelFallbackImplicitConversionKernel(
    tfrt::Argument<tfrt::OpHandler*> op_handler,
    const tfrt::ExecutionContext& exec_ctx) {
  assert(op_handler.get()->GetName() == tfrt::CpuOpHandler::kName);
  tfrt::CpuOpHandler* cpu_op_handler =
      static_cast<tfrt::CpuOpHandler*>(op_handler.get());
  cpu_op_handler->AddImplicitConversion(KernelFallbackTensor::kTensorType,
                                        tfrt::DenseHostTensor::kTensorType);
  cpu_op_handler->AddImplicitConversion(KernelFallbackTensor::kTensorType,
                                        tfrt::AnyScalarHostTensor::kTensorType);
  cpu_op_handler->AddImplicitConversion(KernelFallbackTensor::kTensorType,
                                        tfrt::StringHostTensor::kTensorType);
  return {};
}

void RegisterKernelFallbackKernels(tfrt::KernelRegistry* registry) {
  registry->AddKernel("tfd.forward_kernel", TFRT_KERNEL(TFDForwardKernel));
  registry->AddKernel("tfd.constant_tensor", TFRT_KERNEL(TFDConstantTensor));
  registry->AddKernel("tfd.print_tensor", TFRT_KERNEL(TFDPrintTensor));
  registry->AddKernel("tfd.log_tensor", TFRT_KERNEL(TFDLogTensor));
  registry->AddKernel("corert.create_kernel_fallback_op_handler",
                      TFRT_KERNEL(CreateKernelFallbackOpHandlerKernel));
  registry->AddKernel("corert.add_kernel_fallback_implicit_conversions",
                      TFRT_KERNEL(AddKernelFallbackImplicitConversionKernel));
}
}  // namespace tensorflow
