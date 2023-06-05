/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/kernels/transfer_ops.h"

#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_node_context.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_transfer_manager_interface.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"

namespace tensorflow {

TpuTransferAsyncOpKernelBase::TpuTransferAsyncOpKernelBase(
    OpKernelConstruction* ctx, const string& transfer_type,
    int number_of_threads, std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : AsyncOpKernel(ctx),
      transfer_type_(transfer_type),
      transfer_op_(std::move(transfer_op)),
      thread_pool_(new thread::ThreadPool(
          ctx->env(),
          strings::StrCat(transfer_type, "_thread_",
                          SanitizeThreadSuffix(def().name())),
          /*num_threads=*/8)) {}

void TpuTransferAsyncOpKernelBase::ComputeAsync(OpKernelContext* ctx,
                                                DoneCallback done) {
  profiler::TraceMeProducer schedule_activity(
      "TpuTransferAsyncOpKernelBase::ComputeAsync");
  CancellationToken token =
      ctx->cancellation_manager()->get_cancellation_token();
  bool already_cancelled;
  {
    // Only protect registering the cancellation callback as mu_ cannot be held
    // at a point where `done` could be called.
    mutex_lock lock(mu_);
    already_cancelled =
        !ctx->cancellation_manager()->RegisterCallback(token, [this]() {
          mutex_lock lock(mu_);
          transfer_op_->Cancel();
        });
  }
  OP_REQUIRES_ASYNC(ctx, !already_cancelled,
                    errors::Cancelled("Infeed was cancelled."), done);
  thread_pool_->Schedule(
      [this, ctx, done, token,
       traceme_context_id = schedule_activity.GetContextId()]() {
        profiler::TraceMeConsumer compute_activity(
            [this] { return profiler::TraceMeOp(name(), type_string()); },
            traceme_context_id);
        Status s = RunTransfer(ctx);
        ctx->cancellation_manager()->DeregisterCallback(token);
        OP_REQUIRES_OK_ASYNC(ctx, s, done);
        done();
      });
}

Status TpuTransferAsyncOpKernelBase::RunTransferWithOrdinal(
    OpKernelContext* ctx, int device_ordinal) {

  int real_device_ordinal = device_ordinal;
  if (real_device_ordinal < 0) {
    TF_ASSIGN_OR_RETURN(real_device_ordinal,
                        transfer_op_->GetDeviceOrdinal(ctx));
  }

  profiler::TraceMe activity(
      [real_device_ordinal] {
        return profiler::TraceMeEncode(
            "RunTransferWithOrdinal",
            {{"device_ordinal", real_device_ordinal}});
      },
      profiler::kInfo);
  return DoWork(ctx, real_device_ordinal);
}

TpuTransferAsyncOpKernel::TpuTransferAsyncOpKernel(
    OpKernelConstruction* ctx, const string& transfer_type,
    int number_of_threads, std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernelBase(ctx, transfer_type, number_of_threads,
                                   std::move(transfer_op)) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal_));
  if (ctx->device_type() == DeviceType(DEVICE_CPU)) {
    OP_REQUIRES(
        ctx, device_ordinal_ >= 0,
        errors::InvalidArgument(transfer_type,
                                " ops must specify a device_ordinal when "
                                "placed on CPU."));
  }
}

Status TpuTransferAsyncOpKernel::RunTransfer(OpKernelContext* ctx) {
  return RunTransferWithOrdinal(ctx, device_ordinal_);
}

TpuTransferAsyncDynamicOrdinalOpKernel::TpuTransferAsyncDynamicOrdinalOpKernel(
    OpKernelConstruction* ctx, const string& transfer_type,
    int number_of_threads, std::unique_ptr<TpuTransferOpInterface> transfer_op)
    : TpuTransferAsyncOpKernelBase(ctx, transfer_type, number_of_threads,
                                   std::move(transfer_op)) {}

Status TpuTransferAsyncDynamicOrdinalOpKernel::RunTransfer(
    OpKernelContext* ctx) {
  const Tensor& device_ordinal_tensor = ctx->input(0);
  const int device_ordinal = device_ordinal_tensor.scalar<int32>()();
  XlaDevice* xla_device =
      dynamic_cast<XlaDevice*>(ctx->device()->UnderlyingDevice());
  if (((xla_device == nullptr) || (xla_device->device_type() == DEVICE_CPU)) &&
      (device_ordinal < 0)) {
    return errors::InvalidArgument(transfer_type_,
                                   " ops must specify a device_ordinal when "
                                   "placed on CPU.");
  }
  return RunTransferWithOrdinal(ctx, device_ordinal);
}

StreamExecutorTransferOpImpl::StreamExecutorTransferOpImpl()
    : transfer_manager_(
          xla::TpuTransferManagerInterface::GetRegisteredTpuTransferManager()),
      tpu_platform_(tpu::TpuPlatformInterface::GetRegisteredPlatform(
          /*initialize_platform=*/false)) {}

void StreamExecutorTransferOpImpl::Cancel() {
  TF_CHECK_OK(tpu::TpuNodeContext::CloseTpuHost());
}

StatusOr<int> StreamExecutorTransferOpImpl::GetDeviceOrdinal(
    OpKernelContext* ctx) {
  const XlaDevice::Metadata* metadata;
  TF_RETURN_IF_ERROR(XlaDevice::GetMetadata(ctx, &metadata));
  return metadata->device_ordinal();
}

Status StreamExecutorTransferOpImpl::TransferBuffersToInfeed(
    int device_ordinal,
    const std::deque<tensorflow::tpu::NoncopyableBuffer>& buffers) {
  TF_ASSIGN_OR_RETURN(auto* executor, GetStreamExecutor(device_ordinal));
  return transfer_manager_->TransferBuffersToInfeed(executor, buffers);
}

Status StreamExecutorTransferOpImpl::TransferLiteralToInfeed(
    int device_ordinal, const xla::LiteralSlice& literal) {
  TF_ASSIGN_OR_RETURN(auto* executor, GetStreamExecutor(device_ordinal));
  return transfer_manager_->TransferLiteralToInfeed(executor, literal);
}

Status StreamExecutorTransferOpImpl::TransferLiteralFromOutfeed(
    int device_ordinal, xla::MutableBorrowingLiteral literal) {
  TF_ASSIGN_OR_RETURN(auto* executor, GetStreamExecutor(device_ordinal));
  return transfer_manager_->TransferLiteralFromOutfeed(executor, literal);
}

StatusOr<stream_executor::StreamExecutor*>
StreamExecutorTransferOpImpl::GetStreamExecutor(int device_ordinal) {
  return tpu_platform_->ExecutorForDevice(device_ordinal);
}

}  // namespace tensorflow
