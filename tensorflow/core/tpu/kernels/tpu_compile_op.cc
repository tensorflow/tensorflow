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
#include "tensorflow/core/tpu/kernels/tpu_compile_op.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/tpu/tpu_node_context.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/tpu/compilation_result.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_common.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_options.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace tpu {

TpuCompileOp::TpuCompileOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  absl::StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> compile_op_impl =
      CompileOpImplFactory::Get()->CreateNonMlirImpl(ctx);
  OP_REQUIRES_OK(ctx, compile_op_impl.status());
  impl_ = std::move(compile_op_impl.value());
}

void TpuCompileOp::Compute(OpKernelContext* ctx) { impl_->Compute(ctx); }

TpuCompileMlirOp::TpuCompileMlirOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  absl::StatusOr<std::unique_ptr<TpuCompileOpKernelCommon>> compile_op_impl =
      CompileOpImplFactory::Get()->CreateMlirImpl(ctx);
  OP_REQUIRES_OK(ctx, compile_op_impl.status());
  impl_ = std::move(compile_op_impl.value());
}

void TpuCompileMlirOp::Compute(OpKernelContext* ctx) { impl_->Compute(ctx); }

void TpuCompileSucceededAssertOp::Compute(OpKernelContext* ctx) {
  const Tensor compilation_result = ctx->input(0);
  CompilationResultProto proto;
  absl::Status status;
  if (!proto.ParseFromString(compilation_result.scalar<tsl::tstring>()())) {
    status =
        absl::InvalidArgumentError("Unable to parse compilation result proto");
  }
  if (!status.ok() || proto.status_code() != error::Code::OK) {
    status.Update(
        absl::Status(static_cast<absl::StatusCode>(proto.status_code()),
                     proto.status_error_message()));
    LOG(WARNING) << "TPU compilation failed: " << status;
    errors::AppendToMessage(&status, "TPU compilation failed");
    if (tensorflow::internal::TpuCompilationFailureClosesChips()) {
      // At this point, if compilation fails we do not know if a task
      // is already running that expects results from this compiled
      // program to complete. So close the TPU driver to release all
      // awaiting interactions (all awaiting interaction will fail and
      // continue to fail until reinitialized).
      LOG(ERROR) << "Cloud TPU: Closing chips. TPU compilation is considered "
                    "as part of device state, and a failed compilation results "
                    "in a device reset.";

      absl::Status close_status = TpuNodeContext::CloseTpuHost();

      if (!close_status.ok()) {
        errors::AppendToMessage(&status, close_status.message());
      }
    }
    ctx->CtxFailure(status);
  }
}

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_tpu_compile_op_kernel, {
  VLOG(1) << "Register TpuCompileOp kernel.";
  REGISTER_KERNEL_BUILDER(Name("TPUCompile").Device(DEVICE_CPU), TpuCompileOp);
  REGISTER_KERNEL_BUILDER(Name("_TPUCompileMlir").Device(DEVICE_CPU),
                          TpuCompileMlirOp);
  REGISTER_KERNEL_BUILDER(Name("TPUCompileSucceededAssert").Device(DEVICE_CPU),
                          TpuCompileSucceededAssertOp);
});

}  // namespace tpu
}  // namespace tensorflow
