/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/collective_nccl_reducer.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

void NcclReducer::Run(StatusCallback done) {
  Tensor group_size;
  std::unique_ptr<Notification> group_size_ready;
  Status group_size_status;
  std::unique_ptr<Notification> nccl_done;
  if (col_params_->final_op) {
    group_size_ready = std::make_unique<Notification>();
    // Create an on-device scalar value from group_size_.
    // TODO(ayushd, tucker): avoid this copy by either reusing across
    // invocations or providing the scalar to the kernel in host memory.
    Tensor group_size_val;
    switch (col_ctx_->output->dtype()) {
      case DT_HALF:
        group_size_val =
            Tensor(static_cast<Eigen::half>(col_params_->group.group_size));
        break;
      case DT_FLOAT:
        group_size_val =
            Tensor(static_cast<float>(col_params_->group.group_size));
        break;
      case DT_DOUBLE:
        group_size_val =
            Tensor(static_cast<double>(col_params_->group.group_size));
        break;
      case DT_INT32:
        group_size_val =
            Tensor(static_cast<int32>(col_params_->group.group_size));
        break;
      case DT_INT64:
        group_size_val =
            Tensor(static_cast<int64_t>(col_params_->group.group_size));
        break;
      default:
        done(errors::Internal("Unsupported type ",
                              DataTypeString(col_ctx_->output->dtype())));
        return;
    }
    group_size = Tensor(
        col_ctx_->device->GetAllocator(col_ctx_->op_ctx->input_alloc_attr(0)),
        col_ctx_->output->dtype(), TensorShape({}));
    DeviceContext* op_dev_ctx = col_ctx_->op_ctx->op_device_context();
    // Enqueue copy on gpu stream.
    Notification* copy_note = group_size_ready.get();
    op_dev_ctx->CopyCPUTensorToDevice(
        &group_size_val, col_ctx_->device, &group_size,
        [copy_note, &group_size_status](const Status& s) {
          group_size_status = s;
          copy_note->Notify();
        });
    nccl_done = std::make_unique<Notification>();
  }

  Status nccl_status;
  // If no final_op, then the NCCL callback is just `done`.  Otherwise we notify
  // `nccl_done` so that we can then perform `final_op`.
  StatusCallback done_callback;
  if (col_params_->final_op) {
    Notification* nccl_note = nccl_done.get();
    done_callback = [nccl_note, &nccl_status](const Status& s) {
      nccl_status = s;
      nccl_note->Notify();
    };
  } else {
    done_callback = std::move(done);
  }
  // Hold a ref to col_params for the rest of this function.
  col_params_->Ref();
  core::ScopedUnref unref(col_params_);
  col_ctx_->nccl_communicator->Enqueue(col_ctx_, std::move(done_callback));

  // If no final_op, then this OpKernel is non-blocking.
  if (!col_params_->final_op) {
    return;
  }

  // Wait for nccl op and group_size copy to succeed, then do final_op.  This
  // kernel needs to wait for both notifications because they execute on
  // different GPU streams with no ordering guarantees between them.
  // TODO(b/80529858): make this entirely non-blocking by getting rid of the
  // waits below and calling final op from the nccl kernel's DoneCallback.
  {
    profiler::TraceMe activity("Nccl", profiler::TraceMeLevel::kInfo);
    nccl_done->WaitForNotification();
  }
  {
    profiler::TraceMe activity("GroupSizeCopy", profiler::TraceMeLevel::kInfo);
    group_size_ready->WaitForNotification();
  }
  Status final_status =
      group_size_status.ok() ? nccl_status : group_size_status;
  if (final_status.ok()) {
    final_status = collective_util::ComputeBinOp(
        col_ctx_->op_ctx, col_ctx_->op_params, col_ctx_->device,
        col_params_->final_op, col_ctx_->output, &group_size);
  }
  done(final_status);
}

REGISTER_COLLECTIVE(NcclReduce, NcclReducer);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
