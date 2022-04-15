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
#include "tensorflow/core/common_runtime/all_to_all.h"

#include <utility>

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

AllToAll::AllToAll()
    : col_ctx_(nullptr), col_params_(nullptr), done_(nullptr), counter_(0) {}

StatusCallback AllToAll::CheckCounterAndCallDone() {
  return [this](const Status& s) {
    Status final_status;
    {
      mutex_lock l(mu_);
      status_.Update(s);
      ++counter_;
      // For all devices other than itself, there's a send and a receive. We
      // wait until all of them complete.
      if (counter_ < 2 * col_params_->group.group_size) {
        return;
      }
      CHECK_LE(counter_, 2 * col_params_->group.group_size);  // Crash ok.
      final_status = status_;
    }
    if (!final_status.ok()) {
      done_(final_status);
      return;
    }
    if (col_ctx_->output->SharesBufferWith(output_buffer_)) {
      done_(final_status);
    } else {
      // We are using a temp buffer. Copy to the output tensor.
      CollectiveRemoteAccessLocal::MemCpyAsync(
          col_ctx_->op_ctx->op_device_context(),
          col_ctx_->op_ctx->op_device_context(), col_ctx_->device,
          col_ctx_->device, col_ctx_->op_ctx->input_alloc_attr(0),
          col_ctx_->op_ctx->output_alloc_attr(0), &output_buffer_,
          col_ctx_->output, /*dev_to_dev_stream_index*/ 0, done_);
    }
  };
}

Status AllToAll::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
  if (col_ctx->input->dim_size(0) != col_ctx->col_params->group.group_size) {
    return errors::InvalidArgument("input to all-to-all first dimension size (",
                                   col_ctx->input->dim_size(0),
                                   ") must be the same as the group size (",
                                   col_ctx->col_params->group.group_size, ")");
  }
  DCHECK(col_ctx->dev_mgr);
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

void AllToAll::Run(StatusCallback done) {
  done_ = std::move(done);
  input_chunks_.reserve(col_params_->group.group_size);
  output_chunks_.reserve(col_params_->group.group_size);
  if (col_ctx_->input->SharesBufferWith(*col_ctx_->output)) {
    // The input is forwarded to the output, and we need to use a temp buffer.
    output_buffer_ = Tensor(
        col_ctx_->device->GetAllocator(col_ctx_->op_ctx->output_alloc_attr(0)),
        col_ctx_->output->dtype(), col_ctx_->output->shape());
  } else {
    output_buffer_ = *col_ctx_->output;
  }
  for (int i = 0; i < col_params_->group.group_size; ++i) {
    input_chunks_.push_back(col_ctx_->input->SubSlice(i));
    // Select output index based on user specified rank, if available.
    int output_index = col_params_->group.members[i].rank;
    output_chunks_.push_back(output_buffer_.SubSlice(output_index));
  }

  for (int i = 0; i < col_params_->group.group_size; ++i) {
    auto default_rank = col_params_->default_rank;
    // Issue send request from current device to all devices in group.
    DispatchSend(default_rank, i, &input_chunks_[i], CheckCounterAndCallDone());
    // Issue receive requests from all devices to current device.
    DispatchRecv(i, default_rank, &output_chunks_[i],
                 CheckCounterAndCallDone());
  }
}

void AllToAll::DispatchSend(int src_rank, int target_rank, const Tensor* tensor,
                            const StatusCallback& done) {
  string send_buf_key =
      strings::StrCat(col_ctx_->exec_key, src_rank, target_rank);
  col_ctx_->col_exec->remote_access()->PostToPeer(
      col_params_->group.members[target_rank].device.name(),
      col_params_->group.members[target_rank].task, send_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), tensor, col_ctx_->device_locality,
      col_ctx_->op_ctx->cancellation_manager(), done);
}

void AllToAll::DispatchRecv(int src_rank, int target_rank, Tensor* tensor,
                            const StatusCallback& done) {
  string recv_buf_key =
      strings::StrCat(col_ctx_->exec_key, src_rank, target_rank);
  col_ctx_->col_exec->remote_access()->RecvFromPeer(
      col_params_->group.members[src_rank].device.name(),
      col_params_->group.members[src_rank].task,
      col_params_->group.members[src_rank].is_local, recv_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), tensor, col_ctx_->device_locality,
      0, col_ctx_->op_ctx->cancellation_manager(), done);
}

namespace {
REGISTER_COLLECTIVE(AllToAll, AllToAll);
}  // namespace

}  // namespace tensorflow
