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
#include "tensorflow/core/common_runtime/permuter.h"

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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Permuter::Permuter()
    : col_ctx_(nullptr), col_params_(nullptr), done_(nullptr), counter_(0) {}

StatusCallback Permuter::CheckCounterAndCallDone() {
  return [this](const absl::Status& s) {
    mu_.lock();
    status_.Update(s);
    int counter = ++counter_;
    absl::Status status = status_;
    mu_.unlock();
    if (counter == 2) done_(status);
  };
}

absl::Status Permuter::InitializeCollectiveContext(
    std::shared_ptr<CollectiveContext> col_ctx) {
  DCHECK(col_ctx->dev_mgr);
  col_ctx_ = col_ctx;
  col_params_ = col_ctx->col_params.get();
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

void Permuter::Run(StatusCallback done) {
  if (col_params_->instance.permutation.size() !=
      col_params_->instance.devices.size()) {
    done(errors::Internal("Permutation must be the same size as devices"));
  }
  done_ = std::move(done);
  DispatchSend(col_params_->default_rank,
               col_params_->instance.permutation[col_params_->default_rank],
               col_ctx_->input, CheckCounterAndCallDone());
  for (int i = 0; i < col_params_->instance.permutation.size(); ++i) {
    if (col_params_->default_rank == col_params_->instance.permutation[i]) {
      DispatchRecv(i, col_params_->instance.permutation[i], col_ctx_->output,
                   CheckCounterAndCallDone());
    }
  }
}

void Permuter::DispatchSend(int src_rank, int target_rank, const Tensor* tensor,
                            const StatusCallback& done) {
  string send_buf_key =
      strings::StrCat(col_ctx_->exec_key, src_rank, target_rank);
  VLOG(1) << "DispatchSend " << send_buf_key << " from_device "
          << col_ctx_->device_name << " to_device "
          << col_params_->instance.devices[target_rank]
          << " target_rank=" << target_rank << " src_rank=" << src_rank;
  col_ctx_->col_exec->remote_access()->PostToPeer(
      col_params_->instance.devices[target_rank],
      col_params_->group.members[target_rank].task, send_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), tensor, col_ctx_->device_locality,
      col_ctx_->op_ctx->cancellation_manager(), done);
}

void Permuter::DispatchRecv(int src_rank, int target_rank, Tensor* tensor,
                            const StatusCallback& done) {
  string recv_buf_key =
      strings::StrCat(col_ctx_->exec_key, src_rank, target_rank);
  VLOG(1) << "DispatchRecv " << recv_buf_key << " to_device "
          << col_ctx_->device_name << " from_device "
          << col_params_->instance.devices[src_rank]
          << " target_rank=" << target_rank << " src_rank=" << src_rank;
  col_ctx_->col_exec->remote_access()->RecvFromPeer(
      col_params_->instance.devices[src_rank],
      col_params_->group.members[src_rank].task,
      col_params_->group.members[src_rank].is_local, recv_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), tensor, col_ctx_->device_locality,
      0, col_ctx_->op_ctx->cancellation_manager(), done);
}
namespace {
REGISTER_COLLECTIVE(Permute, Permuter);
}  // namespace

}  // namespace tensorflow
