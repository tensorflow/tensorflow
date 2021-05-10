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
#include "tensorflow/core/common_runtime/ring_gatherer.h"

#include <stdlib.h>

#include <atomic>
#include <functional>
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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {
Status RingGatherer::InitializeCollectiveParams(CollectiveParams* col_params) {
  DCHECK_EQ(col_params->instance.type, GATHER_COLLECTIVE);
  DCHECK_EQ(col_params->instance.impl_details.collective_name, "RingGather");
  // TODO(tucker): Maybe add subdiv support.  It's only useful with
  // multiple NICS, and maybe gather performance isn't important enough.
  // For now, there must always be only a single subdiv at offset 0.
  if (!col_params->instance.impl_details.subdiv_offsets.empty() &&
      (col_params->instance.impl_details.subdiv_offsets.size() > 1 ||
       col_params->instance.impl_details.subdiv_offsets[0] != 0)) {
    return errors::InvalidArgument(
        "RingGather cannot take any subdiv offset other than 0.");
  }
  if (col_params->instance.impl_details.subdiv_offsets.empty()) {
    col_params->instance.impl_details.subdiv_offsets.push_back(0);
  }
  return RingAlg::InitializeCollectiveParams(col_params);
}

void RingGatherer::Run(StatusCallback done) {
  DCHECK(col_ctx_);
  DCHECK(col_params_);
  done_ = std::move(done);
  group_size_ = col_params_->group.group_size;
  num_subdivs_ = static_cast<int>(
      col_params_->instance.impl_details.subdiv_permutations.size());
  DCHECK_GT(num_subdivs_, 0);

  if (VLOG_IS_ON(1)) {
    string buf;
    for (int r = 0; r < col_params_->group.device_names.size(); ++r) {
      strings::StrAppend(&buf, "dev ", r, " : ",
                         col_params_->group.device_names[r], "\n");
    }
    for (int sd = 0;
         sd < col_params_->instance.impl_details.subdiv_permutations.size();
         ++sd) {
      strings::StrAppend(&buf, "\nsubdiv ", sd, " perm: ");
      for (auto x :
           col_params_->instance.impl_details.subdiv_permutations[sd]) {
        strings::StrAppend(&buf, x, ", ");
      }
    }
    VLOG(1) << "RingGatherer::Run for device " << col_ctx_->device_name
            << " default_rank " << col_params_->default_rank << "\n"
            << buf;
  }

  // Prepare to alias fields within the output.
  AllocatorAttributes attr = col_ctx_->op_ctx->output_alloc_attr(0);
  ca_.reset(MakeCollectiveAdapter(col_ctx_->output, group_size_ * num_subdivs_,
                                  col_ctx_->device->GetAllocator(attr),
                                  false /*align_chunks*/));

  // Start by copying input to the rank-specific offset of output.
  // We are running in a blockable thread and the callback can't block so
  // just wait here on the copy.
  {
    profiler::TraceMe activity("MemCpyAsync", profiler::TraceMeLevel::kInfo);
    Notification note;
    Status status;
    Tensor alias_chunk(ca_->ChunkAlias(col_params_->subdiv_rank[0]));
    CollectiveRemoteAccessLocal::MemCpyAsync(
        col_ctx_->op_ctx->op_device_context(),
        col_ctx_->op_ctx->op_device_context(), col_ctx_->device,
        col_ctx_->device, col_ctx_->op_ctx->input_alloc_attr(0),
        col_ctx_->op_ctx->output_alloc_attr(0), col_ctx_->input, &alias_chunk,
        0 /*dev_to_dev_stream_index*/, [&note, &status](const Status& s) {
          status.Update(s);
          note.Notify();
        });
    note.WaitForNotification();
    if (!status.ok()) {
      done_(status);
      return;
    }
  }
  Finish(RunAsyncParts());
}

bool RingGatherer::RunAsyncParts() {
  // This function orchestrates RingGatherer actions on behalf of a
  // single device. It is entered by a blockable thread that
  // loops within it until all actions assigned to that device
  // complete. Hence function local variables are accessible only by that
  // one thread and do not require an explicit mutex.
  rfv_.clear();
  rfv_.resize(group_size_ * num_subdivs_);
  PCQueue ready_queue;
  for (int chunk_idx = 0; chunk_idx < group_size_; ++chunk_idx) {
    for (int subdiv_idx = 0; subdiv_idx < num_subdivs_; ++subdiv_idx) {
      int rf_index = (chunk_idx * num_subdivs_) + subdiv_idx;
      InitRingField(&rfv_[rf_index], chunk_idx, subdiv_idx, rf_index);
      ready_queue.Enqueue(&rfv_[rf_index]);
    }
  }
  const DeviceBase::GpuDeviceInfo* gpu_info =
      col_ctx_->device->tensorflow_gpu_device_info();
  if (gpu_info) {
    // Wait for all currently queued events on the CPU compute stream to
    // complete before proceeding.  The previous InitRingField calls allocated
    // temp memory buffers that are not guaranteed to be valid (e.g. for RDMA
    // write) unless we do.
    profiler::TraceMe activity("WaitForQueuedEvents",
                               profiler::TraceMeLevel::kInfo);
    Notification note;
    Status s = gpu_info->default_context->ThenExecute(
        col_ctx_->device, gpu_info->stream, [&note]() { note.Notify(); });
    if (s.ok()) {
      note.WaitForNotification();
    } else {
      mutex_lock l(status_mu_);
      status_ =
          errors::Internal("Failed to dispatch ThenExecute in RingGatherer");
      return false;
    }
  }

  int field_done_count = 0;
  int send_pending_count = 0;
  int recv_pending_count = 0;
  std::atomic<bool> aborted(false);

  // Loop until all RingFields have advanced to completion.
  {
    profiler::TraceMe activity("Loop", profiler::TraceMeLevel::kInfo);
    while (field_done_count < rfv_.size()) {
      VLOG(4) << FieldState();
      // Wait for a RingField to appear in the ready_queue.
      RingField* rf = ready_queue.Dequeue();
      // Advance the RingField to its next action and execute, repeating
      // until either an async action has been started or the RingField
      // is done.
      bool dispatched = false;  // true if async action was initiated
      do {
        if (aborted) {
          // Requeue this RingField to be counted off below.
          ready_queue.Enqueue(rf);
          break;
        }
        switch (rf->action) {
          case RF_INIT:
            if (rf->do_recv) {
              rf->action = RF_RECV;
              auto requeue = [this, rf, &ready_queue, &aborted](Status s) {
                if (!s.ok()) {
                  aborted = true;
                  StartAbort(s);
                }
                ready_queue.Enqueue(rf);
              };
              DispatchRecv(rf, requeue);
              dispatched = true;
              ++recv_pending_count;
            } else {
              rf->action = RF_SEND_READY;
            }
            break;
          case RF_RECV:
            DCHECK_GT(recv_pending_count, 0);
            --recv_pending_count;
            rf->action = RF_SEND_READY;
            break;
          case RF_REDUCE:
            // Never used for Gather, so just fall through.
            TF_FALLTHROUGH_INTENDED;
          case RF_FINALIZE:
            // Never used for Gather, so just fall through.
            TF_FALLTHROUGH_INTENDED;
          case RF_SEND_READY:
            if (rf->do_send) {
              rf->action = RF_SEND;
              auto send_complete = [this, rf, &ready_queue,
                                    &aborted](Status s) {
                if (!s.ok()) {
                  aborted = true;
                  StartAbort(s);
                }
                ready_queue.Enqueue(rf);
              };
              DispatchSend(rf, send_complete);
              dispatched = true;
              ++send_pending_count;
            } else {
              rf->action = RF_DONE;
            }
            break;
          case RF_SEND:
            DCHECK_GT(send_pending_count, 0);
            --send_pending_count;
            rf->action = RF_DONE;
            break;
          case RF_DONE:
            break;
        }
        if (rf->action == RF_DONE) {
          // There's only one pass.
          ++field_done_count;
          break;  // from do while(!dispatched)
        }
      } while (!dispatched);
      if (aborted) break;
    }  // while (field_done_count < number of fields)

    if (aborted) {
      // All of the pending data actions should be aborted; field the
      // callbacks and clear the queue before quitting.
      while ((send_pending_count > 0) || (recv_pending_count > 0)) {
        RingField* rf = ready_queue.Dequeue();
        switch (rf->action) {
          case RF_RECV:
            --recv_pending_count;
            break;
          case RF_SEND:
            --send_pending_count;
            break;
          default: {
          }  // Ignore any other actions
        }
      }
    }
  }

  DCHECK_EQ(send_pending_count, 0);
  DCHECK_EQ(recv_pending_count, 0);

  VLOG(2) << this << " device=" << col_ctx_->device_name << " finish;"
          << " final value " << TensorDebugString(ca_->Value());
  return !aborted;
}

namespace {
REGISTER_COLLECTIVE(RingGather, RingGatherer);
}  // namespace

}  // namespace tensorflow
