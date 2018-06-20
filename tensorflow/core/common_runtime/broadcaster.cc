/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/broadcaster.h"

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/platform/env.h"

// Set true for greater intelligibility of debug mode log messages.
#define READABLE_KEYS false

namespace tensorflow {

namespace {
// Key to be used for BufRendezvous by Broadcaster.
string BroadcastBufKey(const string& exec_key, int src_rank, int dst_rank) {
  if (READABLE_KEYS) {
    return strings::StrCat("broadcast(", exec_key, "):src(", src_rank, "):dst(",
                           dst_rank, ")");
  } else {
    // TODO(tucker): Try a denser format, e.g. a 64 or 128 bit hash.
    return strings::StrCat(exec_key, ":", src_rank, ":", dst_rank);
  }
}
}  // namespace

Broadcaster::Broadcaster(CollectiveExecutor* col_exec, const DeviceMgr* dev_mgr,
                         OpKernelContext* ctx, OpKernelContext::Params* params,
                         const CollectiveParams& col_params,
                         const string& exec_key, int64 step_id, Tensor* output)
    : col_exec_(col_exec),
      dev_mgr_(dev_mgr),
      ctx_(ctx),
      col_params_(col_params),
      exec_key_(exec_key),
      rank_(col_params.subdiv_rank[0]),
      is_source_(col_params.is_source),
      output_(output),
      done_(nullptr),
      device_(nullptr) {}

void Broadcaster::Run(StatusCallback done) {
  // The optimal data transfer choreography is going to very platform dependent.
  // That will be addressed by later improvements here or by platform-specific
  // overrides of collective broadcast. The initial version is simply
  // a binary tree that completely ignores DeviceLocality.
  done_ = std::move(done);

  // Get the device for which we're executing and look up its locality.
  status_ = dev_mgr_->LookupDevice(
      col_params_.instance.device_names[col_params_.default_rank], &device_);
  if (!status_.ok()) {
    done_(status_);
    return;
  }
  CHECK(device_);
  device_locality_ = device_->attributes().locality();

  RunTree();
}

// Binary tree parent/child relations are trivial to calculate, i.e.
// device at rank r is the parent of 2r+1 and 2r+2.  The one exception
// is if the source is not rank 0.  We treat that case as though the
// source is appended to the front of the rank ordering as well as
// continuing to occupy its current position.  Hence we calculate as
// though each device's rank is actually r+1, then subtract 1 again to
// get the descendent ranks.  If the source is not rank 0 then its
// descendants include both {0,1} and the descendents of its current
// position.  Where a non-0-rank source is a descendent of another
// device, no send to it is necessary.

/* static*/
int Broadcaster::TreeRecvFrom(const CollectiveParams& cp) {
  DCHECK_EQ(1, cp.subdiv_rank.size());
  if (cp.is_source) return -1;
  int source_rank = cp.instance.impl_details.subdiv_source_rank[0];
  int my_rank = cp.subdiv_rank[0];
  if (source_rank == 0) {
    return (my_rank - 1) / 2;
  } else {
    int predecessor_rank = (my_rank / 2) - 1;
    return (predecessor_rank < 0) ? source_rank : predecessor_rank;
  }
}

/* static */
void Broadcaster::TreeSendTo(const CollectiveParams& cp,
                             std::vector<int>* targets) {
  DCHECK_EQ(1, cp.subdiv_rank.size());
  targets->clear();
  int my_rank = cp.subdiv_rank[0];
  DCHECK_EQ(1, cp.instance.impl_details.subdiv_source_rank.size());
  int source_rank = cp.instance.impl_details.subdiv_source_rank[0];
  int successor_rank = 0;
  if (source_rank == 0) {
    successor_rank = (2 * my_rank) + 1;
  } else {
    successor_rank = (2 * (my_rank + 1));
  }
  DCHECK_NE(successor_rank, my_rank);
  if (cp.is_source && source_rank != 0) {
    // The source sends to rank 0,1 in addition to its positional
    // descendants.
    if (cp.group.group_size > 1) {
      targets->push_back(0);
    }
    if (cp.group.group_size > 2 && source_rank != 1) {
      targets->push_back(1);
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (successor_rank < cp.group.group_size && successor_rank != source_rank) {
      targets->push_back(successor_rank);
    }
    ++successor_rank;
  }
}

// Execute a tree broadcast, i.e. each non-source device receives from
// one other and sends to up-to two others.
void Broadcaster::RunTree() {
  mutex mu;               // also guards status_ while callbacks are pending
  int pending_count = 0;  // GUARDED_BY(mu)
  condition_variable all_done;
  std::vector<int> send_to_ranks;
  TreeSendTo(col_params_, &send_to_ranks);

  if (!is_source_) {
    // Begin by receiving the value.
    int recv_from_rank = TreeRecvFrom(col_params_);
    Notification note;
    DispatchRecv(recv_from_rank, output_,
                 [this, recv_from_rank, &mu, &note](const Status& s) {
                   mutex_lock l(mu);
                   status_.Update(s);
                   note.Notify();
                 });
    note.WaitForNotification();
  }

  // Then forward value to all descendent devices.
  if (status_.ok()) {
    for (int i = 0; i < send_to_ranks.size(); ++i) {
      int target_rank = send_to_ranks[i];
      {
        mutex_lock l(mu);
        ++pending_count;
      }
      DispatchSend(
          target_rank, (is_source_ ? &ctx_->input(0) : output_),
          [this, target_rank, &mu, &pending_count, &all_done](const Status& s) {
            mutex_lock l(mu);
            status_.Update(s);
            --pending_count;
            if (pending_count == 0) {
              all_done.notify_all();
            }
          });
    }
  }

  if (status_.ok() && is_source_) {
    // Meanwhile, copy input to output if we weren't lucky enough to
    // be able to reuse input as output.
    const Tensor* input = &ctx_->input(0);
    if (input != output_ &&
        (DMAHelper::base(input) != DMAHelper::base(output_))) {
      {
        mutex_lock l(mu);
        ++pending_count;
      }
      DeviceContext* op_dev_ctx = ctx_->op_device_context();
      CollectiveRemoteAccessLocal::MemCpyAsync(
          op_dev_ctx, op_dev_ctx, device_, device_, ctx_->input_alloc_attr(0),
          ctx_->output_alloc_attr(0), input, output_,
          [this, &mu, &pending_count, &all_done](const Status& s) {
            mutex_lock l(mu);
            status_.Update(s);
            --pending_count;
            if (0 == pending_count) {
              all_done.notify_all();
            }
          });
    }
  }

  // Then wait for all pending actions to complete.
  {
    mutex_lock l(mu);
    if (pending_count > 0) {
      all_done.wait(l);
    }
  }

  VLOG(2) << "return status " << status_;
  done_(status_);
}

void Broadcaster::DispatchSend(int dst_rank, const Tensor* src_tensor,
                               const StatusCallback& done) {
  string send_buf_key = BroadcastBufKey(exec_key_, rank_, dst_rank);
  VLOG(1) << "DispatchSend " << send_buf_key << " from_device "
          << device_->name();
  int dst_idx =
      col_params_.instance.impl_details.subdiv_permutations[0][dst_rank];
  col_exec_->PostToPeer(col_params_.instance.device_names[dst_idx],
                        col_params_.instance.task_names[dst_idx], send_buf_key,
                        device_, ctx_->op_device_context(),
                        ctx_->output_alloc_attr(0), src_tensor,
                        device_locality_, done);
}

void Broadcaster::DispatchRecv(int src_rank, Tensor* dst_tensor,
                               const StatusCallback& done) {
  string recv_buf_key = BroadcastBufKey(exec_key_, src_rank, rank_);
  int src_idx =
      col_params_.instance.impl_details.subdiv_permutations[0][src_rank];
  VLOG(1) << "DispatchRecv " << recv_buf_key << " from_device "
          << col_params_.instance.device_names[src_idx];
  int dst_idx = col_params_.instance.impl_details.subdiv_permutations[0][rank_];
  CHECK_EQ(col_params_.instance.device_names[dst_idx], device_->name());
  col_exec_->RecvFromPeer(col_params_.instance.device_names[src_idx],
                          col_params_.instance.task_names[src_idx],
                          col_params_.task.is_local[src_idx], recv_buf_key,
                          device_, ctx_->op_device_context(),
                          ctx_->output_alloc_attr(0), dst_tensor,
                          device_locality_, done);
}

}  // namespace tensorflow
