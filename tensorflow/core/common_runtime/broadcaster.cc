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
string BroadcastBufKey(const string& exec_key, int subdiv, int src_rank,
                       int dst_rank) {
  if (READABLE_KEYS) {
    return strings::StrCat("broadcast(", exec_key, "):subdiv(", subdiv,
                           "):src(", src_rank, "):dst(", dst_rank, ")");
  } else {
    // TODO(tucker): Try a denser format, e.g. a 64 or 128 bit hash.
    return strings::StrCat(exec_key, ":", subdiv, ":", src_rank, ":", dst_rank);
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
int Broadcaster::TreeRecvFrom(const CollectiveParams& cp, int subdiv) {
  DCHECK_LT(subdiv, static_cast<int>(cp.subdiv_rank.size()));
  int my_rank = cp.subdiv_rank[subdiv];
  if (-1 == my_rank) return -1;

  const auto& impl = cp.instance.impl_details;
  DCHECK_LT(subdiv, static_cast<int>(impl.subdiv_source_rank.size()));
  int source_rank = impl.subdiv_source_rank[subdiv];
  if (my_rank == source_rank) return -1;
  if (source_rank == 0) {
    return (my_rank - 1) / 2;
  } else {
    int predecessor_rank = (my_rank / 2) - 1;
    return (predecessor_rank < 0) ? source_rank : predecessor_rank;
  }
}

/* static */
void Broadcaster::TreeSendTo(const CollectiveParams& cp, int subdiv,
                             std::vector<int>* targets) {
  DCHECK_LT(subdiv, static_cast<int>(cp.subdiv_rank.size()));
  int my_rank = cp.subdiv_rank[subdiv];
  if (-1 == my_rank) return;

  const auto& impl = cp.instance.impl_details;
  DCHECK_LT(subdiv, static_cast<int>(impl.subdiv_source_rank.size()));
  int source_rank = impl.subdiv_source_rank[subdiv];

  int group_size = 0;
  for (int i = 0; i < impl.subdiv_permutations[subdiv].size(); i++) {
    if (impl.subdiv_permutations[subdiv][i] >= 0) {
      group_size++;
    }
  }

  targets->clear();
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
    if (group_size > 1) {
      targets->push_back(0);
    }
    if (group_size > 2 && source_rank != 1) {
      targets->push_back(1);
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (successor_rank < group_size && successor_rank != source_rank) {
      targets->push_back(successor_rank);
    }
    ++successor_rank;
  }
}

// Executes a hierarchical tree broadcast.
// Each subdiv is a broadcast between a subset of the devices.
// If there is only one task, there is one subdiv comprising a broadcast between
// all devices belonging to the task.
// If there are n tasks, n>1, then there are n+1 subdivs.  In the first (global)
// subdiv, one device from each task participates in a binary tree broadcast.
// Each task receives a copy of the tensor on one device via this broadcast.
// Subsequent subdivs correspond to intra-task broadcasts.  Subdiv i+1
// corresponds to broadcast between all devices on task i.  Thus, each task
// participates in at most 2 subdivs.
void Broadcaster::RunTree() {
  int num_subdivs = static_cast<int>(col_params_.subdiv_rank.size());
  // TODO(ayushd): this is easily improved when a node participates in both
  // first and second subdivision.  It would first send to its descendents in
  // the first subdiv, then wait until all pending ops are finished before
  // sending to descendents in second subdiv.  A better implementation would
  // collapse the two send blocks.
  for (int si = 0; si < num_subdivs; si++) {
    int my_rank = col_params_.subdiv_rank[si];
    // If rank is -1, this device does not participate in this subdiv.
    if (-1 == my_rank) continue;
    int source_rank = col_params_.instance.impl_details.subdiv_source_rank[si];
    if (VLOG_IS_ON(1)) {
      string subdiv_buf;
      for (int r : col_params_.instance.impl_details.subdiv_permutations[si]) {
        strings::StrAppend(&subdiv_buf, r, ",");
      }
      VLOG(1) << "Running Broadcast tree device=" << device_->name()
              << " subdiv=" << si << " perm=" << subdiv_buf
              << " my_rank=" << my_rank << " source_rank=" << source_rank;
    }

    mutex mu;               // also guards status_ while callbacks are pending
    int pending_count = 0;  // GUARDED_BY(mu)
    condition_variable all_done;

    if (my_rank >= 0 && my_rank != source_rank) {
      // Begin by receiving the value.
      int recv_from_rank = TreeRecvFrom(col_params_, si);
      Notification note;
      DispatchRecv(si, recv_from_rank, my_rank, output_,
                   [this, &mu, &note](const Status& s) {
                     mutex_lock l(mu);
                     status_.Update(s);
                     note.Notify();
                   });
      note.WaitForNotification();
    }

    // Then forward value to all descendent devices.
    if (my_rank >= 0 && status_.ok()) {
      std::vector<int> send_to_ranks;
      TreeSendTo(col_params_, si, &send_to_ranks);
      for (int i = 0; i < send_to_ranks.size(); ++i) {
        int target_rank = send_to_ranks[i];
        {
          mutex_lock l(mu);
          ++pending_count;
        }
        DispatchSend(si, target_rank, my_rank,
                     (is_source_ ? &ctx_->input(0) : output_),
                     [this, &mu, &pending_count, &all_done](const Status& s) {
                       mutex_lock l(mu);
                       status_.Update(s);
                       --pending_count;
                       if (pending_count == 0) {
                         all_done.notify_all();
                       }
                     });
      }
    }

    // For the original source device, we copy input to output if they are
    // different.
    // If there is only 1 subdiv, we do this in that subdiv.  If there is more
    // than 1 subdiv, then the original source device will participate in 2
    // subdivs - the global inter-task broadcast and one local intra-task
    // broadcast.  In this case, we perform the copy in the second subdiv for
    // this device.
    if (status_.ok() && is_source_ && (1 == num_subdivs || 0 != si)) {
      VLOG(2) << "copying input to output for device=" << device_->name()
              << " subdiv=" << si;
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
            ctx_->output_alloc_attr(0), input, output_, 0, /*stream_index*/
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
  }
  VLOG(2) << "device=" << device_->name() << " return status " << status_;
  done_(status_);
}

void Broadcaster::DispatchSend(int subdiv, int dst_rank, int src_rank,
                               const Tensor* src_tensor,
                               const StatusCallback& done) {
  string send_buf_key = BroadcastBufKey(exec_key_, subdiv, src_rank, dst_rank);
  int dst_idx =
      col_params_.instance.impl_details.subdiv_permutations[subdiv][dst_rank];
  VLOG(1) << "DispatchSend " << send_buf_key << " from_device "
          << device_->name() << " to_device "
          << col_params_.instance.device_names[dst_idx] << " subdiv=" << subdiv
          << " dst_rank=" << dst_rank << " dst_idx=" << dst_idx;
  col_exec_->PostToPeer(col_params_.instance.device_names[dst_idx],
                        col_params_.instance.task_names[dst_idx], send_buf_key,
                        device_, ctx_->op_device_context(),
                        ctx_->output_alloc_attr(0), src_tensor,
                        device_locality_, done);
}

void Broadcaster::DispatchRecv(int subdiv, int src_rank, int dst_rank,
                               Tensor* dst_tensor, const StatusCallback& done) {
  string recv_buf_key = BroadcastBufKey(exec_key_, subdiv, src_rank, dst_rank);
  int src_idx =
      col_params_.instance.impl_details.subdiv_permutations[subdiv][src_rank];
  VLOG(1) << "DispatchRecv " << recv_buf_key << " from_device "
          << col_params_.instance.device_names[src_idx] << " to_device "
          << device_->name() << " subdiv=" << subdiv << " src_rank=" << src_rank
          << " src_idx=" << src_idx;
  col_exec_->RecvFromPeer(col_params_.instance.device_names[src_idx],
                          col_params_.instance.task_names[src_idx],
                          col_params_.task.is_local[src_idx], recv_buf_key,
                          device_, ctx_->op_device_context(),
                          ctx_->output_alloc_attr(0), dst_tensor,
                          device_locality_, 0 /*stream_index*/, done);
}

}  // namespace tensorflow
