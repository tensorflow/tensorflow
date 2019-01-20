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
#include "tensorflow/core/common_runtime/ring_reducer.h"

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

// Set true for greater intelligibility of debug mode log messages.
#define READABLE_KEYS false
// RingReduce algorithm exchanges chunks of tensor between devices.  The chunk
// size depends on the number of subdivisions specified in the algorithm.  If
// the user does not specify the number of subdivisions, we infer the number
// dynamically so that the resulting chunk size does not exceed
// kMaxChunkSizeBytes, empirically set at 4 MiB.
constexpr size_t kMaxChunkSizeBytes = (4 * 1024 * 1024);
// kMaxSubdivsPerDev is used to give an upper bound on the number of
// subdivisions dynamically generated.  A reasonable value would be a small
// multiple of the number of NICs adjacent to each device.
constexpr int kMaxSubdivsPerDevice = 2;

namespace tensorflow {
namespace {
// Each CollectiveOp implementation is free to define its own
// BufRendezvous key format.  This function produces the key used by
// RingReducer.
string RingReduceBufKey(const string& exec_key, int pass, int section,
                        int source_rank) {
  if (READABLE_KEYS) {
    return strings::StrCat("rred(", exec_key, "):pass(", pass, "):section(",
                           section, "):srcrank(", source_rank, ")");
  } else {
    // TODO(b/78352018): Try out some kind of denser encoding, e.g. 128 bit
    // hash.
    return strings::StrCat(exec_key, ":", pass, ":", section, ":", source_rank);
  }
}

}  // namespace

void RingReducer::PCQueue::Enqueue(RingField* rf) {
  mutex_lock l(pcq_mu_);
  deque_.push_back(rf);
  if (waiter_count_ > 0) {
    cv_.notify_one();
  }
}

RingReducer::RingField* RingReducer::PCQueue::Dequeue() {
  mutex_lock l(pcq_mu_);
  if (deque_.empty()) {
    ++waiter_count_;
    while (deque_.empty()) {
      cv_.wait(l);
    }
    --waiter_count_;
  }
  RingField* rf = deque_.front();
  deque_.pop_front();
  return rf;
}

RingReducer::RingReducer()
    : col_ctx_(nullptr),
      col_params_(nullptr),
      done_(nullptr),
      group_size_(-1),
      num_subdivs_(-1) {}

RingReducer::~RingReducer() { group_size_tensor_ready_.WaitForNotification(); }

Status GenerateSubdivsInCollectiveParams(CollectiveParams* col_params) {
  if (col_params->instance.shape.num_elements() == 0) {
    return errors::Internal("shape in CollectiveParams should be non-empty");
  }
  const int kAvgDevPerTask =
      col_params->group.group_size / col_params->group.num_tasks;
  const int kMaxNumSubdivs = kMaxSubdivsPerDevice * kAvgDevPerTask;
  if (kMaxNumSubdivs <= 0) {
    return errors::Internal("Unexpected kMaxNumSubdivs ", kMaxNumSubdivs,
                            " in RingReducer");
  }
  // NOTE(ayushd): If no subdiv_offsets have been specified, dynamically add
  // as many offsets as needed so that the size of tensor chunks <=
  // kMaxChunkSizeBytes.  Empirically, chunks that are too small or too large
  // lead to worse performance.
  int num_subdivs = 0;
  const size_t tensor_size = col_params->instance.shape.num_elements() *
                             DataTypeSize(col_params->instance.data_type);
  size_t chunk_size;
  do {
    ++num_subdivs;
    int num_chunks = col_params->group.group_size * num_subdivs;
    chunk_size = tensor_size / num_chunks;
    VLOG(2) << "num_subdivs " << num_subdivs << " num_chunks " << num_chunks
            << " chunk_size " << chunk_size;
  } while (chunk_size > kMaxChunkSizeBytes && num_subdivs < kMaxNumSubdivs);
  if (num_subdivs <= 0) {
    return errors::Internal("Unexpected num_subdivs ", num_subdivs,
                            " in RingReducer");
  }

  int subdiv_stride = kAvgDevPerTask / num_subdivs;
  if (subdiv_stride == 0) subdiv_stride = 1;
  col_params->instance.impl_details.subdiv_offsets.reserve(num_subdivs);
  for (int sdi = 0; sdi < num_subdivs; ++sdi) {
    int subdiv_offset = subdiv_stride * sdi;
    if (sdi % 2 == 1) subdiv_offset *= -1;
    col_params->instance.impl_details.subdiv_offsets.push_back(subdiv_offset);
  }

  if (VLOG_IS_ON(2)) {
    string subdiv_buf;
    for (const int subdiv_offset :
         col_params->instance.impl_details.subdiv_offsets) {
      strings::StrAppend(&subdiv_buf, " ", subdiv_offset);
    }
    VLOG(2) << "Dynamically generated " << num_subdivs
            << " subdiv_offsets:" << subdiv_buf << " tensor_size "
            << tensor_size << " chunk_size " << chunk_size;
  }

  return Status::OK();
}

Status RingReducer::InitializeCollectiveParams(CollectiveParams* col_params) {
  // TODO(b/113171733): change CHECKs to return errors.
  CHECK_EQ(col_params->instance.type, REDUCTION_COLLECTIVE);
  CHECK_EQ(col_params->instance.impl_details.collective_name, "RingReduce");
  const string& device_name =
      col_params->instance.device_names[col_params->default_rank];
  // Each subdiv permutation is a ring formed by rotating each
  // single-task subsequence of devices by an offset.  This makes most
  // sense when each task has the same number of devices but we can't
  // depend on that being the case so we'll compute something that
  // works in any case.

  // Start by counting the devices in each task.
  // Precondition: device_names must be sorted so that all devices in
  // the same task are adjacent.
  VLOG(2) << "Sorted task names: "
          << str_util::Join(col_params->instance.task_names, ", ");
  std::vector<int> dev_per_task;
  const string* prior_task_name = &col_params->instance.task_names[0];
  int dev_count = 1;
  for (int di = 1; di < col_params->group.group_size; ++di) {
    if (col_params->instance.task_names[di] != *prior_task_name) {
      dev_per_task.push_back(dev_count);
      dev_count = 1;
      prior_task_name = &col_params->instance.task_names[di];
    } else {
      ++dev_count;
    }
  }
  dev_per_task.push_back(dev_count);
  CHECK_EQ(col_params->group.num_tasks, dev_per_task.size());

  if (col_params->instance.impl_details.subdiv_offsets.empty()) {
    TF_RETURN_IF_ERROR(GenerateSubdivsInCollectiveParams(col_params));
  }

  // Generate a ring permutation for requested offset.
  VLOG(2) << "Setting up perms for col_params " << col_params
          << " subdiv_permutations "
          << &col_params->instance.impl_details.subdiv_permutations;
  col_params->instance.impl_details.subdiv_permutations.resize(
      col_params->instance.impl_details.subdiv_offsets.size());
  col_params->subdiv_rank.resize(
      col_params->instance.impl_details.subdiv_offsets.size(), -1);
  for (int sdi = 0;
       sdi < col_params->instance.impl_details.subdiv_offsets.size(); ++sdi) {
    std::vector<int>& perm =
        col_params->instance.impl_details.subdiv_permutations[sdi];
    CHECK_EQ(perm.size(), 0);
    int offset = col_params->instance.impl_details.subdiv_offsets[sdi];
    // A negative subdivision offset is interpreted as follows:
    //  1. Reverse the local device ordering.
    //  2. Begin the subdivision at abs(offset) in the reversed ordering.
    bool reverse = false;
    if (offset < 0) {
      offset = abs(offset);
      reverse = true;
    }
    int prior_dev_count = 0;  // sum over prior worker device counts
    for (int ti = 0; ti < col_params->group.num_tasks; ++ti) {
      for (int di = 0; di < dev_per_task[ti]; ++di) {
        int di_offset = (di + offset) % dev_per_task[ti];
        int offset_di =
            reverse ? (dev_per_task[ti] - (di_offset + 1)) : di_offset;
        // Device index in global subdivision permutation.
        int permuted_di = prior_dev_count + offset_di;
        int rank = static_cast<int>(perm.size());
        perm.push_back(permuted_di);
        if (col_params->instance.device_names[permuted_di] == device_name) {
          CHECK_EQ(permuted_di, col_params->default_rank);
          col_params->subdiv_rank[sdi] = rank;
        }
      }
      prior_dev_count += dev_per_task[ti];
    }
    CHECK_EQ(col_params->group.group_size, perm.size());
  }

  VLOG(2) << collective_util::SubdivPermDebugString(*col_params);
  return Status::OK();
}

Status RingReducer::InitializeCollectiveContext(CollectiveContext* col_ctx) {
  CHECK(col_ctx->dev_mgr);
  col_ctx_ = col_ctx;
  col_params_ = &col_ctx->col_params;
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

void RingReducer::Run(StatusCallback done) {
  CHECK(col_ctx_);
  CHECK(col_params_);
  done_ = std::move(done);
  group_size_ = col_params_->group.group_size;
  num_subdivs_ = static_cast<int>(
      col_params_->instance.impl_details.subdiv_permutations.size());
  CHECK_GT(num_subdivs_, 0);

  if (VLOG_IS_ON(1)) {
    string buf;
    for (int r = 0; r < col_params_->instance.device_names.size(); ++r) {
      strings::StrAppend(&buf, "dev ", r, " : ",
                         col_params_->instance.device_names[r], "\n");
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
    VLOG(1) << "RingReducer::Run for device " << col_ctx_->device_name
            << " default_rank " << col_params_->default_rank << "\n"
            << buf;
  }

  // Start by copying input to output if they're not already the same, i.e. if
  // we're not computing in-place on the input tensor.
  if ((col_ctx_->input != col_ctx_->output) &&
      (DMAHelper::base(col_ctx_->input) != DMAHelper::base(col_ctx_->output))) {
    // We are running in a blockable thread and the callback can't block so
    // just wait here on the copy.
    Notification note;
    Status status;
    CollectiveRemoteAccessLocal::MemCpyAsync(
        col_ctx_->op_ctx->input_device_context(0),
        col_ctx_->op_ctx->op_device_context(), col_ctx_->device,
        col_ctx_->device, col_ctx_->op_ctx->input_alloc_attr(0),
        col_ctx_->op_ctx->output_alloc_attr(0), col_ctx_->input,
        col_ctx_->output, 0 /*dev_to_dev_stream_index*/,
        [&note, &status](const Status& s) {
          status.Update(s);
          note.Notify();
        });
    note.WaitForNotification();
    if (!status.ok()) {
      done_(status);
      return;
    }
  }
  ContinueAfterInputCopy();
}

string RingReducer::TensorDebugString(const Tensor& tensor) {
  const DeviceBase::GpuDeviceInfo* gpu_device_info =
      col_ctx_->op_ctx->device()->tensorflow_gpu_device_info();
  if (gpu_device_info) {
    Tensor cpu_tensor(tensor.dtype(), tensor.shape());
    Notification note;
    gpu_device_info->default_context->CopyDeviceTensorToCPU(
        &tensor, "" /*tensor_name*/, col_ctx_->device, &cpu_tensor,
        [&note](const Status& s) {
          CHECK(s.ok());
          note.Notify();
        });
    note.WaitForNotification();
    return cpu_tensor.SummarizeValue(64);
  } else {
    return tensor.SummarizeValue(64);
  }
}

// Note that this function is blocking and must not run in any thread
// which cannot be blocked.
void RingReducer::ContinueAfterInputCopy() {
  AllocatorAttributes attr = col_ctx_->op_ctx->output_alloc_attr(0);
  ca_.reset(MakeCollectiveAdapter(col_ctx_->output, group_size_ * num_subdivs_,
                                  col_ctx_->device->GetAllocator(attr)));

  if (col_params_->final_op) {
    // Create an on-device scalar value from group_size_ that may be needed
    // later.
    // TODO(tucker): Cache and reuse across invocations? Or maybe the scalar
    // can be provided to the kernel in host memory?
    Tensor group_size_val = ca_->Scalar(group_size_);
    if (col_params_->group.device_type != "CPU") {
      group_size_tensor_ = ca_->Scalar(col_ctx_->device->GetAllocator(
          col_ctx_->op_ctx->input_alloc_attr(0)));
      DeviceContext* op_dev_ctx = col_ctx_->op_ctx->op_device_context();
      op_dev_ctx->CopyCPUTensorToDevice(&group_size_val, col_ctx_->device,
                                        &group_size_tensor_,
                                        [this](const Status& s) {
                                          if (!s.ok()) {
                                            StartAbort(s);
                                          }
                                          group_size_tensor_ready_.Notify();
                                        });
    } else {
      group_size_tensor_ = group_size_val;
      group_size_tensor_ready_.Notify();
    }
  } else {
    // Value won't be used, so no need to initialize.
    group_size_tensor_ready_.Notify();
  }
  Finish(RunAsyncParts());
}

void RingReducer::StartAbort(const Status& s) {
  // In abort mode we stop issuing additional ProvideBuf
  // and ConsumeBuf calls, but we need to wait for all of the
  // outstanding callbacks to be invoked before quitting.
  bool abort_started = false;
  {
    mutex_lock l(status_mu_);
    if (status_.ok()) {
      LOG(ERROR) << "Aborting RingReduce with " << s;
      abort_started = true;
      status_.Update(s);
    }
  }
  // If this is the initial entry to abort mode then invoke StartAbort
  // on the CollectiveExecutor that invoked us.  That should start
  // cancellation on all of the outstanding CollectiveRemoteAccess
  // actions.
  if (abort_started) {
    col_ctx_->col_exec->StartAbort(s);
  }
}

void RingReducer::Finish(bool ok) {
  if (ok) {
    // Recover the output from the adaptor.
    ca_->ConsumeFinalValue(col_ctx_->output);
  }
  Status s;
  {
    mutex_lock l(status_mu_);
    s = status_;
  }
  rfv_.clear();  // Give up Refs on output tensor.
  done_(s);
}

// At the beginning of the algorithm initialize a RingField struct for
// every independent field of the tensor.
void RingReducer::InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                                int field_idx) {
  // Note on field indexing: There are group_size_ devices in the
  // instance, implying the same number of chunks per tensor, where a
  // chunk is the unit of data transferred in a time step.  However, if
  // a device can simultaneously send data by 2 or more independent
  // channels we can speed up the transfer by subdividing chunks and
  // processing multiple subdivisions at once.  So the actual number
  // of RingFields is group_size_ * num_subdivs_.
  DCHECK_EQ(field_idx, (chunk_idx * num_subdivs_) + subdiv_idx);
  rf->chunk_idx = chunk_idx;
  rf->subdiv_idx = subdiv_idx;
  rf->sc_idx = field_idx;
  rf->rank = col_params_->subdiv_rank[subdiv_idx];
  rf->second_pass = false;
  rf->action = RF_INIT;
  // Recv from the device with preceding rank within the subdivision.
  int recv_from_rank = (rf->rank + (group_size_ - 1)) % group_size_;
  int send_to_rank = (rf->rank + 1) % group_size_;
  rf->recv_dev_idx = col_params_->instance.impl_details
                         .subdiv_permutations[subdiv_idx][recv_from_rank];
  int send_dev_idx = col_params_->instance.impl_details
                         .subdiv_permutations[subdiv_idx][send_to_rank];
  rf->recv_is_remote = !col_params_->task.is_local[rf->recv_dev_idx];
  rf->send_is_remote = !col_params_->task.is_local[send_dev_idx];
  if (ca_->ChunkBytes(rf->sc_idx) > 0) {
    // In pass 0 we skip Recv when rank = chunk_idx
    rf->do_recv = (rf->chunk_idx != rf->rank);
    // In pass 0 we skip Send when rank = chunk_idx-1
    rf->do_send =
        (rf->rank != ((rf->chunk_idx + (group_size_ - 1)) % group_size_));
  }
  rf->is_final =
      (rf->rank == ((rf->chunk_idx + (group_size_ - 1)) % group_size_));
  if (rf->do_send || rf->do_recv) {
    rf->chunk = ca_->ChunkAlias(rf->sc_idx);
    CHECK(rf->chunk.IsAligned()) << rf->DebugString();
  }
  if (rf->do_recv) {
    rf->tmp_chunk = ca_->TempChunk(rf->sc_idx);
    CHECK(rf->tmp_chunk.IsAligned()) << rf->DebugString();
  }
  VLOG(2) << this << " InitRingField " << rf->DebugString() << " chunk "
          << ca_->TBounds(rf->chunk);
}

// When a RingField transitions from first to second recompute the
// do_send and do_recv values.
void RingReducer::AdvanceToSecondPass(RingField* rf) {
  VLOG(3) << "IncrRingField old value " << rf->DebugString();
  CHECK(!rf->second_pass);
  rf->second_pass = true;
  rf->action = RF_INIT;
  if (ca_->ChunkBytes(rf->sc_idx) > 0) {
    // In pass 1 the send/no-send boundary moves down 1 place.
    rf->do_recv =
        (rf->rank != ((rf->chunk_idx + (group_size_ - 1)) % group_size_));
    rf->do_send =
        (rf->rank != ((rf->chunk_idx + (group_size_ - 2)) % group_size_));
  }
  rf->is_final =
      (rf->rank == ((rf->chunk_idx + (group_size_ - 2)) % group_size_));
  VLOG(3) << "IncrRingField new value " << rf->DebugString();
}

string RingReducer::RingField::DebugString() const {
  string rv = strings::StrCat("RingField rank=", rank, " chunk_idx=", chunk_idx,
                              " subdiv=", subdiv_idx, " sc_idx=", sc_idx,
                              " action=", action);
  strings::StrAppend(&rv, " pass=", second_pass);
  strings::StrAppend(&rv, " do_send=", do_send, " do_recv=", do_recv,
                     " is_final=", is_final, " recv_is_remote=", recv_is_remote,
                     " recv_dev_idx=", recv_dev_idx, " sc_idx=", sc_idx);
  return rv;
}

void RingReducer::DispatchSend(RingField* rf, const StatusCallback& done) {
  CHECK(rf->do_send);
  string send_buf_key = RingReduceBufKey(col_ctx_->exec_key, rf->second_pass,
                                         rf->sc_idx, rf->rank);
  VLOG(3) << "DispatchSend rank=" << col_params_->default_rank << " send key "
          << send_buf_key << " chunk " << ca_->TBounds(rf->chunk) << " sc_idx "
          << rf->sc_idx;
  int send_to_rank = (rf->rank + 1) % group_size_;
  int send_to_dev_idx = col_params_->instance.impl_details
                            .subdiv_permutations[rf->subdiv_idx][send_to_rank];
  col_ctx_->col_exec->PostToPeer(
      col_params_->instance.device_names[send_to_dev_idx],
      col_params_->instance.task_names[send_to_dev_idx], send_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), &rf->chunk,
      col_ctx_->device_locality, done);
}

void RingReducer::DispatchRecv(RingField* rf, const StatusCallback& done) {
  CHECK(rf->do_recv);
  string recv_buf_key =
      RingReduceBufKey(col_ctx_->exec_key, rf->second_pass, rf->sc_idx,
                       (rf->rank + (group_size_ - 1)) % group_size_);
  VLOG(3) << "DispatchRecv rank=" << col_params_->default_rank << " recv key "
          << recv_buf_key << " chunk " << ca_->TBounds(rf->chunk) << " into "
          << ((col_params_->merge_op != nullptr) ? "tmp_chunk" : "chunk");
  Tensor* dst_tensor = (!rf->second_pass && (col_params_->merge_op != nullptr))
                           ? &rf->tmp_chunk
                           : &rf->chunk;
  col_ctx_->col_exec->RecvFromPeer(
      col_params_->instance.device_names[rf->recv_dev_idx],
      col_params_->instance.task_names[rf->recv_dev_idx],
      col_params_->task.is_local[rf->recv_dev_idx], recv_buf_key,
      col_ctx_->device, col_ctx_->op_ctx->op_device_context(),
      col_ctx_->op_ctx->output_alloc_attr(0), dst_tensor,
      col_ctx_->device_locality, rf->subdiv_idx, done);
}

string RingReducer::FieldState() {
  string s = strings::StrCat(
      "RingReducer ", strings::Hex(reinterpret_cast<uint64>(this)), " exec ",
      col_ctx_->exec_key, " step_id=", col_ctx_->step_id, " state of all ",
      rfv_.size(), " fields:");
  for (int i = 0; i < rfv_.size(); ++i) {
    s.append("\n");
    s.append(rfv_[i].DebugString());
  }
  return s;
}

bool RingReducer::RunAsyncParts() {
  // This function orchestrates RingReduce actions on behalf of a
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
    Notification note;
    Status s = gpu_info->default_context->ThenExecute(
        col_ctx_->device, gpu_info->stream, [&note]() { note.Notify(); });
    if (s.ok()) {
      note.WaitForNotification();
    } else {
      mutex_lock l(status_mu_);
      status_ =
          errors::Internal("Failed to dispatch ThenExecute in RingReducer");
      return false;
    }
  }

  int field_done_count = 0;
  int send_pending_count = 0;
  int recv_pending_count = 0;
  std::atomic<bool> aborted(false);

  // Loop until all RingFields have advanced to completion.
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
          CHECK_GT(recv_pending_count, 0);
          --recv_pending_count;
          if (!rf->second_pass) {
            rf->action = RF_REDUCE;
            Status s = collective_util::ComputeBinOp(
                col_ctx_->op_ctx, col_ctx_->op_params, col_ctx_->device,
                col_params_->merge_op.get(), &rf->chunk, &rf->tmp_chunk);
            if (!s.ok()) {
              aborted = true;
              StartAbort(s);
            }
          } else {
            rf->action = RF_SEND_READY;
          }
          break;
        case RF_REDUCE:
          if (!rf->second_pass && col_params_->final_op.get() && rf->is_final) {
            rf->action = RF_FINALIZE;
            group_size_tensor_ready_.WaitForNotification();
            Status s = collective_util::ComputeBinOp(
                col_ctx_->op_ctx, col_ctx_->op_params, col_ctx_->device,
                col_params_->final_op.get(), &rf->chunk, &group_size_tensor_);
            if (!s.ok()) {
              aborted = true;
              StartAbort(s);
            }
          } else {
            rf->action = RF_SEND_READY;
          }
          break;
        case RF_FINALIZE:
          rf->action = RF_DONE;
          break;
        case RF_SEND_READY:
          if (rf->do_send) {
            rf->action = RF_SEND;
            auto send_complete = [this, rf, &ready_queue, &aborted](Status s) {
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
          CHECK_GT(send_pending_count, 0);
          --send_pending_count;
          rf->action = RF_DONE;
          break;
        case RF_DONE:
          break;
      }
      if (rf->action == RF_DONE) {
        if (rf->second_pass) {
          ++field_done_count;
          break;  // from do while(!dispatched)
        } else {
          AdvanceToSecondPass(rf);
        }
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

  CHECK_EQ(send_pending_count, 0);
  CHECK_EQ(recv_pending_count, 0);

  VLOG(2) << this << " device=" << col_ctx_->device_name << " finish;"
          << " final value " << TensorDebugString(ca_->Value());
  return !aborted;
}

REGISTER_COLLECTIVE(RingReduce, RingReducer);

}  // namespace tensorflow
