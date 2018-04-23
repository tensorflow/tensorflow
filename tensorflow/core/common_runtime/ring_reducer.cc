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

#include "tensorflow/core/common_runtime/collective_rma_local.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

// Set true for greater intelligibility of debug mode log messages.
#define READABLE_KEYS false

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
    // TODO(tucker): Try out some kind of denser encoding, e.g. 128 bit hash.
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

RingReducer::RingReducer(CollectiveExecutor* col_exec, const DeviceMgr* dev_mgr,
                         OpKernelContext* ctx,
                         OpKernelContext::Params* op_params,
                         const CollectiveParams& col_params,
                         const string& exec_key, int64 step_id,
                         const Tensor* input, Tensor* output)
    : col_exec_(col_exec),
      dev_mgr_(dev_mgr),
      ctx_(ctx),
      op_params_(op_params),
      col_params_(col_params),
      exec_key_(exec_key),
      input_(input),
      output_(output),
      rank_(col_params.subdiv_rank[0]),
      step_id_(step_id),
      group_size_(col_params.group.group_size),
      num_subdivs_(static_cast<int>(
          col_params.instance.impl_details.subdiv_permutations.size())),
      done_(nullptr),
      device_(nullptr),
      device_name_(
          col_params_.instance.device_names[col_params_.default_rank]) {
  CHECK_GT(group_size_, 0);
  CHECK_GT(num_subdivs_, 0);
}

string RingReducer::TensorDebugString(Tensor tensor) {
  const DeviceBase::GpuDeviceInfo* gpu_device_info =
      ctx_->device()->tensorflow_gpu_device_info();
  if (gpu_device_info) {
    Tensor cpu_tensor(tensor.dtype(), tensor.shape());
    Notification note;
    gpu_device_info->default_context->CopyDeviceTensorToCPU(
        &tensor, "" /*tensor_name*/, device_, &cpu_tensor,
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

void RingReducer::Run(StatusCallback done) {
  done_ = std::move(done);

  // Get local execution device.
  if (VLOG_IS_ON(1)) {
    string buf;
    for (int r = 0; r < col_params_.instance.device_names.size(); ++r) {
      strings::StrAppend(&buf, "dev ", r, " : ",
                         col_params_.instance.device_names[r], "\n");
    }
    for (int sd = 0;
         sd < col_params_.instance.impl_details.subdiv_permutations.size();
         ++sd) {
      strings::StrAppend(&buf, "\nsubdiv ", sd, " perm: ");
      for (auto x : col_params_.instance.impl_details.subdiv_permutations[sd]) {
        strings::StrAppend(&buf, x, ", ");
      }
    }
    VLOG(1) << "RingReducer::Run for device " << device_name_
            << " default_rank " << col_params_.default_rank << "\n"
            << buf;
  }
  CHECK(dev_mgr_);
  Status status = dev_mgr_->LookupDevice(
      col_params_.instance.device_names[col_params_.default_rank], &device_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to find device "
               << col_params_.instance.device_names[col_params_.default_rank];
    for (auto d : dev_mgr_->ListDevices()) {
      LOG(ERROR) << "Available device " << d->name();
    }
    done_(status);
    return;
  }
  CHECK(device_);
  device_locality_ = device_->attributes().locality();

  VLOG(1) << this << " default_rank " << col_params_.default_rank << " cp "
          << &col_params_ << ": " << col_params_.ToString();

  // Start by copying input to output if they're not already the same, i.e. if
  // we're not computing in-place on the input tensor.
  if ((input_ != output_) &&
      (DMAHelper::base(input_) != DMAHelper::base(output_))) {
    CollectiveRemoteAccessLocal::MemCpyAsync(
        ctx_->input_device_context(0), ctx_->op_device_context(), device_,
        device_, ctx_->input_alloc_attr(0), ctx_->output_alloc_attr(0), input_,
        output_, [this](const Status& s) {
          if (!s.ok()) {
            done_(s);
          } else {
            ContinueAfterInputCopy();
          }
        });
  } else {
    ContinueAfterInputCopy();
  }
}

void RingReducer::ContinueAfterInputCopy() {
  AllocatorAttributes attr = ctx_->output_alloc_attr(0);
  ca_.reset(MakeCollectiveAdapter(output_, group_size_ * num_subdivs_,
                                  device_->GetAllocator(attr)));

  if (col_params_.final_op) {
    // Create an on-device scalar value from group_size_ that may be needed
    // later.
    // TODO(tucker): Cache and reuse across invocations? Or maybe the scalar
    // can be provided to the kernel in host memory?
    Tensor group_size_val = ca_->Scalar(group_size_);
    if (col_params_.group.device_type != "CPU") {
      group_size_tensor_ =
          ca_->Scalar(device_->GetAllocator(ctx_->input_alloc_attr(0)));
      DeviceContext* op_dev_ctx = ctx_->op_device_context();
      op_dev_ctx->CopyCPUTensorToDevice(&group_size_val, device_,
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
    col_exec_->StartAbort(s);
  }
}

void RingReducer::Finish(bool ok) {
  if (ok) {
    // Recover the output from the adaptor.
    ca_->ConsumeFinalValue(output_);
  }
  Status s;
  {
    mutex_lock l(status_mu_);
    s = status_;
  }
  done_(s);
}

RingReducer::SubContext::SubContext(OpKernelContext* ctx,
                                    OpKernelContext::Params* params,
                                    OpKernel* op, Tensor* output, Tensor* input)
    : sub_params_(*params),
      sub_inputs_({output, input}),
      sub_input_attr_({ctx->input_alloc_attr(0), ctx->input_alloc_attr(0)}),
      sub_input_dc_(
          {ctx->input_device_context(0), ctx->input_device_context(0)}) {
  sub_params_.op_kernel = op;
  sub_params_.inputs = &sub_inputs_;
  sub_params_.input_alloc_attrs = &sub_input_attr_;
  sub_params_.input_device_contexts = &sub_input_dc_;
  sub_params_.eigen_gpu_device = nullptr;
  sub_params_.ensure_eigen_gpu_device();
  sub_ctx_ = new OpKernelContext(&sub_params_, 1);
}

Status RingReducer::ComputeBinOp(Device* device, OpKernel* op, Tensor* output,
                                 Tensor* input) {
  // Prepare an OpKernelContext that is identical to that of the original Op
  // (i.e. the collective), except for the input output sizes and identities and
  // the Op itself.
  // TODO(tucker): Is it possible to cache and reuse these objects?  They're
  // mostly identical inside one device execution.
  std::unique_ptr<SubContext> sub_ctx(
      new SubContext(ctx_, op_params_, op, output, input));
  device->Compute(op, sub_ctx->sub_ctx_);
  return sub_ctx->sub_ctx_->status();
}

// At the beginning of the algorithm initialize a RingField struct for
// every independent field of the tensor.
void RingReducer::InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                                int field_idx) {
  // Note on field indexing: There are group_size_ devices in the
  // instance, implying the same number of chunks per tensor, where a
  // chunk is the unit of data transferred in a time step.  However, if
  // a device can simultaenously send data by 2 or more independent
  // channels we can speed up the transfer by subdividing chunks and
  // processing multiple subdivisions at once.  So the actual number
  // of RingFields is group_size_ * num_subdivs_.
  DCHECK_EQ(field_idx, (chunk_idx * num_subdivs_) + subdiv_idx);
  rf->chunk_idx = chunk_idx;
  rf->subdiv_idx = subdiv_idx;
  rf->sc_idx = field_idx;
  rf->rank = col_params_.subdiv_rank[subdiv_idx];
  rf->second_pass = false;
  rf->action = RF_INIT;
  // Recv from the device with preceding rank within the subdivision.
  int recv_from_rank = (rf->rank + (group_size_ - 1)) % group_size_;
  int send_to_rank = (rf->rank + 1) % group_size_;
  rf->recv_dev_idx = col_params_.instance.impl_details
                         .subdiv_permutations[subdiv_idx][recv_from_rank];
  int send_dev_idx = col_params_.instance.impl_details
                         .subdiv_permutations[subdiv_idx][send_to_rank];
  rf->recv_is_remote = !col_params_.task.is_local[rf->recv_dev_idx];
  rf->send_is_remote = !col_params_.task.is_local[send_dev_idx];
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
  string send_buf_key =
      RingReduceBufKey(exec_key_, rf->second_pass, rf->sc_idx, rf->rank);
  VLOG(3) << "DispatchSend rank=" << col_params_.default_rank << " send key "
          << send_buf_key << " chunk " << ca_->TBounds(rf->chunk) << " sc_idx "
          << rf->sc_idx;
  int send_to_rank = (rf->rank + 1) % group_size_;
  int send_to_dev_idx = col_params_.instance.impl_details
                            .subdiv_permutations[rf->subdiv_idx][send_to_rank];
  col_exec_->PostToPeer(col_params_.instance.device_names[send_to_dev_idx],
                        col_params_.instance.task_names[send_to_dev_idx],
                        send_buf_key, device_, ctx_->op_device_context(),
                        ctx_->output_alloc_attr(0), &rf->chunk,
                        device_locality_, done);
}

void RingReducer::DispatchRecv(RingField* rf, const StatusCallback& done) {
  CHECK(rf->do_recv);
  string recv_buf_key =
      RingReduceBufKey(exec_key_, rf->second_pass, rf->sc_idx,
                       (rf->rank + (group_size_ - 1)) % group_size_);
  VLOG(3) << "DispatchRecv rank=" << col_params_.default_rank << " recv key "
          << recv_buf_key << " chunk " << ca_->TBounds(rf->chunk) << " into "
          << ((col_params_.merge_op != nullptr) ? "tmp_chunk" : "chunk");
  Tensor* dst_tensor = (!rf->second_pass && (col_params_.merge_op != nullptr))
                           ? &rf->tmp_chunk
                           : &rf->chunk;
  col_exec_->RecvFromPeer(col_params_.instance.device_names[rf->recv_dev_idx],
                          col_params_.instance.task_names[rf->recv_dev_idx],
                          col_params_.task.is_local[rf->recv_dev_idx],
                          recv_buf_key, device_, ctx_->op_device_context(),
                          ctx_->output_alloc_attr(0), dst_tensor,
                          device_locality_, done);
}

string RingReducer::FieldState() {
  string s = strings::StrCat("RingReducer ",
                             strings::Hex(reinterpret_cast<uint64>(this)),
                             " exec ", exec_key_, " step_id=", step_id_,
                             " state of all ", rfv_.size(), " fields:");
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
  int field_done_count = 0;
  int send_pending_count = 0;
  int recv_pending_count = 0;
  std::atomic<bool> aborted(false);
  field_done_count = 0;
  send_pending_count = 0;
  recv_pending_count = 0;
  for (int chunk_idx = 0; chunk_idx < group_size_; ++chunk_idx) {
    for (int subdiv_idx = 0; subdiv_idx < num_subdivs_; ++subdiv_idx) {
      int rf_index = (chunk_idx * num_subdivs_) + subdiv_idx;
      InitRingField(&rfv_[rf_index], chunk_idx, subdiv_idx, rf_index);
      ready_queue.Enqueue(&rfv_[rf_index]);
    }
  }

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
              const bool bad_status = !s.ok();
              if (bad_status) aborted = true;
              ready_queue.Enqueue(rf);
              if (bad_status) StartAbort(s);
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
            Status s = ComputeBinOp(device_, col_params_.merge_op.get(),
                                    &rf->chunk, &rf->tmp_chunk);
            if (!s.ok()) {
              aborted = true;
              StartAbort(s);
            }
          } else {
            rf->action = RF_SEND_READY;
          }
          break;
        case RF_REDUCE:
          if (!rf->second_pass && col_params_.final_op.get() && rf->is_final) {
            rf->action = RF_FINALIZE;
            group_size_tensor_ready_.WaitForNotification();
            Status s = ComputeBinOp(device_, col_params_.final_op.get(),
                                    &rf->chunk, &group_size_tensor_);
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
              const bool bad_status = !s.ok();
              if (bad_status) aborted = true;
              ready_queue.Enqueue(rf);
              if (bad_status) StartAbort(s);
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
        default: {}  // Ignore any other actions
      }
    }
  }

  CHECK_EQ(send_pending_count, 0);
  CHECK_EQ(recv_pending_count, 0);

  VLOG(2) << this << " rank=" << rank_ << " finish;"
          << " final value " << TensorDebugString(ca_->Value());
  return !aborted;
}

}  // namespace tensorflow
