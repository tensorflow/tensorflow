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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RING_REDUCER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RING_REDUCER_H_

#include <deque>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
class DeviceMgr;

// Ring-algorithm implementation of collective all-reduce.
class RingReducer {
 public:
  RingReducer(CollectiveExecutor* col_exec, const DeviceMgr* dev_mgr,
              OpKernelContext* ctx, OpKernelContext::Params* op_params,
              const CollectiveParams& col_params, const string& exec_key,
              int64 step_id, const Tensor* input, Tensor* output);

  virtual ~RingReducer() {}

  void Run(StatusCallback done);

 private:
  // Called when a bad status is received that implies we should terminate
  // execution and return a bad status.
  void StartAbort(const Status& s);
  void ContinueAfterInputCopy();
  void Finish(bool ok);
  Status ComputeBinOp(Device* device, OpKernel* op, Tensor* output,
                      Tensor* input);
  bool RunAsyncParts();

  // Used for executing a sub-operation, e.g. a merge_op instance, with
  // an OpKernelContext based on the one passed into this Op.
  class SubContext {
   public:
    OpKernelContext::Params sub_params_;
    gtl::InlinedVector<TensorValue, 4> sub_inputs_;
    gtl::InlinedVector<AllocatorAttributes, 4> sub_input_attr_;
    gtl::InlinedVector<DeviceContext*, 4> sub_input_dc_;
    // Used only for Binary and Unary Ops for which we require
    // the calculation to be in-place on the first input.
    int forward_from_ = 0;
    OpKernelContext* sub_ctx_;
    SubContext(OpKernelContext* ctx, OpKernelContext::Params* params,
               OpKernel* op, Tensor* output, Tensor* input);
    ~SubContext() { delete sub_ctx_; }
  };

  // Current status of a RingField
  enum RingFieldAction {
    RF_INIT = 0,    // Just initialized for a pass
    RF_RECV,        // Recv pending
    RF_REDUCE,      // Reduce pending
    RF_FINALIZE,    // FinalOp pending
    RF_SEND_READY,  // Ready to send
    RF_SEND,        // Send pending
    RF_DONE,        // No more work
  };

  // Tracks progress of actions on a single subfield of the entire tensor.
  struct RingField {
    int16 chunk_idx;     // major division index
    int16 subdiv_idx;    // minor division index
    int16 sc_idx;        // subchunk index
    int16 rank;          // rank within subdiv permutation
    int16 recv_dev_idx;  // dev from which value should be recv'd
    RingFieldAction action;
    bool second_pass;
    bool recv_is_remote = false;
    bool send_is_remote = false;
    bool do_send = false;   // is the value sent in this pass?
    bool do_recv = false;   // is the value recv'd in this pass?
    bool is_final = false;  // is the last field in the pass for this rank
    Tensor chunk;           // alias to field values
    Tensor tmp_chunk;
    Status status;
    string DebugString() const;
  };
  void AdvanceToSecondPass(RingField* rf);
  void InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                     int field_idx);
  void DispatchSend(RingField* rf, const StatusCallback& done);
  void DispatchRecv(RingField* rf, const StatusCallback& done);

  // For constructing log messages for debugging.
  string FieldState();
  string TensorDebugString(Tensor tensor);

  // Producer/Consumer Queue of RingField structs.
  class PCQueue {
   public:
    void Enqueue(RingField* rf);
    RingField* Dequeue();

   private:
    mutex pcq_mu_;
    condition_variable cv_;
    int waiter_count_ GUARDED_BY(pcq_mu_) = 0;
    std::deque<RingField*> deque_ GUARDED_BY(pcq_mu_);
  };

  CollectiveExecutor* col_exec_;        // Not owned
  const DeviceMgr* dev_mgr_;            // Not owned
  OpKernelContext* ctx_;                // Not owned
  OpKernelContext::Params* op_params_;  // Not owned
  const CollectiveParams& col_params_;
  const string exec_key_;
  const Tensor* input_;  // Not owned
  Tensor* output_;       // Not owned
  const int rank_;
  const int64 step_id_;
  const int group_size_;
  const int num_subdivs_;
  Tensor group_size_tensor_;
  Notification group_size_tensor_ready_;
  std::unique_ptr<CollectiveAdapter> ca_;
  StatusCallback done_;
  Device* device_;  // The device for which this instance labors
  const string device_name_;
  DeviceLocality device_locality_;

  mutex status_mu_;
  Status status_ GUARDED_BY(status_mu_);

  std::vector<RingField> rfv_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RING_REDUCER_H_
